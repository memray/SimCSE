# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import logging
import copy
import transformers
from transformers import PreTrainedModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from simcse.models_v2 import ContrastiveLearningOutput, gather_norm
from src import contriever, dist_utils, utils

import torch.nn.functional as F

logger = logging.getLogger(__name__)

class MoCo(PreTrainedModel):
    def __init__(self, opt, model_opt, hfconfig):
        super(MoCo, self).__init__(hfconfig)
        self.opt = opt
        self.model_opt = model_opt
        self.config = hfconfig

        self.queue_size = opt.queue_size
        self.active_queue_size = opt.queue_size
        self.warmup_queue_size_ratio = opt.warmup_queue_size_ratio
        self.queue_update_steps = opt.queue_update_steps

        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k  #apply the encoder on keys in train mode
        self.sim_type = model_opt.sim_type
        self.cosine = nn.CosineSimilarity(dim=-1)

        self.pooling = opt.pooling
        self.num_q_view = opt.num_q_view
        self.num_k_view = opt.num_k_view
        retriever, tokenizer = self._load_retriever(model_opt.model_name_or_path,
                                                    pooling=opt.pooling,
                                                    random_init=opt.random_init,
                                                    num_view=self.num_q_view)
        self.tokenizer = tokenizer
        self.encoder_q = retriever

        self.indep_encoder_k = opt.indep_encoder_k
        if not self.indep_encoder_k:
            # MoCo
            self.encoder_k = copy.deepcopy(retriever)
            for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        else:
            # independent q/k encoder
            retriever, _ = self._load_retriever(model_opt.model_name_or_path,
                                                pooling=opt.pooling,
                                                random_init=opt.random_init,
                                                num_view=self.num_k_view)
            self.encoder_k = retriever

        # create the queue
        # update_strategy = ['fifo', 'priority']
        self.queue_strategy = opt.queue_strategy
        # queue.shape = (hdim, q_len * n_view)
        self.register_buffer("queue", torch.randn(opt.projection_size, self.queue_size * self.num_k_view))
        self.queue = nn.functional.normalize(self.queue, dim=0)  # L2 norm
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # https://github.com/SsnL/moco_align_uniform/blob/align_uniform/moco/builder.py
        self.align_unif_loss = opt.align_unif_loss
        # l_align. 2 align_weight = 3, align_alpha = 2 is used in SimCSE/moco_align_uniform by default
        self.align_weight = opt.align_weight
        self.align_alpha = opt.align_alpha
        # l_unif. unif_weight = 1, unif_t = 2  is used in SimCSE/moco_align_uniform by default
        self.unif_weight = opt.unif_weight
        self.unif_t = opt.unif_t

    def _load_retriever(self, model_id, pooling, random_init, num_view):
        cfg = utils.load_hf(transformers.AutoConfig, model_id)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_id)

        if random_init:
            retriever = contriever.Contriever(cfg)
        else:
            retriever = utils.load_hf(contriever.Contriever, model_id)

        if 'bert-' in model_id:
            if tokenizer.bos_token_id is None:
                tokenizer.bos_token = "[CLS]"
            if tokenizer.eos_token_id is None:
                tokenizer.eos_token = "[SEP]"

        retriever.pooling = pooling
        retriever.num_view = num_view
        retriever.cls_token_id = tokenizer.cls_token_id

        return retriever, tokenizer

    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, logits_neg):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())  # [B,D] -> [B*n_gpu,D]
        batch_size = keys.shape[0]

        if self.queue_strategy == 'fifo':
            ptr = int(self.queue_ptr)
            assert self.active_queue_size % batch_size == 0, f'{batch_size}, {self.active_queue_size}'  # for simplicity
            # replace the keys at ptr (dequeue and enqueue)
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.active_queue_size  # move pointer
            self.queue_ptr[0] = ptr
        elif self.queue_strategy == 'priority':
            _, smallest_ids = torch.topk(logits_neg.mean(dim=0), k=batch_size, largest=False)
            self.queue[:, smallest_ids] = keys.T
        else:
            raise NotImplementedError('Unsupported update_strategy: ', self.queue_strategy)

    def _compute_logits(self, q, k):
        if self.sim_type == 'dot':
            assert len(q.shape) == len(k.shape), 'shape(k)!=shape(q)'
            if self.pooling != 'multiview':  # single view
                l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B,H],[B,H] -> [B,1]
                l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
                logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
                # print('l_pos=', l_pos.shape, 'l_neg=', l_neg.shape)
                # print('logits=', logits.shape)
            else:  # multiple view, q and k are represented with multiple vectors
                bs = q.shape[0]
                emb_dim = q.shape[-1]
                _q = q.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
                _k = k.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
                l_pos = torch.einsum('nc,nc->n', [_q, _k]).unsqueeze(-1)  # [B*V,H],[B*V,H] -> [B*V,1]
                l_pos = l_pos.reshape(bs, -1).max(dim=1)[0]  # [B*V,1] -> [B,V] ->  [B,1]
                # or l_pos=torch.diag(torch.einsum('nvc,mvc->nvm', [q, k]).max(dim=1)[0], 0)
                _queue = self.queue.detach().permute(1, 0).reshape(-1,self.num_k_view,emb_dim)  # [H,Q*V] -> [Q*V,H] -> [Q,V,H]
                l_neg = torch.einsum('nvc,mvc->nvm', [q, _queue])  # [B,V,H],[Q,V,H] -> [B,V,Q]
                l_neg = l_neg.reshape(bs, -1).max(dim=1)[0]  # [B,V,Q] -> [B,Q]
                logits = torch.cat([l_pos, l_neg], dim=1)  # [B*V, 1+Q*V]
                logits = logits.reshape(q.shape[0], q.shape[1], -1)  # [B*V, 1+Q*V] -> [B,V,1+Q*V]
                logits = logits.max(dim=1)  # TODO, take rest views as negatives as well?
        elif self.sim_type == 'cosine':
            assert self.pooling != 'multiview', 'multi-view only works with dot-product (you can normalize vectors first)'
            l_pos = self.cosine(q, k).unsqueeze(-1)  # [B,1]
            l_neg = self.cosine(q.unsqueeze(1), self.queue.T.clone().detach())  # [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
        else:
            raise NotImplementedError('Not supported similarity:', self.sim_type)
        return logits, l_neg

    def forward(self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        sent_emb=False,
        is_query=False,
        output_hidden_states=True,
        return_dict=True,
        length=None,
        update_kencoder_queue=True,
        report_align_unif=False,
        report_metrics=False,
    ):
        if sent_emb:
            return self.sentemb_forward(
                is_query,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            return self.cl_forward(
                input_ids,
                attention_mask=attention_mask,
                update_kencoder_queue=update_kencoder_queue,
                report_align_unif=report_align_unif,
                report_metrics=report_metrics
            )

    def cl_forward(self, input_ids, attention_mask, stats_prefix='',
                   update_kencoder_queue=True, report_align_unif=False, report_metrics=False, **kwargs):
        q_tokens = input_ids[:, 1, :]
        q_mask = attention_mask[:, 1, :]
        k_tokens = input_ids[:, 0, :]
        k_mask = attention_mask[:, 0, :]

        iter_stats = {}
        bsz = q_tokens.size(0)

        q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: NxC
        if self.norm_query:
            q = nn.functional.normalize(q, dim=-1)

        # compute key features
        if not self.indep_encoder_k:
            with torch.no_grad():  # no gradient to keys
                if update_kencoder_queue:
                    self._momentum_update_key_encoder()  # update the key encoder

                if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                    self.encoder_k.eval()

                k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: NxC
                if self.norm_doc:
                    k = nn.functional.normalize(k, dim=-1)
        else:
            k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: NxC
            if self.norm_doc:
                k = nn.functional.normalize(k, dim=-1)

        logits, l_neg = self._compute_logits(q, k)
        logits = logits / self.temperature  # shape=[B,1+Q]

        # labels: positive key indicators
        labels = torch.zeros(bsz, dtype=torch.long).cuda()  # shape=[B]
        # contrastive, 1 positive out of Q negatives (in-batch examples are not used)
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + '/'
        iter_stats[f'{stats_prefix}cl_loss'] = torch.clone(loss)

        if report_metrics:
            predicted_idx = torch.argmax(logits, dim=-1)
            accuracy = 100 * (predicted_idx == labels).float()
            stdq = torch.std(q, dim=0).mean()  # q=[Q,D], std(q)=[D], std(q).mean()=[1]
            stdk = torch.std(k, dim=0).mean()
            stdqueue = torch.std(self.queue.T, dim=0).mean()
            iter_stats[f'{stats_prefix}accuracy'] = accuracy.mean()
            iter_stats[f'{stats_prefix}stdq'] = stdq
            iter_stats[f'{stats_prefix}stdk'] = stdk
            iter_stats[f'{stats_prefix}stdqueue'] = stdqueue
            iter_stats[f'{stats_prefix}queue_ptr'] = self.queue_ptr
            iter_stats[f'{stats_prefix}active_queue_size'] = self.active_queue_size

            doc_norm = gather_norm(k)
            query_norm = gather_norm(q)
            queue_norm = gather_norm(self.queue.T)
            iter_stats[f'{stats_prefix}doc_norm'] = doc_norm
            iter_stats[f'{stats_prefix}query_norm'] = query_norm
            iter_stats[f'{stats_prefix}queue_norm'] = queue_norm

        # loss of alignment/uniformity
        # lazyily computed & cached!
        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result
        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = (q @ self.queue).flatten()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result
        def get_q_dot_queue_splits():
            #  split queue to 4 parts, take a slice from each part, same size to q
            if not hasattr(get_q_dot_queue_splits, 'result'):
                get_q_dot_queue_splits.result = []
                for si in range(4):
                    queue_split = self.queue[:, self.active_queue_size // 4 * si: self.active_queue_size // 4 * (si + 1)]
                    get_q_dot_queue_splits.result.append(q @ queue_split)
            return get_q_dot_queue_splits.result
        def get_queue_dot_queue():
            if not hasattr(get_queue_dot_queue, 'result'):
                get_queue_dot_queue.result = torch.pdist(self.queue.T, p=2)
            assert get_queue_dot_queue.result._version == 0
            return get_queue_dot_queue.result

        if (report_metrics and report_align_unif) or self.align_unif_loss:
            if self.active_queue_size >= bsz * 4 and self.active_queue_size % 4 == 0:
                qqueuesplits_dists = get_q_dot_queue_splits()  # [q*queue_split]
                for i in range(4):
                    # mean/min/max distance between q and (k in queue_splits)
                    iter_stats[f'{stats_prefix}mean_dotproduct_q@queue-Q{i}'] = qqueuesplits_dists[i].mean()
                    iter_stats[f'{stats_prefix}min_dotproduct_q@queue-Q{i}'] = qqueuesplits_dists[i].min(dim=1)[0].mean()
                    iter_stats[f'{stats_prefix}max_dotproduct_q@queue-Q{i}'] = qqueuesplits_dists[i].max(dim=1)[0].mean()

            # l_align
            loss_align = None
            loss_unif = None
            if self.align_alpha is not None:
                if self.align_alpha == 2:
                    loss_align = (q - k).norm(dim=1, p=2).pow(2).mean()
                    # the line below is only equivalent to (q-k).norm(p=2, dim=1).pow(2).mean() when k/q are normalized!
                    # iter_stats[f'{stats_prefix}loss_align'] = 2 - 2 * get_q_bdot_k().mean()
                elif self.align_alpha == 1:
                    loss_align = (q - k).norm(dim=1, p=2).mean()
                else:
                    raise NotImplementedError('align_alpha other than 1/2 is not supported')
            # l_uniform
            if self.unif_t is not None:
                # [q*queue]
                qqueue_dists = get_q_dot_queue().pow(2)
                # add 2*self.unif_t to make it non-negative, near zero at optimum
                iter_stats[f'{stats_prefix}loss_unif_q@queue'] = \
                    2 * self.unif_t + qqueue_dists.mul(-self.unif_t).exp().mean().log()
                loss_unif = iter_stats[f'{stats_prefix}loss_unif_q@queue']

                if report_align_unif:
                    # [q*queue_splits]
                    if self.active_queue_size >= bsz * 4 and self.active_queue_size % 4 == 0:
                        qqueuesplits_dists = get_q_dot_queue_splits()  # [q*queue_split]
                        for i in range(4):
                            iter_stats[f'{stats_prefix}loss_unif_q@queue-Q{i}'] = \
                                2 * self.unif_t + qqueuesplits_dists[i].flatten().pow(2).mul(-self.unif_t).exp().mean().log()
                    else:
                        for i in range(4):
                            iter_stats[f'{stats_prefix}loss_unif_q@queue-Q{i}'] = torch.tensor(0.0)
                    # loss_unif_intra_batch is used in moco_align_uniform by default, where the negative pair distances include
                    # both the distance between samples in each batch and features in queue, as well as pairwise distances within each batch
                    # [q*q]
                    qself_dists = torch.pdist(q, p=2).pow(2)
                    iter_stats[f'{stats_prefix}loss_unif_q@q'] = \
                        2 * self.unif_t + qself_dists.mul(-self.unif_t).exp().mean().log()
                    # [queue*queue]
                    # queueself_dists = get_queue_dot_queue().pow(2)
                    # iter_stats[f'{stats_prefix}loss_unif_queue@queue'] = \
                    #     2 * self.unif_t + queueself_dists.mul(-self.unif_t).exp().mean().log()
            if self.align_unif_loss:
                if self.align_weight > 0.0:
                    iter_stats[f'{stats_prefix}loss_align'] = self.align_weight * loss_align
                    loss += iter_stats[f'{stats_prefix}loss_align']
                if self.unif_weight > 0.0:
                    iter_stats[f'{stats_prefix}loss_unif'] = self.unif_weight * loss_unif
                    loss += iter_stats[f'{stats_prefix}loss_unif']
            else:
                if loss_align: iter_stats[f'{stats_prefix}loss_align'] = loss_align
                if loss_unif: iter_stats[f'{stats_prefix}loss_unif'] = loss_align

        iter_stats[f'{stats_prefix}loss'] = loss

        if update_kencoder_queue:
            self._dequeue_and_enqueue(k, l_neg)

        return ContrastiveLearningOutput(
            loss=loss,
            specific_losses=iter_stats
        )

    def sentemb_forward(
        self,
        is_query,
        input_ids=None,
        attention_mask=None,
    ):
        encoder = self.encoder_q
        pooler_output = encoder(input_ids, attention_mask=attention_mask)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
        )
