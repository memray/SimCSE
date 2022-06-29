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
        self.momentum = opt.momentum
        self.temperature = opt.temperature
        self.label_smoothing = opt.label_smoothing
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.moco_train_mode_encoder_k = opt.moco_train_mode_encoder_k #apply the encoder on keys in train mode
        self.sim_type = model_opt.sim_type
        self.cosine = nn.CosineSimilarity(dim=-1)

        retriever, tokenizer = self._load_retriever(model_opt.model_name_or_path,
                                                    pooling=opt.pooling, random_init=opt.random_init)
        
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
                                                pooling=opt.pooling, random_init=opt.random_init)
            self.encoder_k = retriever

        # create the queue
        self.register_buffer("queue", torch.randn(opt.projection_size, self.queue_size))
        self.queue = nn.functional.normalize(self.queue, dim=0)  # L2 norm

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # https://github.com/SsnL/moco_align_uniform/blob/align_uniform/moco/builder.py
        # l_align
        align_weight = 3
        align_alpha = 2  # 2 is used in SimCSE/moco_align_uniform by default
        self.align_weight = align_weight
        self.align_alpha = align_alpha
        # l_unif
        unif_weight = 1
        unif_t = 2  # 2 is used in SimCSE/moco_align_uniform by default
        self.unif_weight = unif_weight
        self.unif_t = unif_t


    def _load_retriever(self, model_id, pooling, random_init):
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

        retriever.config.pooling = pooling

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
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())  # [B,D] -> [B*n_gpu,D]

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f'{batch_size}, {self.queue_size}'  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def _compute_logits(self, q, k):
        if self.sim_type == 'dot':
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)  # [B,D],[D,B] -> [B,1]
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])  # [B,D],[D,Q] -> [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
        elif self.sim_type == 'cosine':
            l_pos = self.cosine(q, k).unsqueeze(-1)  # [B,1]
            l_neg = self.cosine(q.unsqueeze(1), self.queue.T.clone().detach())  # [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
        else:
            raise NotImplementedError('Not supported similarity:', self.sim_type)
        return logits

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
                report_align_unif=report_align_unif
            )

    def cl_forward(self, input_ids, attention_mask, stats_prefix='',
                   update_kencoder_queue=True, report_align_unif=False, **kwargs):
        q_tokens = input_ids[:, 1, :]
        q_mask = attention_mask[:, 1, :]
        k_tokens = input_ids[:, 0, :]
        k_mask = attention_mask[:, 0, :]

        iter_stats = {}
        bsz = q_tokens.size(0)

        q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: NxC
        if self.norm_query:
            q = nn.functional.normalize(q, dim=1)

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

        logits = self._compute_logits(q, k) / self.temperature  # shape=[B,1+Q]

        # labels: positive key indicators
        labels = torch.zeros(bsz, dtype=torch.long).cuda()  # shape=[B]
        # contrastive, 1 positive out of Q negatives (in-batch examples are not used)
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)

        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + '/'
        iter_stats[f'{stats_prefix}loss'] = loss

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
        def get_queue_dot_queue():
            if not hasattr(get_queue_dot_queue, 'result'):
                get_queue_dot_queue.result = torch.pdist(self.queue.T, p=2)
            assert get_queue_dot_queue.result._version == 0
            return get_queue_dot_queue.result
        if report_align_unif:
            # l_align
            if self.align_alpha is not None:
                if self.align_alpha == 2:
                    iter_stats[f'{stats_prefix}loss_align'] = (q - k).norm(dim=1, p=2).pow(2).mean()
                    # the line below is only equivalent to (q-k).norm(p=2, dim=1).pow(2).mean() when k/q are normalized!
                    # iter_stats[f'{stats_prefix}loss_align'] = 2 - 2 * get_q_bdot_k().mean()
                elif self.align_alpha == 1:
                    iter_stats[f'{stats_prefix}loss_align'] = (q - k).norm(dim=1, p=2).mean()
                else:
                    raise NotImplementedError('align_alpha other than 1/2 is not supported')
            # l_uniform
            if self.unif_t is not None:
                qqueue_dists = get_q_dot_queue().pow(2)  # [q*queue]
                # add 2*self.unif_t to make it non-negative, near zero at optimum
                iter_stats[f'{stats_prefix}loss_unif_q@queue'] = \
                    2 * self.unif_t + qqueue_dists.mul(-self.unif_t).exp().mean().log()
                # loss_unif_intra_batch is used in moco_align_uniform by default, where the negative pair distances include
                # both the distance between samples in each batch and features in queue, as well as pairwise distances within each batch
                qself_dists = torch.pdist(q, p=2).pow(2)  # [q*q]
                iter_stats[f'{stats_prefix}loss_unif_q@q'] = \
                    2 * self.unif_t + qself_dists.mul(-self.unif_t).exp().mean().log()
                # queueself_dists = get_queue_dot_queue().pow(2)  # [queue*queue]
                # iter_stats[f'{stats_prefix}loss_unif_queue@queue'] = \
                #     2 * self.unif_t + queueself_dists.mul(-self.unif_t).exp().mean().log()

        if update_kencoder_queue:
            self._dequeue_and_enqueue(k)

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
