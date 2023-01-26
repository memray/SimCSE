# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import torch
import torch.nn as nn
import logging
import copy
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.model.biencoder import BiEncoder
from src.utils import dist_utils
from src.utils.model_utils import load_retriever, gather_norm, ContrastiveLearningOutput, mix_two_inputs
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class MoCo(BiEncoder):
    def __init__(self, moco_config, hf_config):
        super(MoCo, self).__init__(moco_config, hf_config)
        self.moco_config = moco_config
        self.hf_config = hf_config

        self.model_name_or_path = getattr(moco_config, 'model_name_or_path', 'bert-base-uncased')
        self.use_inbatch_negatives = getattr(moco_config, 'use_inbatch_negatives', False)
        self.chunk_query_ratio = getattr(moco_config, 'chunk_query_ratio', 0.0)
        self.num_extra_pos = getattr(moco_config, 'num_extra_pos', 0)
        self.neg_indices = getattr(moco_config, 'neg_indices', None)
        self.queue_size = moco_config.queue_size
        self.q_queue_size = getattr(moco_config, 'q_queue_size', 0)
        self.active_queue_size = moco_config.queue_size
        self.warmup_queue_size_ratio = moco_config.warmup_queue_size_ratio  # probably
        self.queue_update_steps = moco_config.queue_update_steps  # not useful

        self.momentum = moco_config.momentum
        self.temperature = moco_config.temperature
        self.label_smoothing = moco_config.label_smoothing
        self.norm_doc = moco_config.norm_doc
        self.norm_query = moco_config.norm_query
        self.moco_train_mode_encoder_k = moco_config.moco_train_mode_encoder_k  #apply the encoder on keys in train mode

        self.pooling = moco_config.pooling
        self.sim_metric = getattr(moco_config, 'sim_metric', 'dot')
        self.symmetric_loss = getattr(moco_config, 'symmetric_loss', False)
        self.qk_norm_diff_lambda = getattr(moco_config, 'symmetric_loss', 0.0)
        self.cosine = nn.CosineSimilarity(dim=-1)

        # self.num_q_view = moco_config.num_q_view
        # self.num_k_view = moco_config.num_k_view
        retriever, tokenizer = load_retriever(self.model_name_or_path, moco_config.pooling, hf_config)
        self.tokenizer = tokenizer
        self.encoder_q = retriever

        self.indep_encoder_k = getattr(moco_config, 'indep_encoder_k', False)
        if not self.indep_encoder_k:
            # MoCo
            self.encoder_k = copy.deepcopy(retriever)
            for param_q, param_k in zip(self.encoder_q.model.parameters(), self.encoder_k.model.parameters()):
                param_k.data.copy_(param_q.data)
                param_k.requires_grad = False
        else:
            # independent q/k encoder
            retriever, _ = load_retriever(moco_config.model_name_or_path, moco_config.pooling, hf_config)
            self.encoder_k = retriever

        # create the queue
        # update_strategy = ['fifo', 'priority']
        # self.queue_strategy = moco_config.queue_strategy
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        if self.queue_size > 0:
            self.register_buffer("queue_k", torch.randn(moco_config.projection_size, self.queue_size))
            self.queue_k = nn.functional.normalize(self.queue_k, dim=0)  # L2 norm
        else:
            self.queue_k = None
        if self.q_queue_size > 0:
            self.register_buffer("queue_q", torch.randn(moco_config.projection_size, self.q_queue_size))
            self.queue_q = nn.functional.normalize(self.queue_q, dim=0)  # L2 norm
        else:
            self.queue_q = None

        # https://github.com/SsnL/moco_align_uniform/blob/align_uniform/moco/builder.py
        self.align_unif_loss = moco_config.align_unif_loss
        self.align_unif_cancel_step = getattr(moco_config, 'align_unif_cancel_step', 0)
        # l_align. 2 align_weight = 3, align_alpha = 2 is used in SimCSE/moco_align_uniform by default
        self.align_weight = moco_config.align_weight
        self.align_alpha = moco_config.align_alpha
        # l_unif. unif_weight = 1, unif_t = 2  is used in SimCSE/moco_align_uniform by default
        self.unif_weight = moco_config.unif_weight
        self.unif_t = moco_config.unif_t

    def _compute_loss(self, q, k, queue=None):
        bsz = q.shape[0]
        if self.use_inbatch_negatives:
            labels = torch.arange(0, bsz, dtype=torch.long, device=q.device)  # [B]
            labels = labels + dist_utils.get_rank() * len(k)  # positive indices offset=local_rank*B
            logits = self._compute_logits_inbatch(q, k, queue)  # shape=[B,k*n_gpu] or [B,k*n_gpu+Q]
        else:
            assert self.queue_k is not None
            logits = self._compute_logits(q, k)  # shape=[B,1+Q]
            labels = torch.zeros(bsz, dtype=torch.long).cuda()  # shape=[B]
        # contrastive, 1 positive out of Q negatives (in-batch examples are not used)
        logits = logits / self.temperature
        loss = torch.nn.functional.cross_entropy(logits, labels, label_smoothing=self.label_smoothing)
        # print(q_tokens.device, 'q_shape=', q_tokens.shape, 'k_shape=', k_tokens.shape)
        # print('loss=', loss.item(), 'logits.mean=', logits.mean().item())
        return loss, logits, labels

    def _compute_logits_inbatch(self, q, k, queue):
        k = k.contiguous()
        gather_k = dist_utils.gather(k)  # [B,H] -> [B*n_gpu,H]
        if self.sim_metric == 'dot':
            logits = torch.einsum("ih,jh->ij", q, gather_k)  # # [B,H] x [B*n_gpu,H] = [B,B*n_gpu]
            if queue is not None:
                l_neg = torch.einsum('bh,hn->bn', [q, queue.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B,B*n_gpu]+[B,Q] = [B,B*n_gpu+Q]
        else:
            # cast to q.shape=[B,1,H], gather_k.shape=[B*n_device,H] -> [B,B*n_device]
            logits = self.cosine(q.unsqueeze(1), gather_k)
            if queue is not None:
                # cast to q.shape=[B,1,H], queue.shape=[Q,H] -> [B,Q]
                l_neg = self.cosine(q.unsqueeze(1), queue.T.clone().detach())  # [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B, B*n_device+Q]
        return logits

    def _compute_logits(self, q, k):
        if self.sim_metric == 'dot':
            assert len(q.shape) == len(k.shape), 'shape(k)!=shape(q)'
            l_pos = torch.einsum('nh,nh->n', [q, k]).unsqueeze(-1)  # [B,H],[B,H] -> [B,1]
            l_neg = torch.einsum('nh,hk->nk', [q, self.queue_k.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
            # print('l_pos=', l_pos.shape, 'l_neg=', l_neg.shape)
            # print('logits=', logits.shape)
        elif self.sim_metric == 'cosine':
            l_pos = self.cosine(q, k).unsqueeze(-1)  # [B,1]
            l_neg = self.cosine(q.unsqueeze(1), self.queue_k.T.clone().detach())  # [B,Q]
            logits = torch.cat([l_pos, l_neg], dim=1)  # [B, 1+Q]
        else:
            raise NotImplementedError('Not supported similarity:', self.sim_metric)
        return logits

    def _compute_logits_multiview(self, q, k):
        raise NotImplementedError()
        assert self.sim_metric == 'dot'
        assert len(q.shape) == len(k.shape), 'shape(k)!=shape(q)'
        # multiple view, q and k are represented with multiple vectors
        bs = q.shape[0]
        emb_dim = q.shape[-1]
        _q = q.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
        _k = k.reshape(-1, emb_dim)  # [B,V,H]->[B*V,H]
        l_pos = torch.einsum('nc,nc->n', [_q, _k]).unsqueeze(-1)  # [B*V,H],[B*V,H] -> [B*V,1]
        l_pos = l_pos.reshape(bs, -1).max(dim=1)[0]  # [B*V,1] -> [B,V] ->  [B,1]
        # or l_pos=torch.diag(torch.einsum('nvc,mvc->nvm', [q, k]).max(dim=1)[0], 0)
        _queue = self.queue_k.detach().permute(1, 0).reshape(-1, self.num_k_view, emb_dim)  # [H,Q*V] -> [Q*V,H] -> [Q,V,H]
        l_neg = torch.einsum('nvc,mvc->nvm', [q, _queue])  # [B,V,H],[Q,V,H] -> [B,V,Q]
        l_neg = l_neg.reshape(bs, -1).max(dim=1)[0]  # [B,V,Q] -> [B,Q]
        logits = torch.cat([l_pos, l_neg], dim=1)  # [B*V, 1+Q*V]
        logits = logits.reshape(q.shape[0], q.shape[1], -1)  # [B*V, 1+Q*V] -> [B,V,1+Q*V]
        logits = logits.max(dim=1)  # TODO, take rest views as negatives as well?
        return logits


    def forward(self,
        input_ids=None,
        attention_mask=None,
        data=None,
        sent_emb=False,
        is_query=False,
        update_kencoder_queue=True,
        report_align_unif=False,
        report_metrics=False,
        **kwargs
    ):
        if sent_emb:
            return self.infer_forward(
                is_query,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            return self.train_forward(
                data=data,
                update_kencoder_queue=update_kencoder_queue,
                report_align_unif=report_align_unif,
                report_metrics=report_metrics
            )

    def train_forward(self, data, stats_prefix='',
                      update_kencoder_queue=True, report_align_unif=False, report_metrics=False, **kwargs):
        q_tokens = data['queries']['input_ids']
        q_mask = data['queries']['attention_mask']
        k_tokens = data['docs']['input_ids']
        k_mask = data['docs']['attention_mask']

        # print('q_tokens.shape=', q_tokens.shape)
        # print('k_tokens.shape=', k_tokens.shape)

        # prone to OOM, and size is messy, commented out for now
        # if self.neg_indices is not None and max(self.neg_indices) < len(input_ids):
        #     k_tokens = torch.cat([k_tokens]+[input_ids[i] for i in self.neg_indices])
        #     k_mask = torch.cat([k_mask]+[attention_mask[i] for i in self.neg_indices])

        iter_stats = {}
        bsz = q_tokens.size(0)

        # 1. compute key
        if self.indep_encoder_k:
            k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: B,H
        else:
            with torch.no_grad():  # no gradient to keys
                if update_kencoder_queue:
                    self._momentum_update_key_encoder()  # update the key encoder

                if not self.encoder_k.training and not self.moco_train_mode_encoder_k:
                    self.encoder_k.eval()

                k = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask)  # keys: B,H
        if self.norm_doc:
            k = nn.functional.normalize(k, dim=-1)
        # 2. compute query
        if self.q_extract and 'random_chunks' in data:
            # print(chunk_tokens.shape)
            chunk_tokens = data['random_chunks']['input_ids']
            chunk_mask = data['random_chunks']['attention_mask']
            num_chunk = chunk_tokens.shape[0] // bsz
            if self.q_extract == 'self-dot':
                with torch.no_grad():
                    q_cand = self.encoder_q(input_ids=chunk_tokens, attention_mask=chunk_mask).reshape(bsz, -1, k.shape[1])  # queries: B,num_chunk,H
                    chunk_score = torch.einsum('bch,bh->bc', q_cand, k).detach()  # B,num_chunk
                    chunk_idx = torch.argmax(chunk_score, dim=1)  # [B]
            elif self.q_extract == 'bm25':
                chunk_idx = self.q_extract_model.batch_rank_chunks(batch_docs=data['contexts_str'],
                                                                  batch_chunks=[data['random_chunks_str'][i*num_chunk: (i+1)*num_chunk]
                                                                         for i in range(len(data['docs_str']))])
            elif self.q_extract_model:
                chunk_idx = self.rank_chunks_by_ext_model(docs=data['contexts_str'], chunks=data['random_chunks_str'])
            else:
                raise NotImplementedError
            c_tokens = torch.stack([chunks[cidx.item()] for chunks, cidx in zip(chunk_tokens.reshape(bsz, num_chunk, -1), chunk_idx)])
            c_mask = torch.stack([chunks[cidx.item()] for chunks, cidx in zip(chunk_mask.reshape(bsz, num_chunk, -1), chunk_idx)])
            if self.q_extract_ratio is not None:
                c_tokens, c_mask = mix_two_inputs(c_tokens, c_mask, q_tokens, q_mask, input0_ratio=self.q_extract_ratio)
            q = self.encoder_q(input_ids=c_tokens, attention_mask=c_mask)  # queries: B,H
        else:
            q = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: B,H
        if self.norm_query: q = nn.functional.normalize(q, dim=-1)

        # 3. apply projectors
        if self.q_mlp:  q = self.q_mlp(q)
        if self.k_mlp:  k = self.k_mlp(k)

        # 4. apply predictor (q/k interaction)
        # 5. computer loss
        loss, logits, labels = self._compute_loss(q, k.detach(), self.queue_k)
        if self.symmetric_loss:
            _loss, _logits, _labels = self._compute_loss(k, q.detach(), self.queue_q)
            loss = (loss + _loss) / 2
        if self.qk_norm_diff_lambda > 0:
            norm_loss = self._compute_norm_loss(q, k)
            loss += self.qk_norm_diff_lambda * norm_loss
        else:
            norm_loss = 0.0
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + '/'
        iter_stats[f'{stats_prefix}cl_loss'] = torch.clone(loss)
        if report_metrics:
            predicted_idx = torch.argmax(logits, dim=-1)
            accuracy = 100 * (predicted_idx == labels).float()
            stdq = torch.std(q, dim=0).mean()  # q=[Q,H], std(q)=[H], std(q).mean()=[1]
            stdk = torch.std(k, dim=0).mean()
            stdqueue_k = torch.std(self.queue_k.T, dim=0).mean() if self.queue_k is not None else 0.0
            stdqueue_q = torch.std(self.queue_q.T, dim=0).mean() if self.queue_q is not None else 0.0
            # print(accuracy.detach().cpu().numpy())
            iter_stats[f'{stats_prefix}accuracy'] = accuracy.mean()
            iter_stats[f'{stats_prefix}stdq'] = stdq
            iter_stats[f'{stats_prefix}stdk'] = stdk
            iter_stats[f'{stats_prefix}stdqueue_k'] = stdqueue_k
            iter_stats[f'{stats_prefix}stdqueue_q'] = stdqueue_q

            doc_norm = gather_norm(k)
            query_norm = gather_norm(q)
            iter_stats[f'{stats_prefix}doc_norm'] = doc_norm
            iter_stats[f'{stats_prefix}query_norm'] = query_norm
            iter_stats[f'{stats_prefix}norm_diff'] = torch.abs(doc_norm - query_norm)

            queue_k_norm = gather_norm(self.queue_k.T) if self.queue_k is not None else 0.0
            queue_q_norm = gather_norm(self.queue_q.T) if self.queue_q is not None else 0.0
            iter_stats[f'{stats_prefix}queue_ptr'] = self.queue_ptr
            iter_stats[f'{stats_prefix}active_queue_size'] = self.active_queue_size
            iter_stats[f'{stats_prefix}queue_k_norm'] = queue_k_norm
            iter_stats[f'{stats_prefix}queue_q_norm'] = queue_q_norm
            iter_stats[f'{stats_prefix}norm_loss'] = norm_loss

            # compute on each device, only dot-product
            iter_stats[f'{stats_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', q, k).detach().mean()
            iter_stats[f'{stats_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', q, k).detach().fill_diagonal_(0).sum() / (bsz*bsz-bsz)
            if self.queue_k is not None:
                iter_stats[f'{stats_prefix}q@queue_neg_score'] = torch.einsum('id,jd->ij', q, self.queue_k.T).detach().mean()

        # loss of alignment/uniformity
        # lazyily computed & cached!
        def get_q_bdot_k():
            if not hasattr(get_q_bdot_k, 'result'):
                get_q_bdot_k.result = (q * k).sum(dim=1)
            assert get_q_bdot_k.result._version == 0
            return get_q_bdot_k.result
        def get_q_dot_queue():
            if not hasattr(get_q_dot_queue, 'result'):
                get_q_dot_queue.result = (q @ self.queue_k.detach()).flatten()
            assert get_q_dot_queue.result._version == 0
            return get_q_dot_queue.result
        def get_q_dot_queue_splits():
            # split queue to 4 parts, take a slice from each part, same size to q
            # Q0 is oldest portion, Q3 is latest
            if not hasattr(get_q_dot_queue_splits, 'result'):
                get_q_dot_queue_splits.result = []
                # organize the key queue to make it in the order of oldest -> latest
                queue_old2new = torch.concat([self.queue_k.clone()[:, int(self.queue_ptr): self.active_queue_size],
                                              self.queue_k.clone()[:, :int(self.queue_ptr)]], dim=1)
                for si in range(4):
                    queue_split = queue_old2new[:, self.active_queue_size // 4 * si: self.active_queue_size // 4 * (si + 1)]
                    get_q_dot_queue_splits.result.append(q @ queue_split)
            return get_q_dot_queue_splits.result
        def get_queue_dot_queue():
            if not hasattr(get_queue_dot_queue, 'result'):
                get_queue_dot_queue.result = torch.pdist(self.queue_k.T, p=2)
            assert get_queue_dot_queue.result._version == 0
            return get_queue_dot_queue.result

        if (report_metrics and report_align_unif) or self.align_unif_loss:
            if self.queue_q is not None and self.active_queue_size >= bsz * 4 and self.active_queue_size % 4 == 0:
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
                    # the line below is equivalent to (q-k).norm(p=2, dim=1).pow(2).mean() only when k/q are normalized!
                    # iter_stats[f'{stats_prefix}loss_align'] = 2 - 2 * get_q_bdot_k().mean()
                elif self.align_alpha == 1:
                    loss_align = (q - k).norm(dim=1, p=2).mean()
                else:
                    raise NotImplementedError('align_alpha other than 1/2 is not supported')
            # l_uniform
            if self.unif_t is not None:
                # [q*queue]
                if self.queue_k is not None:
                    qqueue_dists = get_q_dot_queue().pow(2)
                    # add 2*self.unif_t to make it non-negative, near zero at optimum
                    loss_unif = 2 * self.unif_t + qqueue_dists.mul(-self.unif_t).exp().mean().log()
                    iter_stats[f'{stats_prefix}loss_unif_q@queue'] = loss_unif.item()
                '''
                if report_align_unif:
                    # [q*queue_splits]
                    if self.queue_q is not None and self.active_queue_size >= bsz * 4 and self.active_queue_size % 4 == 0:
                        qqueuesplits_dists = get_q_dot_queue_splits()  # [q*queue_split]
                        for i in range(4):
                            iter_stats[f'{stats_prefix}loss_unif_q@queue-Q{i}'] = \
                                2 * self.unif_t + qqueuesplits_dists[i].flatten().pow(2).mul(-self.unif_t).exp().mean().log()
                    else:
                        for i in range(4):
                            iter_stats[f'{stats_prefix}loss_unif_q@queue-Q{i}'] = 0.0
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
            '''
            if self.align_unif_loss:
                if self.align_weight > 0.0:
                    loss += self.align_weight * loss_align
                if self.unif_weight > 0.0:
                    loss += self.unif_weight * loss_unif
            if loss_align:
                iter_stats[f'{stats_prefix}loss_align'] = loss_align
            if loss_unif:
                iter_stats[f'{stats_prefix}loss_unif'] = loss_unif

        iter_stats[f'{stats_prefix}loss'] = loss

        if update_kencoder_queue:
            # print('Before \t\t queue_ptr', int(self.queue_ptr), 'queue_k.shape=', self.queue_k.shape)
            if self.queue_k is not None: self._dequeue_and_enqueue(k, self.queue_k)
            if self.queue_q is not None: self._dequeue_and_enqueue(q, self.queue_q)
            if self.queue_k is not None or self.queue_q is not None:
                ptr = int(self.queue_ptr)
                ptr = (ptr + bsz * dist_utils.get_world_size()) % self.active_queue_size  # move pointer
                self.queue_ptr[0] = ptr
            # print('After \t\t queue_ptr', int(self.queue_ptr), 'self.queue_k.shape=', self.queue_k.shape)
        return ContrastiveLearningOutput(
            loss=loss,
            specific_losses=iter_stats
        )

    def infer_forward(
        self,
        is_query,
        input_ids=None,
        attention_mask=None,
    ):
        encoder = self.encoder_q
        # self.encoder_k is momentum during training, so be cautious
        if self.indep_encoder_k and not is_query:
            encoder = self.encoder_k
        pooler_output = encoder(input_ids, attention_mask=attention_mask)
        if is_query and self.q_mlp:
            pooler_output = self.q_mlp(pooler_output)
        if not is_query and self.k_mlp:
            pooler_output = self.k_mlp(pooler_output)
        if is_query and self.norm_query:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)
        elif not is_query and self.norm_doc:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
        )

