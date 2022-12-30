# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src.model.biencoder import BiEncoder
from src.utils import dist_utils
from src.utils.model_utils import load_retriever, gather_norm, ContrastiveLearningOutput, mix_two_inputs

logger = logging.getLogger(__name__)


class InBatch(BiEncoder):
    def __init__(self, moco_config, hf_config=None):
        super(InBatch, self).__init__(moco_config, hf_config)

        self.moco_config = moco_config
        self.hf_config = hf_config
        self.indep_encoder_k = moco_config.indep_encoder_k
        self.neg_names = moco_config.neg_names  # the indices of data for additional negative examples
        self.projection_size = moco_config.projection_size

        self.sim_metric = getattr(moco_config, 'sim_metric', 'dot')
        self.norm_doc = moco_config.norm_doc
        self.norm_query = moco_config.norm_query
        self.label_smoothing = moco_config.label_smoothing
        retriever, tokenizer = load_retriever(moco_config.model_name_or_path, pooling=moco_config.pooling, hf_config=hf_config)
        self.tokenizer = tokenizer
        self.encoder_q = retriever
        if self.indep_encoder_k:
            retriever, tokenizer = load_retriever(moco_config.model_name_or_path, pooling=moco_config.pooling, hf_config=hf_config)
            self.encoder_k = retriever
        else:
            self.encoder_k = self.encoder_q

        self.queue_size = moco_config.queue_size
        if self.queue_size > 0:
            self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
            self.register_buffer("queue_k", torch.randn(self.projection_size, self.queue_size))
            self.queue_k = nn.functional.normalize(self.queue_k, dim=0)  # L2 norm
        else:
            self.queue_ptr = None
            self.queue_k = None

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

    def _compute_logits(self, q, gather_k, queue):
        if self.sim_metric == 'dot':
            logits = torch.einsum("id,jd->ij", q, gather_k)  # # [B,D] x [B*n_gpu,D] = [B,B*n_gpu]
            if queue is not None:
                l_neg = torch.einsum('bd,dn->bn', [q, queue.clone().detach()])  # [B,H],[H,Q] -> [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B,B*n_gpu]+[B,Q] = [B,B*n_gpu+Q]
        else:
            # cast to q.shape=[B,1,H], gather_k.shape=[B*n_device,H] -> [B,B*n_device]
            logits = self.cosine(q.unsqueeze(1), gather_k)
            if queue is not None:
                # cast to q.shape=[B,1,H], queue.shape=[Q,H] -> [B,Q]
                l_neg = self.cosine(q.unsqueeze(1), queue.T.clone().detach())  # [B,Q]
                logits = torch.cat([logits, l_neg], dim=1)  # [B, B*n_device+Q]
        return logits

    @torch.no_grad()
    def _dequeue_and_enqueue(self, gather_k, queue):
        batch_size = gather_k.shape[0]
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0, f'{batch_size}, {self.queue_size}'  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = gather_k.T
        # move pointer
        ptr = int(self.queue_ptr)
        ptr = (ptr + batch_size) % self.queue_size
        self.queue_ptr[0] = ptr

    def train_forward(self, data, stats_prefix='', report_metrics=False, **kwargs):
        q_tokens = data['queries']['input_ids']
        q_mask = data['queries']['attention_mask']
        k_tokens = data['docs']['input_ids']
        k_mask = data['docs']['attention_mask']
        bsz = len(q_tokens)

        # print('q_tokens.shape=', q_tokens.shape, '\n')
        # print('k_tokens.shape=', k_tokens.shape, '\n')

        # 1. compute key
        kemb = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask).contiguous()
        if self.norm_doc:
            kemb = nn.functional.normalize(kemb, dim=-1)

        # qemb = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)

        # 2. compute query
        if self.q_extract and 'random_chunks' in data:
            chunk_tokens = data['random_chunks']['input_ids']
            chunk_mask = data['random_chunks']['attention_mask']
            num_chunk = chunk_tokens.shape[0] // bsz
            # print(chunk_tokens.shape)
            if self.q_extract == 'self-dot':
                with torch.no_grad():
                    q_cand = self.encoder_q(chunk_tokens, chunk_mask).reshape(bsz, -1, kemb.shape[1])  # queries: B,num_chunk,H
                    chunk_score = torch.einsum('bch,bh->bc', q_cand, kemb).detach()  # B,num_chunk
                    chunk_idx = torch.argmax(chunk_score, dim=1)  # [B]
            elif self.q_extract == 'bm25':
                chunk_idx = self.q_extract_model.batch_rank_chunks(batch_docs=data['contexts_str'],
                                                                  batch_chunks=[data['random_chunks_str'][
                                                                                i * num_chunk: (i + 1) * num_chunk]
                                                                                for i in range(len(data['docs_str']))])
            elif self.q_extract_model:
                chunk_idx = self.rank_chunks_by_ext_model(docs=data['contexts_str'], chunks=data['random_chunks_str'])
            else:
                raise NotImplementedError
            c_tokens = torch.stack(
                [chunks[cidx.item()] for chunks, cidx in zip(chunk_tokens.reshape(bsz, num_chunk, -1), chunk_idx)])
            c_mask = torch.stack(
                [chunks[cidx.item()] for chunks, cidx in zip(chunk_mask.reshape(bsz, num_chunk, -1), chunk_idx)])
            if self.q_extract_ratio and self.q_extract_ratio < 1.0:
                c_tokens, c_mask = mix_two_inputs(c_tokens, c_mask, q_tokens, q_mask, input0_ratio=self.q_extract_ratio)
            # print(c_tokens.shape)
            qemb = self.encoder_q(input_ids=c_tokens, attention_mask=c_mask)  # queries: B,H
        else:
            qemb = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)  # queries: B,H

        if self.norm_query:
            qemb = nn.functional.normalize(qemb, dim=-1)

        # 3. apply projectors
        # 4. apply predictor (q/k interaction)
        # 5. computer loss
        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        all_kemb = gather_kemb
        if self.neg_names is not None and len([data[neg_name]['input_ids'] for neg_name in self.neg_names if neg_name in data]) > 0:
            neg_tokens = torch.cat([data[neg_name]['input_ids'] for neg_name in self.neg_names])
            neg_mask = torch.cat([data[neg_name]['attention_mask'] for neg_name in self.neg_names])
            neg_kemb = self.encoder_k(input_ids=neg_tokens, attention_mask=neg_mask).contiguous()
            gather_neg_kemb = gather_fn(neg_kemb)
            all_kemb = torch.cat([gather_kemb, gather_neg_kemb])

        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)
        labels = labels + dist_utils.get_rank() * len(kemb)
        if self.queue_size == 0:
            scores = torch.einsum("id,jd->ij", qemb / self.moco_config.temperature, all_kemb)
        else:
            scores = self._compute_logits(qemb, all_kemb, self.queue_k)
            scores = scores / self.moco_config.temperature
            self._dequeue_and_enqueue(all_kemb, self.queue_k)
        loss = torch.nn.functional.cross_entropy(scores, labels, label_smoothing=self.label_smoothing)

        # log stats
        iter_stats = {}
        if len(stats_prefix) > 0:
            stats_prefix = stats_prefix + "/"
        iter_stats[f"{stats_prefix}loss"] = loss.item()

        if report_metrics:
            predicted_idx = torch.argmax(scores, dim=-1)
            accuracy = 100 * (predicted_idx == labels).float().mean()
            stdq = torch.std(qemb, dim=0).mean().item()
            stdk = torch.std(kemb, dim=0).mean().item()
            stdqueue_k = torch.std(self.queue_k.T, dim=0).mean() if self.queue_k is not None else 0.0
            iter_stats[f"{stats_prefix}accuracy"] = accuracy
            iter_stats[f"{stats_prefix}stdq"] = stdq
            iter_stats[f"{stats_prefix}stdk"] = stdk
            iter_stats[f'{stats_prefix}stdqueue_k'] = stdqueue_k

            doc_norm = gather_norm(kemb)
            query_norm = gather_norm(qemb)
            iter_stats[f'{stats_prefix}doc_norm'] = doc_norm
            iter_stats[f'{stats_prefix}query_norm'] = query_norm
            iter_stats[f'{stats_prefix}norm_diff'] = torch.abs(doc_norm - query_norm)
            iter_stats[f'{stats_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', qemb, kemb).detach().mean()
            iter_stats[f'{stats_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', qemb, kemb).detach().fill_diagonal_(
                0).sum() / (bsz * bsz - bsz)

            iter_stats[f'{stats_prefix}queue_ptr'] = self.queue_ptr
            queue_k_norm = gather_norm(self.queue_k.T) if self.queue_k is not None else 0.0
            iter_stats[f'{stats_prefix}queue_k_norm'] = queue_k_norm
            if self.neg_names is not None and len([data[neg_name]['input_ids'] for neg_name in self.neg_names if neg_name in data]) > 0:
                iter_stats[f'{stats_prefix}inbatch_hardneg_score'] = torch.einsum('bd,bd->b', qemb, neg_kemb).detach().mean()
                iter_stats[f'{stats_prefix}across_neg_score'] = torch.einsum('id,jd->ij', qemb, gather_neg_kemb).detach().mean()

            # compute on each device, only dot-product
            iter_stats[f'{stats_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', qemb, kemb).detach().mean()
            iter_stats[f'{stats_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', qemb, kemb).detach().fill_diagonal_(0).sum() / (bsz*bsz-bsz)
            if self.queue_k is not None:
                iter_stats[f'{stats_prefix}q@queue_neg_score'] = torch.einsum('id,jd->ij', qemb, self.queue_k.T).detach().mean()

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
        if self.indep_encoder_k and not is_query:
            encoder = self.encoder_k
        pooler_output = encoder(input_ids, attention_mask=attention_mask)
        if is_query and self.norm_query:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)
        elif not is_query and self.norm_doc:
            pooler_output = nn.functional.normalize(pooler_output, dim=-1)

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=pooler_output,
        )
