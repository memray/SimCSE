# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch
import torch.nn as nn
import logging
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

from src import contriever, dist_utils
from src.model_utils import load_retriever, load_hf, gather_norm, ContrastiveLearningOutput

logger = logging.getLogger(__name__)


class InBatch(nn.Module):
    def __init__(self, opt, hf_config=None):
        super(InBatch, self).__init__()

        self.opt = opt
        self.hf_config = hf_config
        self.indep_encoder_k = opt.indep_encoder_k
        self.neg_indices = opt.neg_indices
        self.projection_size = opt.projection_size

        self.sim_metric = getattr(opt, 'sim_metric', 'dot')
        self.norm_doc = opt.norm_doc
        self.norm_query = opt.norm_query
        self.label_smoothing = opt.label_smoothing
        retriever, tokenizer = load_retriever(opt.model_name_or_path, pooling=opt.pooling, hf_config=hf_config)
        self.tokenizer = tokenizer
        self.encoder_q = retriever
        if self.indep_encoder_k:
            retriever, tokenizer = load_retriever(opt.model_name_or_path, pooling=opt.pooling, hf_config=hf_config)
            self.encoder_k = retriever
        else:
            self.encoder_k = self.encoder_q


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
            return self.infer_forward(
                is_query,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
        else:
            return self.train_forward(
                input_ids,
                attention_mask=attention_mask,
                update_kencoder_queue=update_kencoder_queue,
                report_align_unif=report_align_unif,
                report_metrics=report_metrics
            )

    def train_forward(self, input_ids, attention_mask, stats_prefix='', report_metrics=False, **kwargs):
        q_tokens = input_ids[0]
        q_mask = attention_mask[0]
        k_tokens = input_ids[1]
        k_mask = attention_mask[1]
        bsz = len(q_tokens)

        # print('q_tokens.shape=', q_tokens.shape, '\n')
        # print('k_tokens.shape=', k_tokens.shape, '\n')

        qemb = self.encoder_q(input_ids=q_tokens, attention_mask=q_mask)
        kemb = self.encoder_k(input_ids=k_tokens, attention_mask=k_mask).contiguous()

        gather_fn = dist_utils.gather
        gather_kemb = gather_fn(kemb)
        all_kemb = gather_kemb
        if self.neg_indices is not None and max(self.neg_indices) < len(input_ids):
            neg_tokens = torch.cat([input_ids[i] for i in self.neg_indices])
            neg_mask = torch.cat([attention_mask[i] for i in self.neg_indices])
            neg_kemb = self.encoder_k(input_ids=neg_tokens, attention_mask=neg_mask).contiguous()
            gather_neg_kemb = gather_fn(neg_kemb)
            all_kemb = torch.cat([gather_kemb, gather_neg_kemb])
            # all_kemb = torch.cat([gather_kemb, neg_kemb])

        labels = torch.arange(0, bsz, dtype=torch.long, device=q_tokens.device)
        labels = labels + dist_utils.get_rank() * len(kemb)
        scores = torch.einsum("id,jd->ij", qemb / self.opt.temperature, all_kemb)
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
            iter_stats[f"{stats_prefix}accuracy"] = accuracy
            iter_stats[f"{stats_prefix}stdq"] = stdq
            iter_stats[f"{stats_prefix}stdk"] = stdk

            doc_norm = gather_norm(kemb)
            query_norm = gather_norm(qemb)
            iter_stats[f'{stats_prefix}doc_norm'] = doc_norm
            iter_stats[f'{stats_prefix}query_norm'] = query_norm
            iter_stats[f'{stats_prefix}norm_diff'] = torch.abs(doc_norm - query_norm)
            iter_stats[f'{stats_prefix}inbatch_pos_score'] = torch.einsum('bd,bd->b', qemb, kemb).detach().mean()
            iter_stats[f'{stats_prefix}inbatch_neg_score'] = torch.einsum('id,jd->ij', qemb, kemb).detach().fill_diagonal_(
                0).sum() / (bsz * bsz - bsz)
            if self.neg_indices is not None and max(self.neg_indices) < len(input_ids):
                iter_stats[f'{stats_prefix}inbatch_hardneg_score'] = torch.einsum('bd,bd->b', qemb, neg_kemb).detach().mean()
                iter_stats[f'{stats_prefix}across_neg_score'] = torch.einsum('id,jd->ij', qemb, gather_neg_kemb).detach().mean()

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
