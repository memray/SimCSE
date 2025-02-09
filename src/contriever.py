# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import os
import torch
import transformers
from transformers import BertModel

from simcse.model_utils import GaussianDropout, VariationalDropout
from src import utils


def dropout(p=None, dim=None, method='standard'):
    if method == 'standard':
        return torch.nn.Dropout(p)
    elif method == 'gaussian':
        return GaussianDropout(p / (1 - p))
    elif method == 'variational':
        return VariationalDropout(p / (1 - p), dim)


class Contriever(BertModel):
    def __init__(self, config, moco_config, **kwargs):
        super().__init__(config, add_pooling_layer=False)
        # will be set outside
        self.pooling = moco_config.pooling
        self.pooling_dropout = moco_config.pooling_dropout
        self.pooling_dropout_prob = moco_config.pooling_dropout_prob
        if self.pooling_dropout in ['standard', 'gaussian', 'variational'] and self.pooling_dropout_prob > 0:
            self.dropout = dropout(p=self.pooling_dropout, method=self.pooling_dropout_prob, dim=1)
        else:
            self.dropout = None
        self.num_view = 1  # set outside
        self.cls_token_id = None  # set outside

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        output_attentions=None,
        output_hidden_states=None,
        normalize=False,
    ):
        # append CLS special tokens to input
        # if self.num_view == 1, just use the default CLS
        if self.num_view > 1:
            bs, seqlen = input_ids.shape
            extended_len = seqlen + self.num_view - 1
            if extended_len > self.encoder.config.max_position_embeddings:
                diff_len = extended_len - self.encoder.config.max_position_embeddings
                extended_len = self.encoder.config.max_position_embeddings
                input_ids = input_ids[:,:-diff_len]
                attention_mask = attention_mask[:,:-diff_len]
            extra_cls_tokens = torch.zeros((bs, self.num_view - 1), device=input_ids.device, dtype=input_ids.dtype) + self.cls_token_id
            extra_mask_tokens = torch.ones((bs, self.num_view - 1), device=input_ids.device, dtype=input_ids.dtype)
            input_ids = torch.cat([extra_cls_tokens, input_ids], dim=1)
            attention_mask = torch.cat([extra_mask_tokens, attention_mask], dim=1)
            position_ids = torch.arange(extended_len, device=input_ids.device).expand(1, -1)

        model_output = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        last_hidden = model_output['last_hidden_state']
        last_hidden = last_hidden.masked_fill(~attention_mask[..., None].bool(), 0.)

        if self.pooling == "average":
            emb = last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]  # shape=[B,H]
        elif self.pooling == "cls":
            emb = last_hidden[:, 0]  # shape=[B,H]
        elif self.pooling == "multiview":
            emb = last_hidden[:, :self.num_view]  # shape=[B,V,H]
        # if normalize: emb = torch.nn.functional.normalize(emb, dim=-1)  # normalized outside
        if self.dropout:
            emb = self.dropout(emb)

        return emb


def load_retriever(model_path):
    #try: #check if model is in a moco wrapper
    path = os.path.join(model_path, "checkpoint.pth")
    if os.path.exists(path):
        pretrained_dict = torch.load(path, map_location="cpu")
        opt = pretrained_dict['opt']
        #retriever_model_id = opt.retriever_model_id
        retriever_model_id = 'bert-base-multilingual-cased'
        tokenizer = utils.load_hf(transformers.AutoTokenizer, retriever_model_id)
        cfg = utils.load_hf(transformers.AutoConfig, retriever_model_id)
        retriever = Contriever(cfg)
        pretrained_dict = pretrained_dict["model"]
        if any("encoder_q." in key for key in pretrained_dict.keys()):
            pretrained_dict = {k.replace("encoder_q.", ""): v for k, v in pretrained_dict.items() if "encoder_q" in k}
        retriever.load_state_dict(pretrained_dict)
    else:
        cfg = utils.load_hf(transformers.AutoConfig, model_path)
        tokenizer = utils.load_hf(transformers.AutoTokenizer, model_path)
        retriever = utils.load_hf(Contriever, model_path)

    return retriever, tokenizer