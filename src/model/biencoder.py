# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from src.model.bm25 import BM25Okapi
from src.utils import dist_utils
from src.utils.model_utils import gather_norm, load_retriever


class BiEncoder(nn.Module):
    def __init__(self, moco_config, hf_config):
        super(BiEncoder, self).__init__()
        self.q_select = getattr(moco_config, 'q_select', None)
        self.num_random_chunk = getattr(moco_config, 'num_random_chunk', 0)
        if self.q_select is not None: assert self.num_random_chunk > 0, 'num_random_chunk must be >0 if q_select enabled'
        self.q_select_ratio = getattr(moco_config, 'q_select_ratio', None)
        if self.q_select is None:
            self.q_select_model, self.q_select_tokenizer = None, None
        elif self.q_select == 'self-dot':
            pass
        elif self.q_select == 'bm25':
            # make sure `bm25` is a substring of the path :D
            self.q_select_model = BM25Okapi()
            self.q_select_model.load_from_json('/export/home/data/search/wiki/UPR_output/bm25-wikipsg/model.json')
        else:
            self.q_select_tokenizer = AutoTokenizer.from_pretrained(moco_config.q_select)
            self.q_select_model = AutoModelForSeq2SeqLM.from_pretrained(moco_config.q_select, torch_dtype=torch.bfloat16)

    def state_dict(self):
        state_dict = super(BiEncoder, self).state_dict()
        if 'queue_k' in state_dict: del state_dict['queue_k']
        if 'queue_q' in state_dict: del state_dict['queue_q']
        if 'q_select_model' in state_dict: del state_dict['q_select_model']
        if 'q_select_tokenizer' in state_dict: del state_dict['q_select_tokenizer']
        return state_dict



    def get_encoder(self, return_encoder_k=False):
        if return_encoder_k:
            return self.encoder_k
        else:
            return self.encoder_q

    def _momentum_update_key_encoder(self):
        """
        Update of the key encoder
        """
        for param_q, param_k in zip(self.encoder_q.model.parameters(), self.encoder_k.model.parameters()):
            param_k.data = param_k.data * self.momentum + param_q.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys, queue):
        # gather keys before updating queue
        keys = dist_utils.gather_nograd(keys.contiguous())  # [B,H] -> [B*n_gpu,H]
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        assert self.active_queue_size % batch_size == 0, f'batch_size={batch_size}, active_queue_size={self.active_queue_size}'  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        queue[:, ptr:ptr + batch_size] = keys.T

    def _compute_norm_loss(self, q, k):
        q_norm = gather_norm(q)
        k_norm = gather_norm(k)
        norm_diff = (q_norm - k_norm)**2
        return norm_diff

    def rank_chunks_by_ext_model(self, docs, chunks):
        num_chunk_per_doc = len(chunks) // len(docs)
        doc_encoding = self.q_select_tokenizer(docs, padding='longest', max_length=256,
                                      truncation=True, add_special_tokens=True, return_tensors='pt')
        chunk_encoding = self.q_select_tokenizer(chunks, padding='longest', max_length=32,
                                        truncation=True, add_special_tokens=False, return_tensors='pt')
        doc_ids, doc_attention_mask = doc_encoding.input_ids, doc_encoding.attention_mask  # [bs, len]
        chunk_ids, chunk_attention_mask = chunk_encoding.input_ids, chunk_encoding.attention_mask  # [bs*nchunk, len]
        doc_ids = doc_ids.to(self.get_encoder().model.device)
        doc_attention_mask = doc_attention_mask.to(self.get_encoder().model.device)
        chunk_ids = chunk_ids.to(self.get_encoder().model.device)
        chunk_attention_mask = chunk_attention_mask.to(self.get_encoder().model.device)
        doc_ids = torch.repeat_interleave(doc_ids.unsqueeze(1), num_chunk_per_doc, dim=1)  # [bs, len] -> [bs,nchunk,len]
        doc_ids = doc_ids.reshape(-1, doc_ids.shape[-1])  #[bs,nchunk,len] -> [bs*nchunk,len]
        doc_attention_mask = torch.repeat_interleave(doc_attention_mask.unsqueeze(1), num_chunk_per_doc, dim=1)
        doc_attention_mask = doc_attention_mask.reshape(-1, doc_attention_mask.shape[-1])
        with torch.no_grad():
            logits = self.q_select_model(input_ids=doc_ids, attention_mask=doc_attention_mask, labels=chunk_ids).logits  # [bs,chunk_len,vocab_size]
            # logits = self.model(input_ids=chunk_ids, attention_mask=chunk_attention_mask, labels=doc_ids).logits
            log_softmax = torch.nn.functional.log_softmax(logits, dim=-1)  # [bs,chunk_len,vocab_size]
            nll = -log_softmax.gather(2, chunk_ids.unsqueeze(2)).squeeze(2)  # [bs,chunk_len]
            avg_nll = torch.sum(nll, dim=1)  # [bs*nchunk]
        # self.print_doc_chunks(docs, chunks, avg_nll)
        selected_chunk_ids = avg_nll.reshape(-1, num_chunk_per_doc).argmin(dim=1)
        return selected_chunk_ids

    def print_doc_chunks(self, docs, chunks, avg_nll):
        num_chunk_per_doc = len(chunks) // len(docs)
        for docid, doc in enumerate(docs):
            chunk_scores = []
            doc_chunks = chunks[docid * num_chunk_per_doc: (docid + 1) * num_chunk_per_doc]
            doc_chunk_scores = avg_nll.tolist()[docid * num_chunk_per_doc: (docid + 1) * num_chunk_per_doc]
            print(docid, doc)
            for cid, (chunk, chunk_score) in enumerate(zip(doc_chunks, doc_chunk_scores)):
                item = {
                    "id": cid,
                    "chunk": chunk,
                    "score": chunk_score}
                chunk_scores.append(item)
            chunk_scores = sorted(chunk_scores, key=lambda k:k['score'])
            for item in chunk_scores:
                print('\t[%.2f] %d. %s' % (item['score'], item['id'], item['chunk']))