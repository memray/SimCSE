# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import os
import inspect

import torch
import torch.distributed as dist
from typing import List, Dict
import numpy as np

import beir_eval.dist_utils as dist_utils

import beir.util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch


class DenseEncoderModel:
    def __init__(
        self, 
        query_encoder, 
        doc_encoder=None, 
        tokenizer=None, 
        maxlength=512, 
        add_special_tokens=True, 
        norm_query=False, 
        norm_doc=False,
        **kwargs
    ):
        self.query_encoder = query_encoder
        self.doc_encoder = doc_encoder
        self.tokenizer = tokenizer
        self.maxlength = maxlength
        self.add_special_tokens = add_special_tokens
        self.norm_query = norm_query
        self.norm_doc = norm_doc
  
    def encode_queries(self, queries: List[str], batch_size: int, **kwargs) -> np.ndarray:

        if dist.is_initialized(): 
            idx = np.array_split(range(len(queries)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(queries))
        queries = [queries[i] for i in idx]

        allemb = []
        nbatch = (len(queries)-1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k+1) * batch_size, len(queries))

                qencode = self.tokenizer.batch_encode_plus(
                    queries[start_idx:end_idx], 
                    max_length=self.maxlength, 
                    padding=True, 
                    truncation=True, 
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt", 
                )
                ids, mask = qencode['input_ids'], qencode['attention_mask']
                ids, mask = ids.cuda(), mask.cuda()

                if 'is_query' in inspect.getfullargspec(self.query_encoder.forward).args:
                    emb = self.query_encoder(ids, mask, sent_emb=True, is_query=True)
                else:
                    emb = self.query_encoder(ids, mask, sent_emb=True)
                # # @memray for ANCE
                # if 'is_query' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, is_query=True)
                # # @memray for SimCSE
                # elif 'sent_emb' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, sent_emb=True)
                # # @memray for some HF models don't have normalize
                # elif 'normalize' in inspect.getfullargspec(self.query_encoder.forward).args:
                #     emb = self.query_encoder(ids, mask, normalize=self.norm_query)
                # else:
                #     emb = self.query_encoder(ids, mask)
                if hasattr(emb, 'pooler_output'):
                    emb = emb['pooler_output']
                allemb.append(emb)

        allemb = torch.cat(allemb, dim=0) 
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb


    def encode_corpus(self, corpus: List[Dict[str, str]], batch_size: int, **kwargs):

        if dist.is_initialized(): 
            idx = np.array_split(range(len(corpus)), dist.get_world_size())[dist.get_rank()]
        else:
            idx = range(len(corpus))
        corpus = [corpus[i] for i in idx]
        corpus = [
            c['title'] + ' ' + c['text'] if len(c['title']) > 0 else c['text'] for c in corpus
        ]
        
        allemb = []
        nbatch = (len(corpus)-1) // batch_size + 1
        with torch.no_grad():
            for k in range(nbatch):
                start_idx = k * batch_size
                end_idx = min((k+1) * batch_size, len(corpus))

                cencode = self.tokenizer.batch_encode_plus(
                    corpus[start_idx:end_idx], 
                    max_length=self.maxlength, 
                    padding=True, 
                    truncation=True, 
                    add_special_tokens=self.add_special_tokens,
                    return_tensors="pt", 
                )
                ids, mask = cencode['input_ids'], cencode['attention_mask']
                ids, mask = ids.cuda(), mask.cuda()

                # @memray for SimCSE
                if 'is_query' in inspect.getfullargspec(self.doc_encoder.forward).args:
                    emb = self.doc_encoder(ids, mask, sent_emb=True, is_query=False)
                else:
                    emb = self.doc_encoder(ids, mask, sent_emb=True)
                if hasattr(emb, 'pooler_output'):
                    emb = emb['pooler_output']
                allemb.append(emb)

        allemb = torch.cat(allemb, dim=0)
        if dist.is_initialized():
            allemb = dist_utils.varsize_gather_nograd(allemb)
        allemb = allemb.cpu().numpy()
        return allemb

def evaluate_model(
        query_encoder, 
        doc_encoder, 
        tokenizer, 
        dataset, 
        batch_size=128, 
        max_length=512,
        add_special_tokens=True,
        norm_query=False, 
        norm_doc=False, 
        is_main=True, 
        split='test', 
        metric='dot',
        beir_data_path="BEIR/datasets",
        add_qd_prompt=False,
    ):
    if metric == 'cosine':
        metric = 'cos_sim'
    if hasattr(query_encoder, "module"):
        query_encoder = query_encoder.module
    query_encoder.eval()

    if doc_encoder is not None:
        if hasattr(doc_encoder, "module"):
            doc_encoder = doc_encoder.module
        doc_encoder.eval()
    else:
        doc_encoder = query_encoder
    
    dmodel = DenseRetrievalExactSearch(
        DenseEncoderModel(
            query_encoder=query_encoder, 
            doc_encoder=doc_encoder, 
            tokenizer=tokenizer,
            maxlength=max_length,
            add_special_tokens=add_special_tokens, 
            norm_query=norm_query, 
            norm_doc=norm_doc,
        ),
        batch_size=batch_size,
        add_qd_prompt=add_qd_prompt
    )
    retriever = EvaluateRetrieval(dmodel, score_function=metric)
    url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
    data_path = beir.util.download_and_unzip(url, beir_data_path)
    if dataset == 'cqadupstack':
        ndcgs, _maps, recalls, precisions, mrrs, recall_caps, holes = [], [], [], [], [], [], []
        cqasubsets = [
            'android', 
            'english', 
            'gaming', 
            'gis', 
            'mathematica', 
            'physics', 
            'programmers', 
            'stats', 
            'tex', 
            'unix', 
            'webmasters', 
            'wordpress'
        ]
        for sub in cqasubsets:
            data_folder = os.path.join(data_path, sub)
            corpus, queries, qrels = GenericDataLoader(data_folder=data_folder).load(split=split)
            # if is_main: print(f'Start retrieving, #(corpus)={len(corpus)}, #(queries)={len(queries)}, '
            #                   f'batch_size={retriever.retriever.batch_size}, chunk_size={retriever.retriever.corpus_chunk_size}')
            results = retriever.retrieve(corpus, queries)
            if is_main:
                # print(f'Start evaluating, #(qrels)={len(qrels)}, #(results)={len(results)}')
                ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
                mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
                recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
                hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
                ndcgs.append(ndcg)
                _maps.append(_map)
                recalls.append(recall)
                precisions.append(precision)
                mrrs.append(mrr)
                recall_caps.append(recall_cap)
                holes.append(hole)
        if is_main:
            print('Dataset: ', dataset)
            ndcg = {key: sum(item.get(key) for item in ndcgs) / 12 for key in ndcgs[0]}
            _map = {key: sum(item.get(key) for item in _maps) / 12 for key in _maps[0]}
            recall = {key: sum(item.get(key) for item in recalls) / 12 for key in recalls[0]}
            precision = {key: sum(item.get(key) for item in precisions) / 12 for key in precisions[0]}
            mrr = {key: sum(item.get(key) for item in mrrs) / 12 for key in mrrs[0]}
            recall_cap = {key: sum(item.get(key) for item in recall_caps) / 12 for key in recall_caps[0]}
            hole = {key: sum(item.get(key) for item in holes) / 12 for key in holes[0]}
        else:
            ndcg, _map, recall, precision = None, None, None, None
            mrr, recall_cap, hole = None, None, None
    else:
        corpus, queries, qrels = GenericDataLoader(data_folder=data_path).load(split=split)
        # if is_main: print(f'Start retrieving, #(corpus)={len(corpus)}, #(queries)={len(queries)},'
        #                   f'batch_size={retriever.retriever.batch_size}, chunk_size={retriever.retriever.corpus_chunk_size}')
        results = retriever.retrieve(corpus, queries)
        if is_main:
            print('Dataset: ', dataset)
            # print(f'Start evaluating, #(qrels)={len(qrels)}, #(results)={len(results)}')
            ndcg, _map, recall, precision = retriever.evaluate(qrels, results, retriever.k_values)
            mrr = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="mrr")
            recall_cap = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="recall_cap")
            hole = retriever.evaluate_custom(qrels, results, retriever.k_values, metric="hole")
        else:
            ndcg, _map, recall, precision = None, None, None, None
            mrr, recall_cap, hole = None, None, None
    return ndcg, _map, recall, precision, mrr, recall_cap, hole
