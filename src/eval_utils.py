import csv
import glob
import json
import os
import pickle

import numpy as np
from torch.nn.parallel import DistributedDataParallel
from transformers.utils import logging
from typing import List, Tuple, Dict

from src.qa.index import Indexer
from src.qa.qa_validation import calculate_matches
from src.beireval import beir_utils, dist_utils
from src.qa.data import load_passages, get_qa_datasets
from src.qa.normalize_text import normalize
from src.senteval import engine
import torch

import torch.distributed as dist
import time
from collections import defaultdict

logger = logging.get_logger(__name__)

# Set path to SentEval
PATH_TO_DATA = '/export/share/ruimeng/project/search/simcse/SentEval/data/'
RECALL_FILE_NAME = "recall_at_k.csv"
RESULTS_FILE_NAME = "results.jsonl"


def embed_passages(passages, model, tokenizer,
                   lowercase=True, normalize_text=True, passage_maxlength=512,
                   per_gpu_batch_size=128):
    total = 0
    allids, allembeddings = [], []
    batch_ids, batch_text = [], []
    with torch.no_grad():
        for k, p in enumerate(passages):
            batch_ids.append(p["id"])
            text = p["title"] + " " + p["text"]
            if lowercase:
                text = text.lower()
            if normalize_text:
                text = normalize(text)
            batch_text.append(text)

            if len(batch_text) == per_gpu_batch_size or k == len(passages) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_text,
                    return_tensors="pt",
                    max_length=passage_maxlength,
                    padding=True,
                    truncation=True,
                )

                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                embeddings = model(**encoded_batch, sent_emb=True, is_query=False).pooler_output

                embeddings = embeddings.cpu()
                total += len(batch_ids)
                allids.extend(batch_ids)
                allembeddings.append(embeddings)

                batch_text = []
                batch_ids = []
                if k % 1000 == 0 and k > 0:
                    logger.info(f"Encoded passages {total}")

    allembeddings = torch.cat(allembeddings, dim=0).numpy()
    return allids, allembeddings


def embed_queries(queries, model, tokenizer,
                  lowercase=True, normalize_text=True,
                  question_maxlength=512, per_gpu_batch_size=128
                  ):
    model.eval()
    query_vectors, batch_question = [], []
    with torch.no_grad():

        for k, q in enumerate(queries):
            if lowercase:
                q = q.lower()
            if normalize_text:
                q = normalize(q)
            batch_question.append(q)

            if len(batch_question) == per_gpu_batch_size or k == len(queries) - 1:

                encoded_batch = tokenizer.batch_encode_plus(
                    batch_question,
                    return_tensors="pt",
                    max_length=question_maxlength,
                    padding=True,
                    truncation=True,
                )
                encoded_batch = {k: v.cuda() for k, v in encoded_batch.items()}
                output = model(**encoded_batch, sent_emb=True, is_query=True)
                output = output.pooler_output
                if output.is_cuda:
                    output = output.cpu()
                query_vectors.append(output)
                batch_question = []

    query_tensor = torch.cat(query_vectors, dim=0)
    logger.info(f"device={dist.get_rank()}, questions embeddings shape: {query_tensor.size()}")
    assert query_tensor.size(0) == len(queries)
    return query_tensor.numpy()


def generate_passage_embeddings(model, tokenizer, passages, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    model = model.cuda()
    shard_id = dist.get_rank()
    num_shards = int(os.environ['WORLD_SIZE'])
    shard_size = len(passages) // num_shards
    start_idx = shard_id * shard_size
    end_idx = start_idx + shard_size
    if shard_id == num_shards - 1:
        end_idx = len(passages)

    logger.info(f"device={shard_id}, embedding generation for {len(passages)} passages, shard-{shard_id} from idx {start_idx} to {end_idx}.")
    passages = passages[start_idx:end_idx]
    allids, allembeddings = embed_passages(passages, model, tokenizer)
    save_path = save_dir + f"/emb_{shard_id:02d}"
    logger.info(f"device={shard_id}, saving {len(allids)} passage embeddings to {save_path}.")
    with open(save_path, mode="wb") as f:
        pickle.dump((allids, allembeddings), f)

    logger.info(f"Total passages processed {len(allids)}. Written to {save_path}.")


def index_encoded_data(index, embedding_files, indexing_batch_size):
    allids = []
    allembeddings = np.array([])
    for i, file_path in enumerate(embedding_files):
        if not os.path.isfile(file_path): continue
        logger.info(f"Loading file {file_path}")
        with open(file_path, "rb") as fin:
            ids, embeddings = pickle.load(fin)

        allembeddings = np.vstack((allembeddings, embeddings)) if allembeddings.size else embeddings
        allids.extend(ids)
        while allembeddings.shape[0] > indexing_batch_size:
            allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    while allembeddings.shape[0] > 0:
        allembeddings, allids = add_embeddings(index, allembeddings, allids, indexing_batch_size)

    logger.info(f"Device {dist.get_rank()}, data indexing completed, allembeddings.shape={allembeddings.shape}.")


def add_embeddings(index, embeddings, ids, indexing_batch_size):
    end_idx = min(indexing_batch_size, embeddings.shape[0])
    ids_toadd = ids[:end_idx]
    embeddings_toadd = embeddings[:end_idx]
    ids = ids[end_idx:]
    embeddings = embeddings[end_idx:]
    index.index_data(ids_toadd, embeddings_toadd)
    return embeddings, ids


def validate(data, workers_num):
    match_stats = calculate_matches(data, workers_num)
    top_k_hits = match_stats.top_k_hits

    logger.info("Validation results: top k documents hits %s", top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    message = ""
    for k in [5, 10, 20, 100]:
        if k <= len(top_k_hits):
            message += f"R@{k}: {top_k_hits[k-1]} "
    logger.info(message)
    return match_stats.questions_doc_hits


def add_passages(data, passages, top_passages_and_scores):
    # add passages to original data
    merged_data = []
    assert len(data) == len(top_passages_and_scores)
    for i, d in enumerate(data):
        results_and_scores = top_passages_and_scores[i]
        docs = [passages[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        ctxs_num = len(docs)
        d["ctxs"] = [
            {
                "id": results_and_scores[0][c],
                "title": docs[c]["title"],
                "text": docs[c]["text"],
                "score": scores[c],
            }
            for c in range(ctxs_num)
        ]


def add_hasanswer(data, hasanswer):
    # add hasanswer to data
    for i, ex in enumerate(data):
        for k, d in enumerate(ex["ctxs"]):
            d["hasanswer"] = hasanswer[i][k]


def load_data(data_path):
    if data_path.endswith(".json"):
        with open(data_path, "r") as fin:
            data = json.load(fin)
    elif data_path.endswith(".jsonl"):
        data = []
        with open(data_path, "r") as fin:
            for k, example in enumerate(fin):
                example = json.loads(example)
                data.append(example)
    elif data_path.endswith(".csv"):
        with open(data_path, "r") as fin:
            reader = csv.reader(fin, delimiter="\t")
            data = []
            for k, row in enumerate(reader):
                ex = {"question": row[0], "answers": row[1]}
                data.append(ex)
    return data


def save_results(
    id2doc: Dict[object, Tuple[str, str]],
    questions: List[str],
    answers: List[List[str]],
    top_passages_and_scores: List[Tuple[List[object], List[float]]],
    per_question_hits: List[List[bool]],
    out_file: str,
    output_no_text: bool = False,
):
    # join passages text with the result ids, their questions and assigning has|no answer labels
    merged_data = []
    assert len(per_question_hits) == len(questions) == len(answers)
    for i, q in enumerate(questions):
        q_answers = answers[i]
        results_and_scores = top_passages_and_scores[i]
        hits = per_question_hits[i]
        docs = [id2doc[doc_id] for doc_id in results_and_scores[0]]
        scores = [str(score) for score in results_and_scores[1]]
        hit_indices = [j+1 for j, is_hit in enumerate(hits) if is_hit]
        hit_min_rank = hit_indices[0] if len(hit_indices) > 0 else None
        ctxs_num = len(hits)

        d = {
                "question": q,
                "answers": q_answers,
                "hit_min_rank": hit_min_rank,
                "all_hits": hit_indices,
                "ctxs": [
                    {
                        "id": results_and_scores[0][c],
                        "rank": (c + 1),
                        "title": docs[c]['title'],
                        "text": docs[c]['text'] if not output_no_text else "",
                        "score": scores[c],
                        "has_answer": hits[c],
                    }
                    for c in range(ctxs_num)
                ],
            }
        merged_data.append(d)

    if out_file.endswith('json'):
        with open(out_file, "w") as writer:
            writer.write(json.dumps(merged_data, indent=4) + "\n")
    else:
        with open(out_file, "w") as writer:
            for d in merged_data:
                writer.write(json.dumps(d) + "\n")
    logger.info("Saved results * scores  to %s", out_file)


def evaluate_qa(model, tokenizer,
                passages, qa_datasets_path,
                passages_embeddings_path, output_dir,
                encode_batch_size=128, search_batch_size=8,
                num_workers=16, n_docs=100,
                ):
    os.makedirs(output_dir, exist_ok=True)
    if isinstance(model, DistributedDataParallel):
        model = model.module
    index = Indexer(model.projection_size, num_threads=num_workers)

    input_paths = glob.glob(passages_embeddings_path)
    input_paths = sorted(input_paths)
    embeddings_dir = os.path.dirname(input_paths[0]) if len(input_paths) > 0 else os.path.dirname(passages_embeddings_path)
    index_path = os.path.join(embeddings_dir, "index.faiss")
    if os.path.exists(index_path):
        index.deserialize_from(embeddings_dir)
    else:
        # index all passages
        os.makedirs(embeddings_dir, exist_ok=True)
        logger.info(f"Indexing passages from files {input_paths}")
        start_time_indexing = time.time()
        index_encoded_data(index, input_paths, encode_batch_size)
        logger.info(f"Indexing time: {time.time()-start_time_indexing:.1f}s")
        index.serialize(embeddings_dir)
    logger.info("Moving index to GPUs")
    start_time_retrieval = time.time()
    index.to_gpu()
    logger.info(f"Moving index to GPUs time: {time.time()-start_time_retrieval:.1f}s")

    eq_score_dict = defaultdict(list)
    score_dict = {}
    # load passages
    id2doc = {d['id']: d for d in passages}
    # get questions & answers
    qa_file_dict = get_qa_datasets(qa_datasets_path)
    for dataset_name, (questions, question_answers) in qa_file_dict.items():
        questions = questions[:32]
        question_answers = question_answers[:32]
        # init
        logger.info("*" * 40)
        logger.info(f"Working on dataset {dataset_name}")
        dataset_output_dir = os.path.join(output_dir, dataset_name)
        os.makedirs(dataset_output_dir, exist_ok=True)

        # encode questions
        questions_embedding = embed_queries(questions, model, tokenizer,
                                            lowercase=True, normalize_text=True,
                                            question_maxlength=512,
                                            per_gpu_batch_size=encode_batch_size)
        # get top k results
        start_time_retrieval = time.time()
        top_ids_and_scores = index.search_knn(questions_embedding, n_docs, index_batch_size=search_batch_size)
        logger.info(f"Search time: {time.time()-start_time_retrieval:.1f} s.")

        # compute scores
        match_type = "regex" if "curated" in dataset_name else "string"
        match_stats = calculate_matches(id2doc, question_answers, top_ids_and_scores, num_workers, match_type)
        top_k_hits = match_stats.top_k_hits
        logger.info("Validation results: top k documents hits %s", top_k_hits)
        top_k_hits = [v / len(top_ids_and_scores) for v in top_k_hits]
        logger.info("Validation results: top k documents hits accuracy %s", top_k_hits)
        out_file = os.path.join(dataset_output_dir, RECALL_FILE_NAME)
        logger.info(f"Saved recall@k info to {out_file}")
        with open(out_file, "w") as f:
            for k, recall in enumerate(top_k_hits):
                f.write(f"{k + 1},{recall}\n")
        out_file = os.path.join(dataset_output_dir, RESULTS_FILE_NAME)
        save_results(
            id2doc,
            questions,
            question_answers,
            top_ids_and_scores,
            match_stats.questions_doc_hits,
            out_file
        )
        if dataset_name.startswith('P'):
            eq_score_dict["entityqs-Acc@5"].append(top_k_hits[4])
            eq_score_dict["entityqs-Acc@20"].append(top_k_hits[19])
            eq_score_dict["entityqs-Acc@100"].append(top_k_hits[-1])
        else:
            score_dict[f"{dataset_name}-Acc@5"] = top_k_hits[4]
            score_dict[f"{dataset_name}-Acc@20"] = top_k_hits[19]
            score_dict[f"{dataset_name}-Acc@100"] = top_k_hits[-1]

    if len(eq_score_dict) > 0:
        assert len(score_dict["entityqs-Acc@5"]) == 24
        score_dict["entityqs-Acc@5"] = np.mean(eq_score_dict["entityqs-Acc@5"])
        score_dict["entityqs-Acc@20"] = np.mean(eq_score_dict["entityqs-Acc@20"])
        score_dict["entityqs-Acc@100"] = np.mean(eq_score_dict["entityqs-Acc@100"])

    return score_dict


def evaluate_beir(model, tokenizer, beir_path, output_dir, sim_function, add_qd_prompt=False, batch_size=32, beir_datasets=None) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    if not beir_datasets:
        # fever will cause gpu error when `Encoding Batch 88/109`
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact'] # quick test
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'quora', 'dbpedia-entity', 'nq'] # mostly reported in Contriever
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid', 'nq', 'dbpedia-entity', 'quora'] # small testsets+NQ+FEVER+Quora
        beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack',
                         'trec-covid', 'quora', 'nq']  # smallest 8 datasets+quora,nq
        # beir_datasets = ['nfcorpus', 'fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack', 'trec-covid']  # smallest 8 datasets
        # beir_datasets = ['fiqa']  # test
    if isinstance(model, DistributedDataParallel):
        model = model.module
    norm_query = model.norm_query
    norm_doc = model.norm_doc
    beir_data_path = beir_path

    metrics = {}
    avg_ndcg_10 = []
    avg_recall_10 = []
    avg_recall_20 = []
    avg_recall_100 = []

    for dataset in beir_datasets:
        torch.cuda.empty_cache()
        logger.info(f"Start evaluating with dataset={dataset}")
        split = 'dev' if dataset == 'msmarco' else 'test'
        ndcg, _map, recall, precision, mrr, recall_cap, hole = beir_utils.evaluate_model(
            query_encoder=model,
            doc_encoder=model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=batch_size,
            norm_query=norm_query,
            norm_doc=norm_doc,
            is_main=dist_utils.is_main(),
            split=split,
            metric=sim_function,
            beir_data_path=beir_data_path,
            add_qd_prompt=add_qd_prompt,
            corpus_chunk_size=20480
        )

        if dist_utils.is_main():
            # logger.info(dataset + ' ' + str(ndcg))
            # logger.info(dataset + ' ' + str(_map))
            # logger.info(dataset + ' ' + str(recall))
            # logger.info(dataset + ' ' + str(precision))
            # logger.info(dataset + ' ' + str(mrr))
            # logger.info(dataset + ' ' + str(recall_cap))
            # logger.info(dataset + ' ' + str(hole))
            ndcg10 = ndcg['NDCG@10']
            recall10 = recall['Recall@10'] if dataset != 'trec-covid' else recall_cap['R_cap@10']
            recall20 = recall['Recall@20'] if dataset != 'trec-covid' else recall_cap['R_cap@20']
            recall100 = recall['Recall@100'] if dataset != 'trec-covid' else recall_cap['R_cap@100']
            metrics[f'eval_{dataset}_ndcg@10'] = ndcg10
            metrics[f'eval_{dataset}_recall@10'] = recall10
            metrics[f'eval_{dataset}_recall@20'] = recall20
            metrics[f'eval_{dataset}_recall@100'] = recall100
            avg_ndcg_10.append(ndcg10)
            avg_recall_10.append(recall10)
            avg_recall_20.append(recall20)
            avg_recall_100.append(recall100)

            result_dict = {
                'dataset': dataset,
                'split': split,
                'metric': sim_function,
                'norm_query': norm_query,
                'norm_doc': norm_doc,
                'scores': {
                    'ndcg': ndcg,
                    'map': _map,
                    'precision': precision,
                    'recall': recall,
                    'mrr': mrr,
                    'recall_cap': recall_cap,
                    'hole': hole,
                }
            }
            logger.info(f"Dump results of {dataset} to {output_dir}/{dataset}.json")
            with open(f"{output_dir}/{dataset}.json", 'w') as writer:
                writer.write(json.dumps(result_dict, indent=4) + "\n")
            rows = ['metric,@1,@3,@5,@10,@20,@50,@100,@200,@1000']
            for metric_name, scores in result_dict['scores'].items():
                row = ','.join([str(s) for s in ([metric_name] + list(scores.values()))])
                rows.append(row)
            with open(f"{output_dir}/{dataset}.csv", 'w') as writer:
                for row in rows:
                    writer.write(row + "\n")

    metrics['eval_avg_ndcg@10'] = np.mean(avg_ndcg_10)
    metrics['eval_avg_recall@10'] = np.mean(avg_recall_10)
    metrics['eval_avg_recall@20'] = np.mean(avg_recall_20)
    metrics['eval_avg_recall@100'] = np.mean(avg_recall_100)

    return metrics


def evaluate_senteval(model, tokenizer, output_dir, eval_senteval_transfer: bool = False) -> Dict[str, float]:
    os.makedirs(output_dir, exist_ok=True)
    # SentEval prepare and batcher
    def prepare(params, samples):
        return

    def batcher(params, batch):
        sentences = [' '.join(s) for s in batch]
        batch = tokenizer.batch_encode_plus(
            sentences,
            return_tensors='pt',
            padding=True,
        )
        for k in batch:
            batch[k] = batch[k].to(model.device)
        with torch.no_grad():
            outputs = model(**batch, output_hidden_states=True, return_dict=True, sent_emb=True)
            pooler_output = outputs.pooler_output
        return pooler_output.cpu()

    # Set params for SentEval (fastmode)
    params = {'task_path': PATH_TO_DATA, 'usepytorch': True, 'kfold': 5}
    params['classifier'] = {'nhid': 0, 'optim': 'rmsprop', 'batch_size': 128,
                            'tenacity': 3, 'epoch_size': 2}

    se = engine.SE(params, batcher, prepare)
    tasks = ['STSBenchmark', 'SICKRelatedness']
    if eval_senteval_transfer:
        tasks = ['STSBenchmark', 'SICKRelatedness', 'MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']
    model.eval()
    results = se.eval(tasks)

    stsb_spearman = results['STSBenchmark']['dev']['spearman'][0]
    sickr_spearman = results['SICKRelatedness']['dev']['spearman'][0]

    metrics = {"eval_stsb_spearman": stsb_spearman, "eval_sickr_spearman": sickr_spearman,
               "eval_avg_sts": (stsb_spearman + sickr_spearman) / 2}
    if eval_senteval_transfer:
        avg_transfer = 0
        for task in ['MR', 'CR', 'SUBJ', 'MPQA', 'SST2', 'TREC', 'MRPC']:
            avg_transfer += results[task]['devacc']
            metrics['eval_{}'.format(task)] = results[task]['devacc']
        avg_transfer /= 7
        metrics['eval_avg_transfer'] = avg_transfer

    results.update(metrics)
    with open(f"{output_dir}/senteval.json", 'w') as writer:
        writer.write(json.dumps(results, indent=4) + "\n")

    return metrics
