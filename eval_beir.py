# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import logging
import json
import os

import numpy as np

import src.beireval.slurm as slurm
import src.beireval.beir_utils as beir_utils
import src.utils.training_utils as utils
import src.utils.dist_utils as dist_utils
from utils import training_utils

logger = logging.getLogger(__name__)

BEIR_datasets = [
        'msmarco',
        'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04',
        'cqadupstack', 'quora'
        ]
BEIR_public_datasets = [
        'msmarco', 'trec-covid', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact',
        'cqadupstack', 'quora'
        ]
small_datasets = ['fiqa', 'arguana', 'scidocs', 'scifact', 'webis-touche2020', 'cqadupstack']
def main(args):
    slurm.init_distributed_mode(args)
    slurm.init_signal_handler()

    os.makedirs(args.output_dir, exist_ok=True)

    logger = utils.init_logger(args, stdout_only=True)
    logger.setLevel(logging.DEBUG)
    logger.info(f"Loading model from [{args.model_name_or_path}]")

    q_model, tokenizer = training_utils.load_model(args.model_name_or_path)
    # tokenizer = transformers.AutoTokenizer.from_pretrained(args.model_name_or_path)
    # q_model = transformers.AutoModel.from_pretrained(args.model_name_or_path)

    if args.doc_model_name_or_path is not None:
        # d_model = transformers.AutoModel.from_pretrained(args.doc_model_name_or_path)
        d_model, _ = training_utils.load_model(args.doc_model_name_or_path)
    else:
        d_model = q_model
    q_model = q_model.cuda()
    d_model = d_model.cuda()

    logger.info(f"Start indexing with dataset=[{args.dataset}]")

    if args.dataset == 'all':
        datasets = BEIR_public_datasets
    elif args.dataset == 'small':
        datasets = small_datasets
    else:
        assert args.dataset in BEIR_datasets, f'Unknown dataset [{args.dataset}], supported datasets: \n {str(BEIR_datasets)}'
        datasets = [args.dataset]

    metrics = {}
    avg_ndcg_10, avg_recall_10, avg_recall_20, avg_recall_100 = [], [], [], []
    for dataset in datasets:
        split = 'dev' if dataset == 'msmarco' else 'test'
        logger.info(f"Start evaluating with dataset=[{dataset}], split=[{split}]")
        if os.path.exists(f"{args.output_dir}/{dataset}.json"):
            logger.info(f"Found previous results, skip evaluating [{dataset}]")
            continue
        ndcg, _map, recall, precision, mrr, recall_cap, hole = beir_utils.evaluate_model(
            query_encoder=q_model,
            doc_encoder=d_model,
            tokenizer=tokenizer,
            dataset=dataset,
            batch_size=args.per_gpu_batch_size,
            norm_query=args.norm_query,
            norm_doc=args.norm_doc,
            is_main=dist_utils.is_main(),
            split=split,
            metric=args.metric,
            beir_data_path=args.beir_data_path,
            add_qd_prompt=False,
            corpus_chunk_size=20480
        )

        if dist_utils.is_main():
            ndcg10 = ndcg['NDCG@10']
            recall10 = recall['Recall@10'] if dataset != 'trec-covid' else recall_cap['R_cap@10']
            recall20 = recall['Recall@20'] if dataset != 'trec-covid' else recall_cap['R_cap@20']
            recall100 = recall['Recall@100'] if dataset != 'trec-covid' else recall_cap['R_cap@100']
            metrics[f'eval_beir-{dataset}_ndcg@10'] = ndcg10
            metrics[f'eval_beir-{dataset}_recall@10'] = recall10
            metrics[f'eval_beir-{dataset}_recall@20'] = recall20
            metrics[f'eval_beir-{dataset}_recall@100'] = recall100
            avg_ndcg_10.append(ndcg10)
            avg_recall_10.append(recall10)
            avg_recall_20.append(recall20)
            avg_recall_100.append(recall100)

            result_dict = {
                'dataset': dataset,
                'split': split,
                'metric': args.metric,
                'norm_query': args.norm_query,
                'norm_doc': args.norm_doc,
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
            logger.info(f"Dump results of {dataset} to {args.output_dir}/{dataset}.json")
            print(result_dict)
            with open(f"{args.output_dir}/{dataset}.json", 'w') as writer:
                writer.write(json.dumps(result_dict, indent=4) + "\n")
            rows = ['metric,@1,@3,@5,@10,@20,@50,@100,@200,@1000']
            for metric_name, scores in result_dict['scores'].items():
                row = ','.join([str(s) for s in ([metric_name] + list(scores.values()))])
                rows.append(row)
            with open(f"{args.output_dir}/{dataset}.csv", 'w') as writer:
                for row in rows:
                    writer.write(row + "\n")

    metrics['eval_beir-avg_ndcg@10'] = np.mean(avg_ndcg_10)
    metrics['eval_beir-avg_recall@10'] = np.mean(avg_recall_10)
    metrics['eval_beir-avg_recall@20'] = np.mean(avg_recall_20)
    metrics['eval_beir-avg_recall@100'] = np.mean(avg_recall_100)
    print(metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--dataset", type=str, help="Evaluation dataset from the BEIR benchmark")
    parser.add_argument("--beir_data_path", type=str, default="BEIR/datasets", help="Directory to save and load beir datasets")
    parser.add_argument("--text_maxlength", type=int, default=512, help="Maximum text length")

    parser.add_argument("--per_gpu_batch_size", default=128, type=int, help="Batch size per GPU/CPU for indexing.")
    parser.add_argument("--output_dir", type=str, default="./my_experiment", help="Output directory")
    parser.add_argument("--model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--doc_model_name_or_path", type=str, default=None, help="Model name or path")
    parser.add_argument("--metric", type=str, default="dot", help="Metric used to compute similarity between two embeddings")
    parser.add_argument("--norm_query", action="store_true", help="Normalize query representation")
    parser.add_argument("--norm_doc", action="store_true", help="Normalize document representation")

    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument("--main_addr", type=str, default=-1, help="Main IP address.")
    parser.add_argument("--main_port", type=int, default=-1, help="Main port (for multi-node SLURM jobs)")

    args, _ = parser.parse_known_args()
    main(args)
    
