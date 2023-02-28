# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import glob
from collections import defaultdict

import numpy as np

from src.qa.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def validate(data, workers_num, match_type):
    match_stats = calculate_matches(data, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    #logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    #logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits


def read_score_file(path, name2scores, ranks):
    dataset_name = path[path.rfind('/') + 1: path.rfind('recall_at_k.csv')].strip('-')
    if not dataset_name.strip() or dataset_name.startswith('recall_at_k'):
        dataset_name = path.split('/')[-2]
    scores = {}
    with open(path, 'r') as fin:
        for line in fin:
            rank, score = line.split(',')
            scores[int(rank)] = float(score)
    message = f"Evaluate results from {path}:\n"
    for k in ranks:
        recall = 100 * scores[k]
        name2scores[dataset_name].append(recall)
        message += f' R@{k}: {recall:.1f}'
    logger.info(message)


def read_retrieved_result(path, name2scores):
    # For some reason, on curatedtrec the score is always a bit lower than the one computed by spider, so abandoned.
    dataset_name = path[path.rfind('/')+1: path.rfind('-results')]
    with open(path, 'r') as fin:
        if path.endswith('json'):
            data = json.load(fin)
        else:
            data = []
            for line in fin:
                data.append(json.loads(line))
    match_type = "regex" if "curatedtrec" in dataset_name else "string"
    top_k_hits = validate(data, args.validation_workers, match_type)
    message = f"Evaluate results from {path}:\n"
    for k in [1, 5, 10, 20, 100]:
        if k <= len(top_k_hits):
            recall = 100 * top_k_hits[k-1]
            name2scores[dataset_name].append(recall)
            message += f' R@{k}: {recall:.1f}'
    logger.info(message)


def main(args):
    datapaths = sorted(glob.glob(args.data, recursive=True))
    name2scores = defaultdict(list)
    if len(datapaths) == 0:
        print('Found no output for eval!')
    for path in datapaths:
        read_score_file(path, name2scores, ranks=[1, 5, 10, 20, 100])

    for dataset_name, scores in name2scores.items():
        rows = [dataset_name] + [f'{s:.1f}' for s in scores]
        print(','.join(rows))

    eq_score_dict = defaultdict(list)
    score_dict = {}
    for dataset_name, scores in name2scores.items():
        if dataset_name.startswith('P'):
            eq_score_dict["entityqs-acc@1"].append(scores[0])
            eq_score_dict["entityqs-acc@5"].append(scores[1])
            eq_score_dict["entityqs-acc@10"].append(scores[2])
            eq_score_dict["entityqs-acc@20"].append(scores[3])
            eq_score_dict["entityqs-acc@100"].append(scores[4])
        else:
            score_dict[f"{dataset_name}-acc@1"] = scores[0]
            score_dict[f"{dataset_name}-acc@5"] = scores[1]
            score_dict[f"{dataset_name}-acc@10"] = scores[2]
            score_dict[f"{dataset_name}-acc@20"] = scores[3]
            score_dict[f"{dataset_name}-acc@100"] = scores[4]
    assert len(eq_score_dict["entityqs-acc@5"]) == 24
    score_dict["entityqs-macro-acc@1"] = np.mean(eq_score_dict["entityqs-acc@1"])
    score_dict["entityqs-macro-acc@5"] = np.mean(eq_score_dict["entityqs-acc@5"])
    score_dict["entityqs-macro-acc@10"] = np.mean(eq_score_dict["entityqs-acc@10"])
    score_dict["entityqs-macro-acc@20"] = np.mean(eq_score_dict["entityqs-acc@20"])
    score_dict["entityqs-macro-acc@100"] = np.mean(eq_score_dict["entityqs-acc@100"])

    dataset_names = ['nq-test', 'trivia-test', 'webq-test', 'curatedtrec-test', 'squad1-test', 'entityqs-macro']
    header, row = [], []
    for dataset_name in dataset_names:
        for rank in [1, 5, 10, 20, 100]:
            header.append(f"{dataset_name}@{rank}")
            row.append('{:.2f}'.format(score_dict[f"{dataset_name}-acc@{rank}"]))

    print()
    print(args.data)
    print(','.join(header))
    print(','.join(row))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=None)
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")
    args = parser.parse_args()
    # exp_name = 'wiki.ExtQ-bm25-chunk16.seed477.moco-2e14.contriever256-special.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
    # args.data = f'/export/home/exp/search/unsup_dr/wikipsg_v1/{exp_name}/qa_output/**/*.csv'

    # exp_name = 'fb-contriever.msmarco.dot'  # bm25, fb-contriever.dot, fb-contriever.msmarco.dot, spar-wiki-bm25, spar-paq-bm25
    # args.data = f'/export/home/exp/search/unsup_dr/baselines/{exp_name}/qa_output/**/*.csv'

    # exp_name = 'wiki.extphrase3-50.seed477.moco-2e14.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5'
    # args.data = f'/export/home/exp/search/unsup_dr/wikipsg_v1/{exp_name}/qa_output/**/*.csv'

    # exp_name = 'cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'
    # args.data = f'/export/home/exp/search/unsup_dr/cc_v1/{exp_name}/qa_output/**/*.csv'

    # exp_name = 'FT-inbatch-random-neg1023+1024.cc-hybrid.RC20+T0gen80.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step200k.bs2048.lr5e5'
    # args.data = f'/export/home/exp/search/unsup_dr/wikipsg_v1-FT/{exp_name}/qa_output/**/*.csv'

    exp_name = 'wiki+cc.RC20+T0gen80.moco-2e14.contriever256.bert-base-uncased.avg.dot.len256qd128.step200k.bs4096.lr5e5'
    args.data = f'/export/home/exp/search/unsup_dr/cc_v2/{exp_name}/qa_output/**/*.csv'

    main(args)
