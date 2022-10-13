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

from src.qa.evaluation import calculate_matches

logger = logging.getLogger(__name__)

def validate(data, workers_num, match_type):
    match_stats = calculate_matches(data, workers_num, match_type)
    top_k_hits = match_stats.top_k_hits

    #logger.info('Validation results: top k documents hits %s', top_k_hits)
    top_k_hits = [v / len(data) for v in top_k_hits]
    #logger.info('Validation results: top k documents hits accuracy %s', top_k_hits)
    return top_k_hits


def read_score_file(path, name2scores):
    dataset_name = path[path.rfind('/') + 1: path.rfind('-recall_at_k.csv')]
    scores = {}
    with open(path, 'r') as fin:
        for line in fin:
            rank, score = line.split(',')
            scores[int(rank)] = float(score)
    message = f"Evaluate results from {path}:\n"
    for k in [5, 10, 20, 100]:
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
    for k in [5, 10, 20, 100]:
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
        read_score_file(path, name2scores)

    for dataset_name, scores in name2scores.items():
        rows = [dataset_name] + [f'{s:.1f}' for s in scores]
        print(','.join(rows))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, type=str, default=None)
    parser.add_argument('--validation_workers', type=int, default=16,
                        help="Number of parallel processes to validate results")

    args = parser.parse_args()
    main(args)
