import json
import os.path
import pandas as pd

'''
4 datasets are not directly available by BEIR: 'bioasq', 'signal1m', 'robust04', 'trec-news'
2 datasets are not in leaderboards by Mar 19, 2022: 'cqadupstack', 'quora'
'''

def main():
    # exp_base_dir = '/export/home/exp/search/unsup_dr/exp_v1/'
    # exp_names = [
    #     'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr3e5',
    #     ]
    # exp_base_dir = '/export/home/exp/search/unsup_dr/exp_v2/'
    # exp_names = [
    #     'pile.1stlineastitle.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5/'
    #     # 'pile+wiki.moco-2e14.contriever-256-prompt.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr3e5'
    #     # 'pile+wiki.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
    #     # 'pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5'
    #
    #     # Ablation of queue size
    #     # 'wikipedia.contriever-256.moco-2e10.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e13.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e16.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr5e5',
    #
    #     # Ablation of lr schedule
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr5e6',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-decayed_cosine-cycle2',
    #
    #     # Ablation of similarity metric
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.cosine.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.normQD.maxlen256.step200k.bs256.lr1e5',
    #
    #     # Ablation of momentum
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum0',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum025',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum099',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum1',
    #
    #     # Ablation of datasets
    #     # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
    #     # 'pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
    #     # 'pile.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2',
    #     # 'c4.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'c4.contriever-256.moco-2e13.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'c4.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'cc100.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr1e5',
    #     # 'cc100.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
    #     # 'cc100.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
    #     # 'cc100.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr5e6',
    # ]
    exp_base_dir = '/export/home/exp/search/unsup_dr/exp_v3/'
    exp_names = [
        # 'wiki.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5',
        # 'wiki.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr5e5',
        # 'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5',
        # 'cc.moco-2e14.contriever-256-prompt.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5',
        # 'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr3e5',
        # 'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen192.step200k.bs1024.lr3e5',
        # 'cc.moco-2e15.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen192.step200k.bs1024.lr3e5',
        'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr3e5',

        # 'cc.moco-2e17.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'cc.moco-2e17.updatefreq4.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'stackex.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5',
        # 'pile.1stlineastitle.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'wiki+cc+stackex.uniform.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'wiki+cc.uniform.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'wiki+cc.equal.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
    ]
    beir_datasets = [
        'msmarco',
        'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04',
        'quora', 'cqadupstack']
    # beir_datasets = [
    #     'msmarco',
    #     'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
    #     'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
    #     'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04'
    #     ]

    beir_metric_cats = ['ndcg', 'recall', 'map', 'mrr', 'precision', 'recall_cap', 'hole']
    beir_metrics = ['ndcg', 'recall', 'map', 'mrr', 'p', 'r_cap', 'hole']
    core_metric = 'ndcg@10'
    exp2scores = {}

    for exp_name in exp_names:
        print('=-' * 20)
        print(exp_name.upper())
        print('=-' * 20)
        header_row = None
        data2scores = {}

        for dataset in beir_datasets:
            if not header_row: _header_row = ['']
            score_dict = {}
            for metric_prefix in beir_metrics:
                for k in [1, 3, 5, 10, 100, 1000]:
                    score_dict[f'{metric_prefix}@{k}'] = 0.0
            data2scores[dataset] = score_dict

            score_json_path = os.path.join(exp_base_dir, exp_name, f'{dataset}.json')
            if not os.path.exists(score_json_path):
                print(f'{dataset} not found at: {score_json_path}')
            else:
                print(dataset.upper())
                with open(score_json_path, 'r') as jfile:
                    result_data = json.load(jfile)
                # print(result_data)

                for metric_prefix in beir_metric_cats:
                    for metric, score in result_data['scores'][metric_prefix].items():
                        score_dict[metric.lower()] = score

        exp2scores[exp_name] = pd.DataFrame.from_dict(data2scores)

    for exp_name, score_pd in exp2scores.items():
        print('*' * 20)
        print(exp_name)
        print(score_pd.to_csv())


if __name__ == '__main__':
    main()
