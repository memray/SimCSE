import json
import os.path
import pandas as pd

'''
4 datasets are not directly available by BEIR: 'bioasq', 'signal1m', 'robust04', 'trec-news'
2 datasets are not in leaderboards by Mar 19, 2022: 'cqadupstack', 'quora'
'''

def main():
    exp_base_dir = '/export/home/exp/search/unsup_dr/wikipsg_v1/'
    exp_names = [
        # 'cc.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
        # 'cc.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
        # 'cc+wikipsg.equal.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs2048.lr5e5',
        # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs2048.lr5e5',
        # 'cc.inbatch.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'cc.inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'cc.inbatch-indep.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'rerun.cc.moco-2e14.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step200k.bs2048.lr5e5',
        # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd224.step200k.bs2048.lr5e5',
        # 'cc.moco-2e17.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs2048.lr5e5',

        # 'wikipsg.seed477.inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'wikipsg.seed477.inbatch.contriever-256-Qtitle50.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'wikipsg.seed477.inbatch.contriever-256-Qtitle1.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'wikipsg.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'wikipsg.seed477.moco-2e14.contriever-256-Qtitle50.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',
        # 'wikipsg.seed477.moco-2e14.contriever-256-Qtitle1.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5',

        # 'wiki_allphrase1.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki_allphrase1.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_allphrase3.seed477.inbatch.bert-base-uncased.avg.dot.d128d256.step100k.bs1024.lr5e5',
        # 'wiki_allphrase3.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_allphrase5.seed77.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki_allphrase5.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

        # 'paq.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'paq.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5'

        # 'wiki_T03b_topic.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_topic50.seed477.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_topic.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_topic50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki.T03b_topic50.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki.T03b_topic50.seed477.moco-2e14.wikipsg256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

        # 'wiki_T03b_title.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_title50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_title.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_title50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_absum.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_absum50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_absum.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_absum50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

        # 'wiki_T03b_exsum.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_exsum50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_exsum50.seed77.inbatch.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_exsum.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_T03b_exsum50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',

        # 'wiki_doc2query.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_doc2query50.seed477.inbatch.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_doc2query.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki_doc2query50.seed477.moco-2e14.contriever-256.bert-base-uncased.avg.dot.maxlen256.step100k.bs1024.lr5e5',
        # 'wiki.doc2query50-t2q.seed477.moco-2e14.contriever256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',
        # 'wiki.doc2query50-t2q.seed477.moco-2e14.wikipsg256-special50.bert-base-uncased.avg.dot.q128d256.step100k.bs1024.lr5e5',

        # 'wikipsg.seed477.moco-inbatch.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
        # 'wikipsg.seed477.moco-inbatch-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
        # 'wikipsg.seed477.moco-2e17.contriever-256.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
        # 'cc+wikipsg.equal.inbatch.contriever-256-Qtitle05.bert-base-uncased.avg.dot.qd128.step100k.bs1024.lr5e5'
        # 'cc.moco-2e14.contriever-256.bert-base-uncased.avg.dot.qd128.step200k.bs1024.lr5e5'
        ]

    # exp_base_dir = '/export/home/exp/search/unsup_dr/finetune/'
    # exp_names = [
        # 'mm.inbatch-random-neg1023+1024.arch-inbatch.model-cc-title50-bs2048.avg.dot.qd184.step20k.bs1024.lr1e5',
        # 'mm.inbatch-random-neg1023+1024.arch-inbatch.model-contriever.avg.dot.qd192.step20k.bs1024.lr1e5',
        # ]

    # exp_base_dir = '/export/home/exp/search/unsup_dr/baselines/'
    # exp_names = [
    #     'bm25',
    # ]

    # exp_base_dir = '/export/home/exp/search/contriever/'
    # exp_names = [
    #     'fb-contriever.dot',
    #     'fb-contriever.msmarco.dot',
        # 'sup-simcse-bert-base-uncased.dot',
        # 'unsup-simcse-bert-base-uncased.dot',
        # 'unsup-simcse-bert-large-uncased.dot',
        # 'unsup-simcse-roberta-base.dot',
        # 'sup-simcse-roberta-large.dot',
        # 'sup-simcse-roberta-large.cos_sim',
        # 'unsup-simcse-roberta-large.dot',
        # 'unsup-simcse-roberta-large.cos_sim',
        # ]

    # exp_base_dir = '/export/home/exp/search/unsup_dr/exp_v3/'
    # exp_names = [
    #     'pile.1stlineastitle.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'pile+wiki.moco-2e14.contriever-256-prompt.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr3e5'
        # 'pile+wiki.moco-2e14.contriever-256-prompt-Qtitle05.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr3e5'
        # 'pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5'

        # Ablation of queue size
        # 'wikipedia.contriever-256.moco-2e10.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e13.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e16.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr5e5',

        # Ablation of lr schedule
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr5e6',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-decayed_cosine-cycle2',

        # Ablation of similarity metric
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.cosine.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.normQD.maxlen256.step200k.bs256.lr1e5',

        # Ablation of momentum
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum0',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum025',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum099',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5.momentum1',

        # Ablation of datasets
        # 'wikipedia.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'wikipedia.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
        # 'pile.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
        # 'pile.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5.lr-polynomial-power2',
        # 'c4.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'c4.contriever-256.moco-2e13.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'c4.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'cc100.contriever-256.moco-2e12.bert-base-uncased.avg.dot.maxlen256.step200k.bs512.lr1e5',
        # 'cc100.contriever-256.moco-2e14.bert-base-uncased.avg.dot.maxlen256.step200k.bs256.lr1e5',
        # 'cc100.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr1e5',
        # 'cc100.contriever-256.moco-2e17.bert-base-uncased.avg.dot.maxlen256.step200k.warmup10k.bs256.lr5e6',
    # ]

    beir_datasets = [
        'msmarco',
        'trec-covid', 'bioasq', 'nfcorpus', 'nq', 'hotpotqa',
        'fiqa', 'signal1m', 'trec-news', 'arguana', 'webis-touche2020',
        'dbpedia-entity', 'scidocs', 'fever', 'climate-fever', 'scifact', 'robust04',
        'quora', 'cqadupstack']

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

            # score_json_path = os.path.join(exp_base_dir, exp_name, f'{dataset}.json')
            score_json_path = os.path.join(exp_base_dir, exp_name, 'beir_output', f'{dataset}.json')
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
