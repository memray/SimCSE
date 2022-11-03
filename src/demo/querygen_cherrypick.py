import json
import os.path
from collections import defaultdict

dir_path = '/export/home/data/search/upr/wikipsg'
file_list = [
    'T03B_wikipsg_title_shard100k.jsonl',
    'T03B_wikipsg_topic_shard100k.jsonl',
    'T03B_wikipsg_exsum_shard100k.jsonl',
    'T03B_wikipsg_absum_shard100k.jsonl',
    'doc2query-t2q-wikipsg-shard100k.jsonl',
    'doc2query-a2t-wikipsg-shard100k.jsonl',
    'doc2query-r2t-wikipsg-shard100k.jsonl',
    't5xl-insummary-wikipsg-shard100k.jsonl',
]

# dir_path = '/export/home/data/search/upr/cc'
# file_list = [
#     'T03B_PileCC_title.json',
#     'T03B_PileCC_topic.json',
#     'T03B_PileCC_exsum.json',
#     'T03B_PileCC_absum.json',
#     'PileCC-doc2query-t2q.jsonl',
# ]

id2output = defaultdict(dict)
for fname in file_list:
    path = os.path.join(dir_path, fname)
    print(fname)
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            if i >= 10: break
            d = json.loads(line)
            print(d)
            id2output[d['text']][fname] = d['output-prompt0']

for text, outputs in id2output.items():
    print('=' * 100)
    print(text[:1024])
    print('-' * 50)
    for k, v in outputs.items():
        print(k, ':', v)
    print('=' * 100)