import json

input_wiki_file = '/export/home/data/pretrain/pile/Wikipedia.json'
output_wiki_file = '/export/home/data/pretrain/pile/Wikipedia_dedup.json'

url2row = {}
num_rows = 0
with open(input_wiki_file, 'r') as input:
    for r in input:
        ex = json.loads(r)
        title = [l.strip() for l in ex['text'].split('\n') if len(l.strip()) > 0][0]
        url2row[title] = r
        num_rows += 1

print('#row=', num_rows)
print('#dedup_row=', len(url2row))

with open(output_wiki_file, 'w') as output:
    for r in url2row.values():
        output.write(r + '\n')
