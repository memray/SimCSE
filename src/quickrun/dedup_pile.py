import json
import os


def dedup(dataset_name):
    input_file = f'/export/home/data/pretrain/pile/{dataset_name}.json'
    output_file = f'/export/home/data/pretrain/pile/{dataset_name}_dedup.json'

    print(input_file)
    print(output_file)
    url2row = {}
    num_rows = 0

    with open(input_file, 'r') as input:
        for rid, r in enumerate(input):
            if rid % 100000 == 0: print(rid)
            ex = json.loads(r)
            if not ex['text'].strip(): continue
            # title = [l.strip() for l in ex['text'].split('\n') if len(l.strip()) > 0][0]
            # url2row[title] = r
            leading = ex['text'][:1024]
            # leading = r[:2048]
            url2row[leading] = r
            num_rows += 1

    print('#row=', num_rows)
    print('#dedup_row=', len(url2row))

    with open(output_file, 'w') as output:
        for r in url2row.values():
            output.write(r.strip() + '\n')

def clean_empty_lines():
    pile_dir = f'/export/home/data/pretrain/pile/'
    for filename in os.listdir(pile_dir):
        if filename.find('_dedup') >= 0:
            original_dataset_name = filename[: -6]
            input_file = pile_dir+filename
            output_file = pile_dir+filename.replace('_dedup', '_clean')
            print('=' * 50)
            print(original_dataset_name)
            print(input_file)
            print(output_file)
            clean_empty_lines_and_dump_to_file(input_file=input_file, output_file=output_file)
            print('=' * 50)


def clean_empty_lines_and_dump_to_file(input_file, output_file):
    lines = []
    line_count, valid_line_count = 0, 0
    with open(input_file, 'r') as dup_file:
        for l in dup_file:
            line_count += 1
            if not l.strip(): continue
            ex = json.loads(l)
            if len(ex['text'].strip()) < 16: continue  # filter out very short examples
            lines.append(l)
            valid_line_count += 1
    print('#in_lines=', line_count)
    print('#out_lines=', valid_line_count)
    with open(output_file, 'w') as out_file:
        for l in lines:
            out_file.write(l.strip() + '\n')

if __name__ == '__main__':
    # dataset_name = 'OpenWebText2'
    # dataset_name = 'StackExchange'
    # dataset_name = 'ArXiv'
    # dataset_name = 'USPTO_Backgrounds'
    # dataset_name = 'NIH_ExPorter'
    # dataset_name = 'PubMed_Abstracts'
    # dataset_name = 'PubMed_Central'
    # dataset_name = 'PhilPapers'
    # dataset_name = 'Enron_Emails'
    # dataset_name = 'DM_Mathematics'
    # dataset_name = 'EuroParl'
    # dataset_name = 'OpenSubtitles'
    # dataset_name = 'HackerNews'
    # dataset_name = 'BookCorpus2'
    # dataset_name = 'YoutubeSubtitles'
    # dataset_name = 'Ubuntu_IRC'
    # dedup(dataset_name)
    clean_empty_lines()