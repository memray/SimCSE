import json
import datasets

train_dataset = datasets.load_dataset("the_pile", cache_dir='/export/home/data/pretrain/.cache', split='train', streaming=True)
export_dir = '/export/home/data/pretrain/pile/'
subset2writer = {}

for i, ex in enumerate(train_dataset):
    if i % 10000 == 0:
        print(i)
    # if i >= 10000:
    #     break
    subset_name = ex['meta']['pile_set_name'].replace(' ', '_')
    if subset_name not in subset2writer:
        _writer = open(export_dir + subset_name + '.json', 'w')
        subset2writer[subset_name] = _writer
    _writer = subset2writer[subset_name]
    _writer.write(json.dumps(ex) + '\n')

print('Done')
