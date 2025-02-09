import os.path
from datetime import datetime
from functools import partial
import datasets

import numpy as np
import torch

from simcse.data_configs import load_data_config
from simcse.data_process import passage_prepare_features, hfdataset_prepare_features, wiki_prepare_features
from typing import List, Optional, TypeVar

from datasets.arrow_dataset import Dataset, _concatenate_map_style_datasets, _interleave_map_style_datasets
from datasets.info import DatasetInfo
from datasets.splits import NamedSplit
from datasets import interleave_datasets as hf_interleave_datasets
DatasetType = TypeVar("DatasetType", "Dataset", "IterableDataset")


def _interleave_map_style_datasets(
    datasets: List["Dataset"],
    num_step: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    new_fingerprint: str = None,
    **kwargs,
) -> "Dataset":
    """
    Modified based on _interleave_map_style_datasets() in datasets.arrow_dataset.py
    """
    # To interleave the datasets, we concatenate them and then we re-order the indices
    concatenated_datasets = _concatenate_map_style_datasets(datasets, info=info, split=split)

    # Let's now build the indices to pass to .select()
    lengths = [len(dset) for dset in datasets]
    offsets = np.cumsum([0] + lengths[:-1])
    if probabilities is None:
        # Example:: If lengths of the datasets are [3, 4, 5]
        # Then the resulting indices should be [0, 3, 7, 1, 4, 8, 2, 6, 9]
        # Note that we only have 3 examples per dataset since the first dataset ran out of examples
        indices = (offsets.reshape(1, -1) + np.arange(min(lengths)).reshape(-1, 1)).flatten().tolist()
    else:
        def iter_random_indices():
            """Get an infinite iterator that randomly samples the index of the source to pick examples from."""
            rng = np.random.default_rng(seed)
            while True:
                yield from (int(i) for i in rng.choice(len(datasets), size=10000000, p=probabilities))
        current_index = [0] * len(datasets)
        indices = []

        start = datetime.now()
        for source_idx in iter_random_indices():
            # keep sampling until we have enough number of training indices
            if len(indices) > num_step:
                break
            # let's add the example at the current index of the `source_idx`-th dataset
            indices.append(current_index[source_idx] + offsets[source_idx])
            current_index[source_idx] = (current_index[source_idx] + 1) % lengths[source_idx]
        end = datetime.now()
        print('Time used for indices sampling', end - start)
        print('#indices=', len(indices))

    start = datetime.now()
    merged_dataset = concatenated_datasets.select(indices, keep_in_memory=True,
                                                  writer_batch_size=10000000,
                                                  new_fingerprint=new_fingerprint,
                                                  **kwargs)
    end = datetime.now()
    print('Time used for datasets.select()', end - start)
    return merged_dataset


def interleave_datasets(
    datasets: List[DatasetType],
    num_step: int,
    probabilities: Optional[List[float]] = None,
    seed: Optional[int] = None,
    info: Optional[DatasetInfo] = None,
    split: Optional[NamedSplit] = None,
    new_fingerprint: str = None,
) -> DatasetType:
    """
    Modified based on interleave_datasets() in datasets.combine.py
    Interleave several datasets (sources) into a single dataset.
    The new dataset is constructed by alternating between the sources to get the examples.
    """
    from datasets.arrow_dataset import Dataset
    map_style = isinstance(datasets[0], Dataset)
    for dataset in datasets[1:]:
        if (map_style and not isinstance(dataset, Dataset)):
            raise ValueError(
                f"Unable to interleave a {type(datasets[0])} with a {type(dataset)}. Expected a list of Dataset objects or a list of IterableDataset objects."
            )
    return _interleave_map_style_datasets(datasets, num_step, probabilities, seed,
                                          info=info, split=split,
                                          new_fingerprint=new_fingerprint)


def load_datasets(tokenizer, training_args, model_args, hftraining_args, moco_args):
    if hftraining_args.do_train and training_args.train_file:
        # wikipedia is implemented in Apache Beam and it's not streamable
        streaming = False # if 'wiki' in training_args.train_file else True, set_transform doesn't work with IterableDataset
        train_dataset_names = training_args.train_file.split(':')
        train_datasets = []

        data_prep_config = load_data_config(training_args, hftraining_args)
        for dataset_name in train_dataset_names:
            if dataset_name.startswith('beir_'):
                beir_dataset = dataset_name[5:]
                corpus_jsonl_path = os.path.join(training_args.beir_path, beir_dataset, 'corpus.jsonl')
                print(corpus_jsonl_path)
                loaded_dataset = datasets.load_dataset("json",
                                                        data_files=corpus_jsonl_path,
                                                        keep_in_memory=False,
                                                        cache_dir=model_args.cache_dir,
                                                        streaming=streaming)
                loaded_dataset = loaded_dataset['train']
                if 'metadata' in loaded_dataset.column_names:
                    loaded_dataset = loaded_dataset.remove_columns('metadata')
                title_field, text_field = 'title', 'text'
            elif dataset_name.startswith('pile_'):
                pile_dataset = dataset_name[5:]
                corpus_jsonl_path = os.path.join('/export/home/data/pretrain/pile/', f'{pile_dataset}.json')
                # corpus_jsonl_path = os.path.join('/export/home/data/pretrain/pile/10k/', f'{pile_dataset}.json')
                print(corpus_jsonl_path)
                loaded_dataset = datasets.load_dataset("json",
                                                        data_files=corpus_jsonl_path,
                                                        keep_in_memory=False,
                                                        cache_dir=model_args.cache_dir,
                                                        streaming=streaming)
                loaded_dataset = loaded_dataset['train']
                title_field, text_field = None, 'text'
            elif dataset_name == 'c4':
                # https://huggingface.co/datasets/c4
                # #en=364,868,892, #realnewslike=13,799,838, columns=['url', 'timestamp', 'text']
                loaded_dataset = datasets.load_dataset("c4", "en", cache_dir=model_args.cache_dir,
                                                       split='train',
                                                       # split='validation',
                                                       streaming=streaming,
                                                       # download_mode="force_redownload"
                                                       )
                title_field, text_field = None, 'text'
            elif dataset_name == 'wiki':
                # https://huggingface.co/datasets/wikipedia
                # size=6,458,670, columns=['id', 'url', 'title', 'text']
                loaded_dataset = datasets.load_dataset("wikipedia", "20220301.en",
                                                       split='train', cache_dir=model_args.cache_dir,
                                                       streaming=streaming)
                title_field, text_field = 'title', 'text'
            elif dataset_name == 'pile':
                # https://huggingface.co/datasets/the_pile
                loaded_dataset = datasets.load_dataset("the_pile", cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
            elif dataset_name == 'owt2':
                # dataset_size=63.8G
                # train=17,103,059, columns=['title', 'text', 'reddit_scores']
                loaded_dataset = datasets.load_dataset("the_pile_openwebtext2", cache_dir=model_args.cache_dir, split='train', streaming=streaming, features=['title', 'text'])
                title_field, text_field = 'title', 'text'
            elif dataset_name == 'pmc':
                # 180.55 GiB
                loaded_dataset = datasets.load_dataset("the_pile", subsets=['pubmed_central'], cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
            elif dataset_name == 'stackex':
                # https://huggingface.co/datasets/the_pile_stack_exchange
                # train=5,096,117, columns=['domain', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_stack_exchange", cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
            elif dataset_name == 'books3':
                # https://huggingface.co/datasets/the_pile_books3
                # train=196,640, columns=['title', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_books3", cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
            else:
                loaded_dataset = datasets.load_dataset("text",
                                                        data_files=dataset_name,
                                                        keep_in_memory=False,
                                                        cache_dir=model_args.cache_dir,
                                                        streaming=streaming)
            train_datasets.append(loaded_dataset)

        if training_args.train_prob == 'uniform':
            probs = [len(d) for d in train_datasets]
        elif training_args.train_prob and ':' in training_args.train_prob:
            probs = [float(p) for p in training_args.train_prob.split(':')]
        else:
            if len(train_dataset_names) > 1:
                print(f'training_args.train_prob is not given, using even sampling probability for {len(train_dataset_names)} datasets: {train_dataset_names}')
            probs = [1 for _ in train_dataset_names]
            training_args.train_prob = 'equal'

        total_train_batch_size = (
                hftraining_args.train_batch_size
                * hftraining_args.gradient_accumulation_steps
                * (torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)
                * moco_args.queue_update_steps
        )
        num_examples = total_train_batch_size * (hftraining_args.max_steps + 100)

        prob_sum = sum(probs)
        sample_probs = [p / prob_sum for p in probs]
        fingerprint_name = '-'.join(['data_['+str(training_args.train_file).replace('pile_', '')+']', 'prob_'+str(probs), 'num_'+str(num_examples)]).replace(':', ',')
        # train_dataset = train_datasets[0]
        if hftraining_args.local_rank == 0 or hftraining_args.local_rank == -1:
            print(f"  - Fingerprint_name = {fingerprint_name}")
            print(f"  - train_datasets = {training_args.train_file}")
            for d_id, dataset in enumerate(train_datasets):
                print(f"  Num examples in dataset {d_id + 1} = {len(dataset)}")
            print(f"  - prob style = {training_args.train_prob}")
            print(f"  - prob = {probs}")
            print(f"  - Total train_batch_size = {total_train_batch_size}")
            print(f"  \t train_batch_size = {hftraining_args.train_batch_size}")
            print(f"  \t gradient_accumulation_steps = {hftraining_args.gradient_accumulation_steps}")
            print(f"  \t world_size = {(torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)}")
            print(f"  \t queue_update_steps = {moco_args.queue_update_steps}")
            print(f"  - Total optimization steps = {hftraining_args.max_steps}")
            print(f"  \t max_steps = {hftraining_args.max_steps}")
            print(f"  - Total training examples = {num_examples}")

        # train_dataset = hf_interleave_datasets(train_datasets, probabilities=sample_probs, seed=hftraining_args.seed)
        train_dataset = interleave_datasets(train_datasets,
                                            num_step=num_examples, probabilities=sample_probs,
                                            seed=hftraining_args.seed, new_fingerprint=fingerprint_name[:64])
        # interleave_datasets() samples examples in sequence, so still need to shuffle its order for batching
        train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)

        data_prep_config = load_data_config(training_args, hftraining_args)
        parse_fn = partial(hfdataset_prepare_features, tokenizer=tokenizer,
                           padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                           max_seq_length=training_args.max_seq_length,
                           title_field=title_field,
                           text_field=text_field,
                           **data_prep_config
                           )
        train_dataset = train_dataset.with_transform(parse_fn)
        psg_parse_fn = partial(passage_prepare_features, tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest')

        # load a subset of wikipedia as devset
        if training_args.dev_file:
            dev_dataset = datasets.load_dataset("csv",
                                                data_files={"dev": training_args.dev_file},
                                                keep_in_memory=False,
                                                cache_dir=model_args.cache_dir,
                                                delimiter="\t" if "tsv" in training_args.dev_file else ",",
                                                split='dev')
            dev_dataset.set_transform(psg_parse_fn)
        else:
            dev_dataset = None
    else:
        train_dataset = None
        dev_dataset = None

    return train_dataset, dev_dataset
