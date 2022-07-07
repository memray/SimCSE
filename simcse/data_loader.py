import os.path
from functools import partial

import datasets
from datasets import concatenate_datasets, interleave_datasets, Features

from simcse.data_configs import load_data_config
from simcse.data_process import passage_prepare_features, hfdataset_prepare_features, document_prepare_features


def load_datasets(tokenizer, training_args, model_args, hftraining_args):
    if hftraining_args.do_train and training_args.train_file:
        # wikipedia is implemented in Apache Beam and it's not streamable
        streaming = False # if 'wiki' in training_args.train_file else True
        train_dataset_names = training_args.train_file.split(':')
        train_datasets = []
        probs = []
        for dataset_name in train_dataset_names:
            if dataset_name.startswith('beir_'):
                beir_dataset = dataset_name[5:]
                corpus_jsonl_path = os.path.join(training_args.beir_path, beir_dataset, 'corpus.jsonl')
                loaded_dataset = datasets.load_dataset("json",
                                                        data_files=corpus_jsonl_path,
                                                        keep_in_memory=False,
                                                        cache_dir=model_args.cache_dir,
                                                        streaming=streaming)
                loaded_dataset = loaded_dataset['train']
                if 'metadata' in loaded_dataset.column_names:
                    loaded_dataset = loaded_dataset.remove_columns('metadata')
                title_field, text_field = 'title', 'text'
                if beir_dataset in ['msmarco', 'fever', 'climate-fever', 'hotpotqa', 'dbpedia-entity', 'nq']:
                    probs.append(10)
                elif beir_dataset in ['quara', 'cqadupstack', 'webis-touche2020', 'trec-covid']:
                    probs.append(5)
                else:
                    probs.append(1)
            elif dataset_name == 'c4':
                # https://huggingface.co/datasets/c4
                # #en=364,868,892, #realnewslike=13,799,838, columns=['url', 'timestamp', 'text']
                loaded_dataset = datasets.load_dataset("c4", "en", cache_dir=model_args.cache_dir,
                                                       # split='train',
                                                       split='validation',
                                                       streaming=streaming,
                                                       # download_mode="force_redownload"
                                                       )
                title_field, text_field = None, 'text'
                probs.append(50)
            elif dataset_name == 'wiki':
                # https://huggingface.co/datasets/wikipedia
                # size=6,458,670, columns=['id', 'url', 'title', 'text']
                loaded_dataset = datasets.load_dataset("wikipedia", "20220301.en",
                                                       split='train', cache_dir=model_args.cache_dir,
                                                       streaming=streaming)
                title_field, text_field = 'title', 'text'
                probs.append(50)
            elif dataset_name == 'pile':
                # https://huggingface.co/datasets/the_pile
                loaded_dataset = datasets.load_dataset("the_pile", cache_dir=model_args.cache_dir, split='train[:10%]', streaming=streaming)
                title_field, text_field = None, 'text'
                probs.append(50)
            elif dataset_name == 'owt2':
                # dataset_size=63.8G
                # train=17,103,059, columns=['title', 'text', 'reddit_scores']
                loaded_dataset = datasets.load_dataset("the_pile_openwebtext2", cache_dir=model_args.cache_dir, split='train', streaming=streaming, features=['title', 'text'])
                title_field, text_field = 'title', 'text'
                probs.append(20)
            elif dataset_name == 'pmc':
                # 180.55 GiB
                loaded_dataset = datasets.load_dataset("the_pile", subsets=['pubmed_central'], cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
                probs.append(10)
            elif dataset_name == 'stackex':
                # https://huggingface.co/datasets/the_pile_stack_exchange
                # train=5,096,117, columns=['domain', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_stack_exchange", cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
                probs.append(10)
            elif dataset_name == 'books3':
                # https://huggingface.co/datasets/the_pile_books3
                # train=196,640, columns=['title', 'text']
                loaded_dataset = datasets.load_dataset("the_pile_books3", cache_dir=model_args.cache_dir, split='train', streaming=streaming)
                title_field, text_field = None, 'text'
                probs.append(10)
            else:
                loaded_dataset = datasets.load_dataset("text",
                                                        data_files=dataset_name,
                                                        keep_in_memory=False,
                                                        cache_dir=model_args.cache_dir,
                                                        streaming=streaming)
                probs.append(10)
            train_datasets.append(loaded_dataset)

        prob_sum = sum(probs)
        probs = [p / prob_sum for p in probs]
        if len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            train_dataset = interleave_datasets(train_datasets, probabilities=probs, seed=hftraining_args.seed)
        data_prep_config = load_data_config(training_args)
        parse_fn = partial(hfdataset_prepare_features, tokenizer=tokenizer,
                           padding_strategy='max_length' if training_args.pad_to_max_length else 'longest',
                           max_seq_length=training_args.max_seq_length,
                           title_field=title_field,
                           text_field=text_field,
                           **data_prep_config
                           )
        train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)
        train_dataset.set_transform(parse_fn)

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
