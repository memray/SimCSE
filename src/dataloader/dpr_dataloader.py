import collections
import os
import random
from functools import partial
from typing import List

import datasets
import torch
from datasets import concatenate_datasets

from src.dataloader.data_process import prepare_wiki4valid_features
from src.qa.normalize_text import normalize


BiEncoderPassage = collections.namedtuple("BiEncoderPassage", ["text", "title"])
class BiEncoderSample(object):
    query: str
    positive_passages: List[BiEncoderPassage]
    negative_passages: List[BiEncoderPassage]
    hard_negative_passages: List[BiEncoderPassage]

def _normalize_passage(ctx_text: str):
    ctx_text = ctx_text.replace("\n", " ").replace("’", "'")
    if ctx_text.startswith('"'):
        ctx_text = ctx_text[1:]
    if ctx_text.endswith('"'):
        ctx_text = ctx_text[:-1]
    return ctx_text

def _create_passage(ctx: dict, normalize: bool=False):
    return BiEncoderPassage(
        _normalize_passage(ctx["text"]) if normalize else ctx["text"],
        ctx["title"],
    )

def _parse_dpr_json(json_sample, exclude_gold=False):
    r = BiEncoderSample()
    r.query = json_sample["question"].replace("’", "'")
    positive_ctxs = json_sample["positive_ctxs"]
    if exclude_gold:
        ctxs = [ctx for ctx in positive_ctxs if "score" in ctx]
        if ctxs:
            positive_ctxs = ctxs
    negative_ctxs = json_sample["negative_ctxs"] if "negative_ctxs" in json_sample else []
    hard_negative_ctxs = json_sample["hard_negative_ctxs"] if "hard_negative_ctxs" in json_sample else []
    for ctx in positive_ctxs + negative_ctxs + hard_negative_ctxs:
        if "title" not in ctx:
            ctx["title"] = None

    r.positive_passages = [_create_passage(ctx) for ctx in positive_ctxs]
    r.negative_passages = [_create_passage(ctx) for ctx in negative_ctxs]
    r.hard_negative_passages = [_create_passage(ctx) for ctx in hard_negative_ctxs]


def nq_data_prepare(examples, tokenizer, negative_strategy):
    num_example = len(examples['question'])
    sep_token = tokenizer.sep_token if tokenizer.sep_token else '[SEP]'
    try:
        # sents0=q+, sents1=d+, sents2=q-, sents3=d-
        sents0, sents1, sents2, sents3 = [], [], [], []
        for i in range(num_example):
            q = examples['question'][i]
            # ans = examples['answers'][i][0] if 'answers' in examples else ''
            pos_d = examples['positive_ctxs'][i][0]
            title = _normalize_passage(pos_d['title']) if 'title' in pos_d else ''
            text = _normalize_passage(pos_d['text'])
            d = title + sep_token + text if 'title' in pos_d else text
            sents0.append(q)
            sents1.append(d)
            # prepare a negative example, TODO: add option of sampling among multiple
            if 'hard_negative_ctxs' in examples and len(examples['hard_negative_ctxs'][i]) > 0:
                if negative_strategy == 'first':
                    neg_d = examples['hard_negative_ctxs'][i][0]
                else:
                    neg_d = random.choice(examples['hard_negative_ctxs'][i])
            elif 'negative_ctxs' in examples and len(examples['negative_ctxs'][i]) > 0:
                neg_d = random.choice(examples['negative_ctxs'][i])
            else:
                neg_d = None
            title = _normalize_passage(neg_d['title']) if 'title' in neg_d else ''
            text = _normalize_passage(neg_d['text'])
            d = title + sep_token + text if 'title' in pos_d and len(title) > 0 else text
            sents2.append(q)
            sents3.append(d)
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'sentences': [[]]}

    if not all(len(sents0) == len(s) for s in [sents1, sents2, sents3]):
        sentences = sents0 + sents1
    else:
        sentences = sents0 + sents1 + sents2 + sents3

    return {'sentences': [sentences]}


def mm_data_prepare(examples, tokenizer, normalize_text=True,
                    dq_prompt_ratio=0.0,
                    negative_strategy='first', hard_negative_ratio=0.0, hard_negative_num=-1):
    num_example = len(examples['question'])
    sep_token = tokenizer.sep_token if tokenizer.sep_token else '[SEP]'
    try:
        # sents0=q+, sents1=d+, sents2=q-, sents3=d-
        sents0, sents1, sents2, sents3 = [], [], [], []
        for i in range(num_example):
            q = examples['question'][i]
            # ans = examples['answers'][i][0] if 'answers' in examples else ''
            pos_d = examples['positive_ctxs'][i][0]
            id_key = None
            if 'id' in pos_d:
                id_key = 'id'
            elif 'passage_id' in pos_d:
                id_key = 'passage_id'
            title = pos_d['title'].strip() if 'title' in pos_d else ''
            text = pos_d['text']
            d = title + sep_token + text if title else text
            if random.uniform(0, 1) < dq_prompt_ratio:
                q = '[Q]' + q
                d = '[D]' + d
            if normalize_text:
                q = normalize(q)
                d = normalize(d)
            sents0.append(q)
            sents1.append(d)
            # prepare a negative example
            neg_d = None
            if 'negative_ctxs' in examples and len(examples['negative_ctxs'][i]) > 0:
                neg_ds = [d for d in examples['negative_ctxs'][i] if d[id_key] != pos_d[id_key]] if id_key else [d for d in examples['negative_ctxs'][i]]
                neg_d = neg_ds[0] if negative_strategy == 'first' else random.choice(neg_ds)
            # prepare a hard negative example, depending on hard_negative_ratio
            if random.random() < hard_negative_ratio and 'hard_negative_ctxs' in examples and len(examples['hard_negative_ctxs'][i]) > 0:
                neg_ds = [d for d in examples['hard_negative_ctxs'][i] if d[id_key] != pos_d[id_key]] if id_key else [d for d in examples['hard_negative_ctxs'][i]]
                if hard_negative_num > 0:
                    neg_ds = neg_ds[: min(hard_negative_num, len(neg_ds))]
                neg_d = neg_ds[0] if negative_strategy == 'first' else random.choice(neg_ds)
            title = neg_d['title'].strip() if neg_d and 'title' in neg_d else ''
            text = neg_d['text'] if neg_d and 'text' in neg_d else ''
            d = title + sep_token + text if title else text
            if random.uniform(0, 1) < dq_prompt_ratio:
                q = '[Q]' + q
                d = '[D]' + d
            if normalize_text:
                q = normalize(q)
                d = normalize(d)
            sents2.append(q)
            sents3.append(d)
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'sentences': [[]]}

    if not all(len(sents0) == len(s) for s in [sents1, sents2, sents3]):
        sentences = sents0 + sents1
    else:
        sentences = sents0 + sents1 + sents2 + sents3

    return {'sentences': [sentences]}


def load_finetune_dataset(tokenizer, training_args, hftraining_args, moco_args):
    print(training_args.train_file)
    assert os.path.isfile(training_args.train_file), f'{training_args.train_file} does not exist.'
    if training_args.train_file.endswith('.json') or training_args.train_file.endswith('.jsonl'):
        train_dataset = datasets.load_dataset("json", data_files=training_args.train_file,
                                               keep_in_memory=False,
                                               cache_dir=training_args.cache_dir,
                                               streaming=False)
    elif training_args.train_file.endswith('.csv'):
        train_dataset = datasets.load_dataset("csv", data_files=training_args.train_file,
                                               keep_in_memory=False,
                                               cache_dir=training_args.cache_dir,
                                               streaming=False)
    else:
        raise NotImplementedError(f'Not supported file type of data {training_args.train_file}')

    train_dataset = train_dataset['train']

    total_train_batch_size = (
            hftraining_args.train_batch_size
            * hftraining_args.gradient_accumulation_steps
            * (torch.distributed.get_world_size() if hftraining_args.local_rank != -1 else 1)
            * moco_args.queue_update_steps
    )
    num_examples = total_train_batch_size * (hftraining_args.max_steps + 100)

    # simply duplicate the data multiple times if it needs multiple epochs
    sum_len = len(train_dataset)
    if sum_len < num_examples:
        train_dataset = [train_dataset] * (num_examples // sum_len + 1)
        train_dataset = concatenate_datasets(train_dataset)

    if 'nq' in training_args.train_file:
        parse_fn = partial(nq_data_prepare, tokenizer=tokenizer, negative_strategy=training_args.negative_strategy)
    else:
        parse_fn = partial(mm_data_prepare, tokenizer=tokenizer, dq_prompt_ratio=training_args.dq_prompt_ratio,
                           negative_strategy=training_args.negative_strategy,
                           hard_negative_ratio=training_args.hard_negative_ratio,
                           hard_negative_num=training_args.hard_negative_num)
    train_dataset = train_dataset.shuffle(seed=hftraining_args.seed)
    train_dataset.set_transform(parse_fn)

    # load a subset of wikipedia as devset
    if training_args.dev_file:
        dev_dataset = datasets.load_dataset("csv",
                                            data_files={"dev": training_args.dev_file},
                                            keep_in_memory=False,
                                            cache_dir=training_args.cache_dir,
                                            delimiter="\t" if "tsv" in training_args.dev_file else ",",
                                            split='dev')
        psg_parse_fn = partial(prepare_wiki4valid_features, tokenizer=tokenizer,
                               max_seq_length=training_args.max_seq_length,
                               padding_strategy='max_length' if training_args.pad_to_max_length else 'longest')
        dev_dataset.set_transform(psg_parse_fn)
    else:
        dev_dataset = None

    return train_dataset, dev_dataset
