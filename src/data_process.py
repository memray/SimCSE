import copy
import json
import random
import string
import sys

import numpy as np
from dataclasses import dataclass, field
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict, Tuple
import torch

# Prepare features
from src import utils

sent2_cname = None
# Unsupervised datasets
title_cname = 'title'
sectitle_cname = 'sectitles'
sent0_cname = 'text'
sent1_cname = 'text'


# Data collator
@dataclass
class PassageDataCollatorWithPadding:
# class PassageDataCollatorWithPaddingQDSeparate:
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    padding_strategy: Union[bool, str, PaddingStrategy]
    max_length: Optional[int] = None
    cap_qd_tokens: bool = False  # if true, limit len(q)+len(d) <= max_length
    max_q_tokens: Union[tuple] = None  # used if cap_qd_len==True, it's max length of q, and d_len=cap_qd_len-q_len

    def __call__(self, batch_data) -> Dict[str, torch.Tensor]:
        batch_data = [e for e in batch_data if e and 'sentences' in e and len(e['sentences']) > 0]
        bs = len(batch_data)
        if bs > 0:
            # (IMPORTANT) pad batch to batch_size to avoid hanging in distributed training
            if bs < self.batch_size:
                batch_data.extend([batch_data[0]] * (self.batch_size - len(batch_data)))
            bs = len(batch_data)
        else:
            print('Empty batch?!')
            return
        # flatten sentences and tokenize q/d separately
        sentences = [e['sentences'] for e in batch_data]
        # sources = [e['sources'][0] if 'sources' in e else '' for e in batch_data]
        # titles = sorted(sorted([e['titles'][0] if 'titles' in e and e['titles'] and e['titles'][0] else '' for e in batch_data]))
        # print('*' * 50)
        # for t, s in zip(titles, sentences):
        #     # print('\t\t', t)
        #     print('\t\t', s[0])
        num_sent_per_example = len(sentences[0])
        if num_sent_per_example == 2:  # validation
            flat_q_sents = [s for e in sentences for s in [e[0]]]
            flat_d_sents = [s for e in sentences for s in [e[1]]]
        else:  # training
            flat_q_sents = [s for e in sentences for s in [e[0], e[2]]]
            flat_d_sents = [s for e in sentences for s in [e[1], e[3]]]
        if self.cap_qd_tokens:
            assert isinstance(self.max_q_tokens, list)
            if self.max_q_tokens[0] > 0:
                max_q_len = np.random.randint(self.max_q_tokens[0], self.max_q_tokens[1])
            else:
                max_q_len = self.max_q_tokens[1]
            max_d_len = self.max_length - max_q_len
        else:
            max_q_len = self.max_length
            max_d_len = self.max_length
        q_feats = self.tokenizer(
            flat_q_sents,
            max_length=max_q_len,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        # to allow longer D sequences if actual Q is shorter
        if self.cap_qd_tokens and q_feats['input_ids'].shape[1] < max_q_len:
            max_q_len = q_feats['input_ids'].shape[1]
            max_d_len = self.max_length - max_q_len
        k_feats = self.tokenizer(
            flat_d_sents,
            max_length=max_d_len,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        # print(max_q_len, max_d_len)
        # print('q_shape=', q_feats['input_ids'].shape)
        # print('k_shape=', k_feats['input_ids'].shape)
        # unflatten sentences and return tensors, order is [Q0, D0, Q1, D1]
        batch = {}
        for key in q_feats:
            if num_sent_per_example == 2:  # training & validation
                qfeat = q_feats[key]
                kfeat = k_feats[key]
                batch[key] = [qfeat, kfeat]  # [2,B,L], order is [Q0, D0]
            else:  # training w/ extra pairs
                qfeat = q_feats[key].reshape([bs, 2, -1]).permute(1, 0, 2)  # [2*B,L]->[B,2,L]->[2,B,L]
                kfeat = k_feats[key].reshape([bs, 2, -1]).permute(1, 0, 2)  # [2*B,L]->[B,2,L]->[2,B,L]
                batch[key] = [qfeat[0], kfeat[0], qfeat[1], kfeat[1]]  # [4,B,L], order is [Q0, D0, Q1, D1]
            if key == 'attention_mask':
                sent_lens = []
                for i in range(bs):
                    sent_len_i = []
                    for attn in batch[key]:
                        sent_len_i.append(sum(attn[i]))
                    sent_lens.append(torch.Tensor(sent_len_i).type(torch.int32))
                batch['length'] = torch.stack(sent_lens)  # [B,2/4]
        return batch


@dataclass
class PassageDataCollatorWithPaddingQDTogether:
# class PassageDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    padding_strategy: Union[bool, str, PaddingStrategy]
    max_length: Optional[int] = None
    cap_qd_tokens: bool = False  # if true, limit len(q)+len(d) <= max_length
    max_q_tokens: Union[tuple] = None  # used if cap_qd_len==True, it's max length of q, and d_len=cap_qd_len-q_len

    def __call__(self, batch_data) -> Dict[str, torch.Tensor]:
        batch_data = [e for e in batch_data if e and 'sentences' in e and len(e['sentences']) > 0]
        bs = len(batch_data)
        if bs > 0:
            # (IMPORTANT) pad batch to batch_size to avoid hanging in distributed training
            if bs < self.batch_size:
                batch_data.extend([batch_data[0]] * (self.batch_size - len(batch_data)))
            bs = len(batch_data)
        else:
            print('Empty batch?!')
            return
        # flatten sentences and tokenize q/d separately
        sentences = [e['sentences'] for e in batch_data]
        # sources = [e['sources'][0] if 'sources' in e else '' for e in batch_data]
        # titles = sorted(sorted([e['titles'][0] if 'titles' in e and e['titles'] and e['titles'][0] else '' for e in batch_data]))
        # print('*' * 50)
        # for t, s in zip(titles, sentences):
        #     # print('\t\t', t)
        #     print('\t\t', s[0])
        num_sent_per_example = len(sentences[0])
        flat_q_sents = [s for e in sentences for s in [e[0]]]
        flat_d_sents = [s for e in sentences for s in [e[1]]]
        flat_sents = flat_q_sents + flat_d_sents
        feats = self.tokenizer(
            flat_sents,
            max_length=self.max_length,
            truncation=True,
            padding=self.padding_strategy,
            return_tensors="pt",
        )
        # print(max_q_len, max_d_len)
        # print('q_shape=', q_feats['input_ids'].shape)
        # print('k_shape=', k_feats['input_ids'].shape)
        # unflatten sentences and return tensors, order is [Q0, D0, Q1, D1]
        batch = {}
        for key in feats:
            qfeat = feats[key][:bs]
            kfeat = feats[key][bs:]
            batch[key] = [qfeat, kfeat]  # [2,B,L], order is [Q0, D0]
            if key == 'attention_mask':
                sent_lens = []
                for i in range(bs):
                    sent_len_i = []
                    for attn in batch[key]:
                        sent_len_i.append(sum(attn[i]))
                    sent_lens.append(torch.Tensor(sent_len_i).type(torch.int32))
                batch['length'] = torch.stack(sent_lens)  # [B,2/4]
        return batch


def _extract_title_v1(text, source):
    if source in ['Wikipedia', 'Pile-CC', 'OpenWebText2']:
        lines = [l for l in text.split('\n') if len(l.strip()) > 0]
        if len(lines) > 0: title = lines[0]
    else:
        lines = [l for l in text.split('\n') if len(l.strip().split()) > 3]
        if len(lines) > 0: title = lines[0]
    title = ' '.join(title.split()[:64])  # truncate titles, no longer than 64 words
    return title


def _extract_title(input_text, source, retain_title=False, min_len=16):
    '''
    if title is not given or not Wiki, we take the heading substring as a pseudo title
    '''
    lines = [l for l in input_text.split('\n') if len(l.strip()) > 3]
    title, text = '', input_text
    if source in ['Wikipedia', 'Pile-CC', 'OpenWebText2', 'HackerNews', 'Enron Emails', 'StackExchange', 'PubMed Abstracts']:
        if len(lines) > 0:
            title = lines[0]
            if not retain_title: text = '\n'.join(lines[1:])
    if source in ['ArXiv', 'PubMed Central']:
        input_text = input_text.strip(string.punctuation + string.whitespace + string.digits)
        if input_text.lower().startswith('abstract'): input_text = input_text[8:]
        if input_text.lower().startswith('introduction'): input_text = input_text[12:]
        if input_text.lower().startswith('background'): input_text = input_text[10:]
        input_text = input_text.strip(string.punctuation + string.whitespace).replace('==', '')
    if source == 'USPTO Backgrounds' and input_text.startswith('1. Field of the Invention'):
        input_text = input_text[25:]
    if not title or len(lines) <= 1 or (source != 'Wikipedia' and len(title.split()) <= 2):
        tokens = input_text.split()
        title = ' '.join(tokens[: min(min_len, len(tokens) // 2)])
        if not retain_title: text = ' '.join(tokens[min(min_len, len(tokens) // 2):])
    # corner case, either one is very short
    if len(title.strip()) < min_len or len(text.strip()) < min_len:
        title, text = input_text, input_text
    title = title.replace('\n', '\t').strip()
    if source == 'Enron Emails': title = title.replace('--', '')
    text = text.strip()
    return title, text


def hfdataset_prepare_features(examples, tokenizer, max_seq_length, padding_strategy,
                               text_field,
                               max_context_len=256, min_dq_len=10,
                               min_q_len=0.05, max_q_len=0.5,
                               min_d_len=0.05, max_d_len=0.5,
                               word_del_ratio=0.0,
                               dq_prompt_ratio=0.0,
                               title_as_query_ratio=0.0,
                               query_in_doc=0.0,
                               include_title_ratio=0.0,
                               **config_kwargs
                               ):
    ''' examples only contain one element, if using HF_dataset.set_transform() '''
    try:
        examples = examples['data'] if 'data' in examples else examples
        texts = examples[text_field]
        metas = examples['meta'] if 'meta' in examples else [None] * len(texts)
        titles = examples['title'] if 'title' in examples else [''] * len(texts)
        urls = examples['url'] if 'url' in examples else [''] * len(texts)
        extra_output_key = [k for k in examples.keys() if k.startswith('output-prompt')]
        extra_queries = examples[extra_output_key[0]] if len(extra_output_key) > 0 else [''] * len(texts)
        # print(f'[{len(texts[0].split())}]', texts)
        # (Q1) sent0 is title;
        # (D1) sent1 is psg.
        # (Q2) sent2 is another psg, or a section title/class label (if applicable);
        # (D2) sent3 is another psg.
        sents0, sents1, sents2, sents3 = [], [], [], []
        sources = []
        for text, title, meta, url, extra_query in zip(texts, titles, metas, urls, extra_queries):
            if not text: continue
            text = text.encode('utf-8', 'ignore').decode()
            title = title.encode('utf-8', 'ignore').decode() if title else ''
            if extra_query:
                extra_query = extra_query.encode('utf-8', 'ignore').decode()
                title = extra_query
            if url and 'wikipedia' in url:
                source = 'Wikipedia'
            elif meta and 'pile_set_name' in meta:
                source = meta['pile_set_name']
                if 'wikipedia' in source.lower(): source = 'Wikipedia'
            else:
                source = None
            sources.append(source)
            if not title:
                # title, text = _extract_title(text, source, retain_title=np.random.uniform() <= include_title_ratio)
                title = _extract_title_v1(text, source)
            # print('*' * 100)
            # print('[source]', source)
            # print('[title]', title.replace('\n', '\t'))
            # print('[text]', len(text), len(text.split()), text.replace('\n', '\t'))
            text_tokens = text.split()
            if max_context_len > 0:
                context_tokens = crop_sequence(text_tokens, max_len=max_context_len, crop_to_maxlen=True)
            else:
                context_tokens = copy.copy(text_tokens)
            # prepare for Q1/D1
            d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            d_tokens = word_replace(d_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            d = ' '.join(d_tokens)
            if title and title_as_query_ratio and np.random.uniform() < title_as_query_ratio:
                title_tokens = title.split()
                q_tokens = crop_sequence(title_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                # q_tokens = crop_sequence(title_tokens, max_len=max_context_len, min_len=min_q_len, min_cap=min_dq_len, crop_to_maxlen=True)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            elif query_in_doc and np.random.uniform() < query_in_doc:
                q_tokens = crop_sequence(d_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            else:
                q_tokens = crop_sequence(context_tokens, max_len=max_q_len, min_len=min_q_len, min_cap=min_dq_len)
                q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = ' '.join(q_tokens)
            if np.random.uniform() < dq_prompt_ratio:
                q = '[Q]' + q
                # d = '[D]' + source + '[SEP]' + d if source else '[D]' + d
                d = '[D]' + d
            sents0.append(q)
            sents1.append(d)
            # print('Q1: ', len(q_tokens), q.replace('\n', '\t'))
            # print('D1: ', len(d_tokens), d.replace('\n', '\t'))

            # prepare for Q2/D2, sample another set of Q/D in a new context
            '''
            if source == 'Wikipedia':
                sections = _parse_wiki(text, title)
                q, d = sections[np.random.randint(len(sections))]
                q_tokens, d_tokens = q.split(), d.split()
                if max_context_len > 0:
                    q_tokens = crop_sequence(q.split(), max_len=max_context_len, crop_to_maxlen=True)
                    d_tokens = crop_sequence(d.split(), max_len=max_context_len, crop_to_maxlen=True)
                q_tokens = crop_sequence(q_tokens, max_len=max_d_len, min_len=min_d_len, max_cap=max_context_len, min_cap=min_dq_len)
                d_tokens = crop_sequence(d_tokens, max_len=max_d_len, min_len=min_d_len, max_cap=max_context_len, min_cap=min_dq_len)
            else:
                context_tokens = copy.copy(text_tokens)
                if max_context_len > 0:
                    context_tokens = crop_sequence(text_tokens, max_len=max_context_len, crop_to_maxlen=True)
                q_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
                d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_cap=min_dq_len)
            q_tokens = word_replace(q_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            d_tokens = word_replace(d_tokens, replace_ratio=word_del_ratio, replace_with_mask=False)
            q = ' '.join(q_tokens)
            d = ' '.join(d_tokens)
            if np.random.uniform() < dq_prompt_ratio:
                q = '[Q]' + q
                # d = '[D]' + source + '[SEP]' + d if source else '[D]' + d
                d = '[D]' + d
            # print('Q2: ', len(q_tokens))
            # print('D2: ', len(d_tokens))
            sents2.append(q)
            sents3.append(d)
            '''
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        print(examples)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    sentences = sents0 + sents1
    if len(sentences) == 0 or (not all(len(sents0) == len(s) for s in [sents1])):
    # sentences = sents0 + sents1 + sents2 + sents3
    # if len(sentences) == 0 or (not all(len(sents0) == len(s) for s in [sents1, sents2, sents3])):
        print('No valid text for D/Q')
        print(examples)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    return {'sentences': [sentences], 'sources': [sources], 'titles': [titles]}


def prepare_wiki4valid_features(examples, tokenizer, max_seq_length, padding_strategy):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    # Avoid "None" fields
    valid_ids = [i for i in range(len(examples[sent0_cname]))
                 if examples[sent0_cname][i] is not None]
    total = len(valid_ids)
    if total == 0:
        # print('No valid text for D/Q in Wikipedia for eval')
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }
    # sent0 is doc, sent1 is query
    sents0, sents1 = [], []
    # random crop 1st sentence as query
    for sid in valid_ids:
        sent = examples[sent0_cname][sid]
        tokens = sent.split()
        q_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_cap=8, crop_to_maxlen=False)
        d_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_cap=8, crop_to_maxlen=False)
        sents0.append(' '.join(d_tokens))  # cropped psg as doc
        sents1.append(' '.join(q_tokens))  # cropped psg as query
        # sents0.append(' '.join(tokens))
        # sents1.append(examples[sent0_cname][sid])

    # print(f'[id={examples["id"][-1]}][DOC]', len(d_tokens), sents0[-1])
    # print(f'[id={examples["id"][-1]}][QUERY]', len(q_tokens), sents1[-1])
    sentences = sents0 + sents1
    # print(f'len(sentences)={len(sentences)}')
    return {'sentences': [sentences]}


def mask_tokens(
        self, inputs: torch.Tensor, special_tokens_mask: Optional[torch.Tensor] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    inputs = inputs.clone()
    labels = inputs.clone()
    # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
    probability_matrix = torch.full(labels.shape, self.mlm_probability)
    if special_tokens_mask is None:
        special_tokens_mask = [
            self.tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True) for val in
            labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
    else:
        special_tokens_mask = special_tokens_mask.bool()

    probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def word_replace(tokens, replace_ratio=0.0, replace_with_mask=False):
    '''
    if replace_with_mask=True, tokens will be replaced with [MASK] randomly
    else it works as token deletion
    '''
    if replace_ratio <= 0.0:
        return tokens
    if replace_with_mask:
        _tokens = ['[MASK]' if np.random.uniform() < replace_ratio else t for t in tokens]
    else:
        _tokens = [t for t in tokens if np.random.uniform() > replace_ratio]
    return _tokens


def crop_sequence(tokens, max_len, min_len=0,
                  max_cap=sys.maxsize, min_cap=0,
                  crop_to_maxlen=False, return_index=False):
    '''
    if crop_to_maxlen is True, we crop the sequence to max_len, at a random position
        otherwise, the length of the cropped sequence is sampled from [min_len, max_len]
        max_cap/min_cap are absolute length limit
    '''
    # directly return if sequence is shorter than max_len
    if crop_to_maxlen and len(tokens) <= max_len:
        if return_index:
            return tokens, 0, len(tokens)
        else:
            return tokens
    if 0 < max_len <= 1:
        max_len = int(len(tokens) * max_len)
    if 0 < min_len <= 1:
        min_len = int(len(tokens) * min_len)
    min_len = min(len(tokens), max(min_cap, min_len))
    max_len = min(len(tokens), max(max_len, min_len), max_cap)
    if crop_to_maxlen:
        cropped_len = max_len
    else:
        cropped_len = np.random.randint(min_len, max_len + 1)
    start_idx = np.random.randint(0, len(tokens) - cropped_len + 1)
    _tokens = tokens[start_idx: start_idx + cropped_len]

    if return_index:
        return _tokens, start_idx, start_idx + cropped_len
    else:
        return _tokens


def simple_document_prepare_features(examples, tokenizer, max_seq_length, padding_strategy):
    # padding = longest (default)
    #   If no sentence in the batch exceed the max length, then use
    #   the max sentence length in the batch, otherwise use the
    #   max sentence length in the argument and truncate those that
    #   exceed the max length.
    # padding = max_length (when pad_to_max_length, for pressure test)
    #   All sentences are padded/truncated to data_args.max_seq_length.
    # Avoid "None" fields
    docs = []
    try:
        docs = [json.loads(e) for e in examples['text']]
        docs = [d for d in docs if len(d['sections']) > 0]
    except Exception as e:
        # print('Error in loading text from json')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(docs) == 0:
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }

    total = len(docs)
    # sent0 is doc, sent1 is query
    sents0, sents1 = [], []

    try:
        for doc in docs:
            doc['sections'] = [s for s in doc['sections'] if len(s[0].strip()) > 0]
            if len(doc['sections']) == 0:
                continue
            sent = doc['sections'][0][0]
            tokens = sent.split()
            q_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
            d_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
            sents0.append(' '.join(d_tokens))  # cropped psg as doc
            sents1.append(' '.join(q_tokens))  # cropped psg as query
    except Exception as e:
        print('Error in processing text to D/Q')
        print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(sents0) == 0 or len(sents1) == 0:
        print('No valid text for D/Q')
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    sentences = sents0 + sents1
    sent_features = tokenizer(
        sentences,
        max_length=max_seq_length,
        truncation=True,
        padding=padding_strategy,
    )

    features = {}
    if sent2_cname is not None:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total], sent_features[key][i + total * 2]]
                             for i in range(total)]
    else:
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]

    return features


def _parse_wiki(text, title):
    sec_title = title
    sections = []
    sec_lines = []
    for l in text.split('\n\n'):
        l = l.strip()
        if l.startswith('See also') or l.startswith('Notes') or l.startswith('References') or l.startswith('Further reading') or l.startswith('External links'):
            break
        if len(sec_lines) > 0 and '\n' in l[:30]:
            sections.append((sec_title, '\n'.join(sec_lines)))
            sec_title = l[: l.index('\n')]
            sec_lines = [l[l.index('\n') + 1:]]
        else:
            sec_lines.append(l)
    sections.append((sec_title, '\n'.join(sec_lines)))
    return sections
