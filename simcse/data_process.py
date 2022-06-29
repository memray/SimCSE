import json
import random
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
    tokenizer: PreTrainedTokenizerBase
    batch_size: int
    padding_strategy: Union[bool, str, PaddingStrategy]
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    do_mlm: bool = False
    mlm_probability: float = 0.15

    def __call__(self, features: List[Dict[str, Union[List[int], List[List[int]], torch.Tensor]]]) -> Dict[
        str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids', 'mlm_input_ids', 'mlm_labels']
        features = [f for f in features if len(f['input_ids']) > 0]
        bs = len(features)
        # print(f'bs={bs}')
        if bs > 0:
            # (IMPORTANT) pad batch to batch_size to avoid hanging in distributed training
            if bs < self.batch_size:
                features.extend([features[0]] * (self.batch_size - len(features)))
            num_sent = len(features[0]['input_ids'])
            bs = len(features)
        else:
            print('Empty batch?!')
            return
        flat_features = []
        for feature in features:
            for i in range(num_sent):
                flat_features.append({k: feature[k][i] if k in special_keys else feature[k] for k in feature})

        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding_strategy,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        if self.do_mlm:
            batch["mlm_input_ids"], batch["mlm_labels"] = self.mask_tokens(batch["input_ids"])

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys
                    else batch[k].view(bs, num_sent, -1)[:, 0] for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

        sent_lens = [len(f['attention_mask']) - f['attention_mask'][::-1].index(1) for f in flat_features]
        batch['length'] = torch.Tensor([[sent_lens[bi*num_sent + si] for si in range(num_sent)] for bi in range(bs)]).type(torch.int32)

        return batch

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


def passage_prepare_features(examples, tokenizer, max_seq_length, padding_strategy):
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
        q_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
        d_tokens = crop_sequence(tokens, max_len=128, min_len=8, min_dq_len=8, crop_to_maxlen=False)
        sents0.append(' '.join(d_tokens))  # cropped psg as doc
        sents1.append(' '.join(q_tokens))  # cropped psg as query
        # sents0.append(' '.join(tokens))
        # sents1.append(examples[sent0_cname][sid])

    # print(f'[id={examples["id"][-1]}][DOC]', len(d_tokens), sents0[-1])
    # print(f'[id={examples["id"][-1]}][QUERY]', len(q_tokens), sents1[-1])
    sentences = sents0 + sents1
    # print(f'len(sentences)={len(sentences)}')

    # If hard negative exists
    if sent2_cname is not None:
        for idx in valid_ids:
            if examples[sent2_cname][idx] is None:
                examples[sent2_cname][idx] = " "
        sentences += examples[sent2_cname]

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


def word_replace(tokens, replace_ratio, replace_with_mask=False):
    '''
    if replace_with_mask=True, tokens will be replaced with [MASK] randomly
    else it works as token deletion
    '''
    if replace_with_mask:
        _tokens = ['[MASK]' if random.random() < replace_ratio else t for t in tokens]
    else:
        _tokens = [t for t in tokens if random.random() > replace_ratio]
    return _tokens


def crop_sequence(tokens, max_len, min_len=0, min_dq_len=0,
                  crop_to_maxlen=False, return_index=False):
    '''
    if crop_to_maxlen is True, we crop the sequence to max_len, at a random position
    otherwise, the length of the cropped sequence is sampled from [min_len, max_len]
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
    min_len = min(len(tokens), max(min_dq_len, min_len))
    max_len = min(len(tokens), max(max_len, min_len))
    if crop_to_maxlen:
        cropped_len = max_len
    else:
        cropped_len = np.random.randint(min_len, max_len + 1)
    start_idx = np.random.randint(0, len(tokens) - cropped_len + 1)
    tokens = tokens[start_idx: start_idx + cropped_len]

    if return_index:
        return tokens, start_idx, start_idx + cropped_len
    else:
        return tokens


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
        # print('Error in processing text to D/Q')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(sents0) == 0 or len(sents1) == 0:
        # print('No valid text for D/Q')
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


def hfdataset_prepare_features(examples, tokenizer, max_seq_length, padding_strategy,
                               text_field, title_field=None,
                               max_context_len=256, min_dq_len=10,
                               min_q_len=0.05, max_q_len=0.5,
                               min_d_len=0.05, max_d_len=0.5,
                               include_doctitle_ratio=0.0,
                               seed=67,
                               **config_kwargs
                               ):
    try:
        texts = examples[text_field]
        titles = examples[title_field] if title_field else [''] * len(texts)
        # sent0 is doc, sent1 is query
        sents0, sents1 = [], []
        for text, title in zip(texts, titles):
            context_tokens = text.split()
            # print('=' * 20)
            # print(len(context_tokens), text)
            if max_context_len > 0:
                context_tokens = crop_sequence(context_tokens, max_len=max_context_len, crop_to_maxlen=True)
            d_tokens = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_dq_len=min_dq_len, crop_to_maxlen=False)
            d = title + ' # ' + ' '.join(d_tokens) if random.random() <= include_doctitle_ratio else ' '.join(d_tokens)
            q_tokens = crop_sequence(context_tokens, max_len=max_q_len, min_len=min_q_len, min_dq_len=min_dq_len, crop_to_maxlen=False)
            q = title + ' # ' + ' '.join(q_tokens) if random.random() <= include_doctitle_ratio else ' '.join(q_tokens)
            sents0.append(d)  # cropped psg as doc
            sents1.append(q)  # cropped psg as query
    except Exception as e:
        # print('Error in processing text to D/Q')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    if len(sents0) == 0 or len(sents1) == 0:
        # print('No valid text for D/Q')
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    total = len(sents0)
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


def document_prepare_features(examples,
                              tokenizer, max_seq_length, padding_strategy,
                              max_context_len=256, min_dq_len=10,
                              min_q_len=0.05, max_q_len=0.5,
                              min_d_len=0.05, max_d_len=0.5,
                              query_in_doc=False, q_retain_ratio=0.0,
                              word_del_ratio=0.1,
                              context_range='paragraph',
                              include_doctitle_ratio=0.0,
                              include_nbr_negative=False,
                              include_title2ctx=False,
                              seed=67
                              ):
    '''
    context_range: ['paragraph', 'section', 'document']
    for each doc in wiki, randomly sample a section as doc, and generate corresponding queries
        Tuple 1, psg2psg CL
          D+: sents[0], anchor passage, cropped ver1;
          Q+: sents[1], anchor passage, cropped ver2;
          D-: sents[2], another passage in the same doc, cropped ver1;
          Q-: sents[3], another passage in the same doc, cropped ver2;
        Tuple 2, phrs2psg CL
          D+: sents[0], anchor passage, cropped ver1;
          Q+: sents[1], section titles of anchor passage
          D-: sents[2], another passage (cropped) in the same doc
          Q-: sents[3], titles of another section in the same doc
    '''
    try:
        docs = [json.loads(e) for e in examples['text']]
        docs = [d for d in docs if len(d['sections']) > 0]
    except Exception as e:
        # print('Error in loading text to json')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}
    if len(docs) == 0:
        # print('Empty lines')
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}
    
    def _sample_D_Q(psg_tokens):
        d_tokens = crop_sequence(psg_tokens, max_len=max_d_len, min_len=min_d_len, min_dq_len=min_dq_len, crop_to_maxlen=False)
        if query_in_doc:
            q_tokens, q_start, q_end = crop_sequence(d_tokens, max_len=max_q_len, min_len=min_q_len, min_dq_len=min_dq_len,
                                                 crop_to_maxlen=False, return_index=True)
            if q_retain_ratio < 1.0 and random.random() > q_retain_ratio:
                d_tokens = d_tokens[:q_start] + ['[MASK]'] + d_tokens[q_end:]
        else:
            q_tokens = crop_sequence(psg_tokens, max_len=max_q_len, min_len=min_q_len, min_dq_len=min_dq_len, crop_to_maxlen=False)
        return d_tokens, q_tokens

    def _sample_context(_doc):
        ''' given context_range, sample a piece of context, used for sampling Q/D later'''
        if context_range == 'paragraph':
            # sample a passage from the section, passages are delimited using ' | '
            sec, titles = random.sample(_doc['sections'], 1)[0]
            psgs = [p.strip() for p in sec.split(' | ') if len(p.strip()) > 0 and len(p.split()) > min_dq_len]
            if len(psgs) == 0:
                return None
            context = random.sample(psgs, k=1)[0]
        elif context_range == 'section':
            context, titles = random.sample(_doc['sections'], 1)[0]
        elif context_range == 'document':
            titles = [_doc['title']]
            context = ' || '.join([t[0] for t in _doc['sections']])
        else:
            raise NotImplementedError(f'context_range not supported: {context_range}')
        title_tokens = (' ; '.join(random.sample(titles, len(titles)))).split()
        ctx_tokens = context.split()
        return title_tokens, ctx_tokens

    if include_title2ctx:
        sents = [[] for _ in range(8)]
    elif include_nbr_negative:
        sents = [[] for _ in range(4)]
    else:
        sents = [[] for _ in range(2)]

    try:
        for doc in docs:  # there's only one doc if this method is called by datasets.set_transform()
            # Tuple 1, cropped wiki passage as Q/D, randomly select two sections
            doc['sections'] = [s for s in doc['sections'] if s and s[0] and len(s[0].strip()) > 0 and len(s[0].split()) > min_dq_len]
            if len(doc['sections']) == 0:
                continue

            # Tuple 1 D+/Q+, sample a contiguous span from the passage
            title_tokens, ctx_full_tokens = _sample_context(doc)
            # shorten texts to max_psg_len
            context_tokens = crop_sequence(ctx_full_tokens, max_len=max_context_len, crop_to_maxlen=True)
            d, q = _sample_D_Q(context_tokens)
            if word_del_ratio:
                d = word_replace(d, replace_ratio=word_del_ratio, replace_with_mask=False)
                q = word_replace(q, replace_ratio=word_del_ratio, replace_with_mask=False)
            d = doc['title'] + ' # ' + ' '.join(d) if random.random() <= include_doctitle_ratio else ' '.join(d)
            q = doc['title'] + ' # ' + ' '.join(q) if random.random() <= include_doctitle_ratio else ' '.join(q)
            # print(f'[title,len={len(title_tokens)}]', title_tokens)
            # print(f'[ctx_full,len={len(ctx_full_tokens)}]', ' '.join(ctx_full_tokens))
            # print(f'[ctx_cropped,len={len(context_tokens)}]', ' '.join(context_tokens))
            # print(f'[Doc,len={len(d.split())}]', d)
            # print(f'[Query,len={len(q.split())}]', q)
            sents[0].append(d)
            sents[1].append(q)

            if include_nbr_negative or include_title2ctx:
                _, nbr_ctx_tokens = _sample_context(doc)
                # shorten texts to max_psg_len
                nbr_ctx_tokens = crop_sequence(nbr_ctx_tokens, max_len=max_context_len, crop_to_maxlen=True)
                # Tuple 1 D-/Q-, sample a contiguous span from the neighbor passage
                d, q = _sample_D_Q(nbr_ctx_tokens)
                if word_del_ratio:
                    d = word_replace(d, replace_ratio=word_del_ratio, replace_with_mask=False)
                    q = word_replace(q, replace_ratio=word_del_ratio, replace_with_mask=False)
                d = doc['title'] + ' # ' + ' '.join(d) if random.random() <= include_doctitle_ratio else ' '.join(d)
                q = doc['title'] + ' # ' + ' '.join(q) if random.random() <= include_doctitle_ratio else ' '.join(q)
                sents[2].append(d)
                sents[3].append(q)

            # Tuple 2, use section titles as query, will be sents[4-7]
            if include_title2ctx:
                title_tokens, context_tokens = _sample_context(doc)
                if len(context_tokens) < min_dq_len or len(title_tokens) == 0:
                    continue
                # Tuple 2 D+/Q+
                q = ' '.join(title_tokens)
                d = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_dq_len=min_dq_len, crop_to_maxlen=False)
                if word_del_ratio:
                    d = word_replace(d, replace_ratio=word_del_ratio, replace_with_mask=False)
                d = doc['title'] + ' # ' + ' '.join(d) if random.random() <= include_doctitle_ratio else ' '.join(d)
                sents[4].append(d)
                sents[5].append(q)
                # Tuple 2 D-/Q-
                title_tokens, context_tokens = _sample_context(doc)
                if len(context_tokens) < min_dq_len or len(title_tokens) == 0:
                    continue
                q = ' '.join(title_tokens)
                d = crop_sequence(context_tokens, max_len=max_d_len, min_len=min_d_len, min_dq_len=min_dq_len,
                                  crop_to_maxlen=False)
                if word_del_ratio:
                    d = word_replace(d, replace_ratio=word_del_ratio, replace_with_mask=False)
                d = doc['title'] + ' # ' + ' '.join(d) if random.random() <= include_doctitle_ratio else ' '.join(d)
                sents[6].append(d)
                sents[7].append(q)
    except Exception as e:
        pass

    sentences = [s for slist in sents for s in slist]
    if len(sentences) == 0 or (include_nbr_negative and not include_title2ctx and len(sentences) != 4) or (include_title2ctx and len(sentences) != 8):  # no valid inputs
        # print('No valid data points for Q/D')
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }
    # for i, sent_list in enumerate(sents):
    #     print(f'[sent#{i}]', f'len={len(sent_list[0].split())},', sent_list[0])
    # print('=' * 30)
    try:
        sent_features = tokenizer(
            sentences,
            max_length=max_seq_length,
            truncation=True,
            padding=padding_strategy,
            return_length=True
        )
        features = {}
        num_doc = len(docs)
        num_sent = len(sents)
        # flatten the features, feature.shape=num_doc * num_sent -> [num_doc, num_sent]
        for key in sent_features:
            features[key] = [[sent_features[key][i + num_doc*j] for j in range(num_sent)]
                             for i in range(num_doc)]
    except Exception as e:
        # print('Error in tokenizing D/Q')
        # print(e)
        return {'input_ids': [[]], 'token_type_ids': [[]], 'attention_mask': [[]]}

    return features
