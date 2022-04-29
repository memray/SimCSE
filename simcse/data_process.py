import json
import random
from dataclasses import dataclass, field
from transformers.tokenization_utils_base import BatchEncoding, PaddingStrategy, PreTrainedTokenizerBase
from typing import Optional, Union, List, Dict, Tuple
import torch

# Prepare features
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
        if bs > 0:
            num_sent = len(features[0]['input_ids'])
        else:
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

        batch = {k: batch[k].view(bs, num_sent, -1) if k in special_keys else batch[k].view(bs, num_sent, -1)[:, 0]
                 for k in batch}

        if "label" in batch:
            batch["labels"] = batch["label"]
            del batch["label"]
        if "label_ids" in batch:
            batch["labels"] = batch["label_ids"]
            del batch["label_ids"]

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

    # sent0 is query, sent1 is doc
    sents0, sents1 = [], []
    # random crop 1st sentence as query
    for sid in valid_ids:
        sent = examples[sent0_cname][sid]
        tokens = sent.split()
        if len(tokens) > 20:
            start_idx = random.randint(0, len(tokens) - 20)
            end_idx = random.randint(start_idx, len(tokens))
            if end_idx - start_idx > 10:
                tokens = tokens[start_idx: end_idx]
            else:
                tokens = tokens[start_idx: start_idx + 10]
        sents0.append(' '.join(tokens))
        sents1.append(examples[sent0_cname][sid])

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


def document_prepare_features(examples, tokenizer, max_seq_length, padding_strategy):
    '''
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
    docs = [json.loads(e) for e in examples['text']]
    docs = [d for d in docs if len(d['sections']) > 2]  # ensure docs have more than 2 secs
    if len(docs) == 0:
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }
    sents = [[] for i in range(8)]
    for doc in docs:  # there's only one doc if this method is called by datasets.set_transform()
        # Tuple 1
        (sec, _), (nbr_sec, _) \
            = random.sample(doc['sections'], 2)  # randomly select two sections
        psg_tokens = random.sample(sec.split(' | '), k=1)[0].split()  # sample a passage from the section
        nbr_psg_tokens = random.sample(nbr_sec.split(' | '), k=1)[0].split()
        if len(psg_tokens) < 10 or len(nbr_psg_tokens) < 10:
            continue
        for sent_type_i, tokens in enumerate([psg_tokens, psg_tokens, nbr_psg_tokens, nbr_psg_tokens]):
            min_len = int(len(tokens) * 0.05)
            if sent_type_i == 1 or sent_type_i == 3:  # intentionally make query shorter
                max_len = min(int(len(tokens) * 0.25), 32, max_seq_length)
            else:
                max_len = min(int(len(tokens) * 0.5), max_seq_length)
            cropped_len = random.randint(min_len, len(tokens) - max_len - 1)
            start_idx = random.randint(0, len(tokens) - cropped_len)
            end_idx = start_idx + cropped_len
            if end_idx - start_idx > 10:
                tokens = tokens[start_idx: end_idx]
            else:
                tokens = tokens[start_idx: start_idx + 10]
            sents[sent_type_i].append(' '.join(tokens))
        # Tuple 2
        (sec, titles), (nbr_sec, nbr_titles) \
            = random.sample(doc['sections'], 2)  # randomly select two sections
        psg_tokens = random.sample(sec.split(' | '), k=1)[0].split()  # sample a passage from the section
        nbr_psg_tokens = random.sample(nbr_sec.split(' | '), k=1)[0].split()
        if len(psg_tokens) < 10 or len(nbr_psg_tokens) < 10 or len(titles) == 0 or len(nbr_titles) == 0:
            continue
        for sent_type_i, tokens in enumerate([psg_tokens, titles, nbr_psg_tokens, nbr_titles]):
            if sent_type_i == 1 or sent_type_i == 3:  # titles as query (in-place shuffle)
                sents[4 + sent_type_i].append(';'.join(random.sample(tokens, len(tokens))))
            else:
                min_len = int(len(tokens) * 0.05)
                max_len = min(int(len(tokens) * 0.5), max_seq_length)
                cropped_len = random.randint(min_len, len(tokens) - max_len - 1)
                start_idx = random.randint(0, len(tokens) - cropped_len)
                end_idx = start_idx + cropped_len
                if end_idx - start_idx > 10:
                    tokens = tokens[start_idx: end_idx]
                else:
                    tokens = tokens[start_idx: start_idx + 10]
                sents[4 + sent_type_i].append(' '.join(tokens))

    sentences = [s for slist in sents for s in slist]

    if len(sentences) == 0:  # no valid inputs
        return {
            'input_ids': [[]],
            'token_type_ids': [[]],
            'attention_mask': [[]]
        }

    # print(f'len(sentences)={len(sentences)}')

    sent_features = tokenizer(
        sentences,
        max_length=max_seq_length,
        truncation=True,
        padding=padding_strategy,
    )

    features = {}
    num_doc = len(docs)
    num_sent = len(sents)
    # flatten the features, feature.shape=num_doc * num_sent -> [num_doc, num_sent]
    for key in sent_features:
        features[key] = [[sent_features[key][i + num_doc*j] for j in range(num_sent)]
                         for i in range(num_doc)]

    return features
