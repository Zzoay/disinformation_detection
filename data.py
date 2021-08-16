
import os
import json
import operator

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


label2id = {"real":0, "fake":1}


class FakeNewNet(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.texts, self.labels, self.seq_lens = self.read_data(data_path)

    def read_data(self, data_path):# TODO tokenize
        labels, texts, seq_lens = [], [], []
        for data in record_files_gen(data_path):
            label = label2id.get(data[0], -1)
            text = read_text(data[1])
            if text is None:
                continue
            
            labels.append(label)
            texts.append(text)
            seq_lens.append(len(text))

        return texts, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.labels)


def read_text(data_file):
    try:
        with open(data_file, 'r') as f:
            line = f.readline()
            dct = json.loads(line)
            return dct['text']
    except FileNotFoundError:
        return

def record_files_gen(data_path):
    """
    -gossipcop
    -politifact
    -real
    -fake
        -politifact31
            -news content.json
            -tweets
        -...
    """
    for label in os.listdir(data_path):
        records_path = data_path + "\\" + label
        for record in os.listdir(records_path):
            record_file = records_path + "\\" + record + "\\news content.json"
            yield label, record_file


def build_vocab(data_path):
    word_vocab = {}
    for data in record_files_gen(data_path):
        text = read_text(data[1])
        if text is None:
            continue
        for word in text.split():
            try:
                word_vocab[word] += 1
            except KeyError:
                word_vocab[word] = 1
    word_vocab = sorted(word_vocab.items(), key=operator.itemgetter(1), reverse=True)
    return word_vocab


class TokenizedCollator():
    def __init__(self, tokenizer, token_idx, label_idx, sort_key):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer
    
    def _collate_fn(self, batch):
        ret = []
        max_len = 0
        batch.sort(key=self.sort_key, reverse=True)  
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # {'input_ids':..., 'token_type_ids':..., 'attention_mask': ...}
                max_len = max(len(sentence.split()) for sentence in samples)
                input_ids, attention_mask  = self.tokenizer(samples,
                                                                     padding=True,
                                                                     truncation=True,
                                                                     return_tensors="pt").values()
                max_len = input_ids.shape[1]
                ret.append(input_ids)
                ret.append(attention_mask)
            else:
                ret.append(samples)
        # input_ids, attention_mask, label_ids, seq_lens
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)