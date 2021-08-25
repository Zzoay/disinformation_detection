
import os
import json
import operator

import torch
import torch.nn.utils.rnn as rnn_utils
from torch.utils.data import Dataset
from transformers import RobertaTokenizer


label2id = {"real":0, "fake":1}


class FakeNewsNet(Dataset):
    def __init__(self, data_path) -> None:
        super().__init__()

        self.data_path = data_path

        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

        self.texts, self.labels, self.seq_lens, self.label_weigths = self.read_data(data_path)

        self.max_len = 512

    def read_data(self, data_path):# TODO tokenize
        max_len = 512
        labels, texts, seq_lens, weights = [], [], [], []
        for data in record_files_gen(data_path):
            label = label2id.get(data[0], -1)
            text = read_text(data[1])
            if text is None or len(text.split()) == 0:  # TODO configuration
                continue
            text_split = text.split()
            if len(text_split) > max_len:
                text = " ".join(text_split[:max_len])

            if label == 0:
                weights.append(1/4)  # TODO it is for gossipcop, should be configured
            else:
                weights.append(3/4)

            labels.append(label)
            texts.append(text)
            seq_lens.append(len(text))

        return texts, labels, seq_lens, weights
    
    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx], self.seq_lens[idx]

    def __len__(self):
        return len(self.labels)


# class FakeNewNetWithPrompt(Dataset):
#     def __init__(self, data_path) -> None:
#         super().__init__()

#         self.data_path = data_path

#         self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

#         self.texts, self.labels, self.seq_lens = self.read_data(data_path)

#     def read_data(self, data_path):# TODO tokenize
#         labels, texts, seq_lens = [], [], []
#         prefix_prompt = "Here is a piece of news with <mask> information . "
#         postfix_prompt = " In general , this article is <mask> news ."
#         max_len = 510 - len(prefix_prompt.split()) - len(postfix_prompt.split())
#         for data in record_files_gen(data_path):
#             label = label2id.get(data[0], -1)
#             text = read_text(data[1])
#             if text is None:
#                 continue
#             text_split = text.split()
#             if len(text_split) > max_len:
#                 text = " ".join(text_split[:max_len])

#             text = prefix_prompt + text + postfix_prompt
#             texts.append(text)

#             labels.append(label)
#             seq_lens.append(len(text.split()))

#         print(f"max len: {max(seq_lens)}")

#         return texts, labels, seq_lens
    
#     def __getitem__(self, idx):
#         return self.texts[idx], self.labels[idx], self.seq_lens[idx]

#     def __len__(self):
#         return len(self.labels)


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
        records_path = data_path + "/" + label
        for record in os.listdir(records_path):
            record_file = records_path + "/" + record + "/news content.json"
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
                                 
        batch.sort(key=self.sort_key, reverse=True)  
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # max_len = max(len(sentence.split()) for sentence in samples)
                input_ids, attention_mask = self.tokenizer(samples,
                                                            padding=True,
                                                            truncation=True,
                                                            return_tensors="pt").values()

                # max_len = input_ids.shape[1]
                ret.append(input_ids)
                ret.append(attention_mask)
            else:
                ret.append(torch.tensor(samples))
        # input_ids, attention_mask, label_ids, seq_lens
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)


class TokenizedWithPromptCollator():
    def __init__(self, tokenizer, token_idx, label_idx, sort_key):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer

        self.prefix_prompt = "Here is a piece of news with <mask> information . "
        self.postfix_prompt = " In general , this article is <mask> news ."
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        self.postfix_ids = self.tokenizer(self.postfix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.postfix_ids = self.postfix_ids[0][1:]  # ignore <cls>

        self.add_len = int(len(self.prefix_ids) + len(self.postfix_ids))
        self.add_attention_mask = torch.ones(self.add_len)

        self.max_len = 512 - self.add_len
    
    def _collate_fn(self, batch):
        ret = []
        batch.sort(key=self.sort_key, reverse=True)  
        
        for i, samples in enumerate(zip(*batch)):
            if i == self.token_idx:
                # max_len = max(len(sentence.split()) for sentence in samples)
                input_ids_lst, attention_mask_lst = [], []
                for sample in samples:
                    input_ids, attention_mask = self.tokenizer(sample,
                                                                padding=False,
                                                                truncation=False,
                                                                return_tensors="pt").values()
                    input_ids = input_ids[0][1:-1]
                    attention_mask = attention_mask[0][1:-1]
                    if len(input_ids) > self.max_len:
                        input_ids = input_ids[:self.max_len]    
                        attention_mask = attention_mask[:self.max_len]                                    
                    input_ids = torch.cat([self.prefix_ids, input_ids, self.postfix_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, self.add_attention_mask], dim=0)
                        
                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)
                # input_ids = torch.tensor(input_ids_lst)
                # attention_mask = torch.tensor(attention_mask_lst)

                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
                # max_len = input_ids.shape[1]
                ret.append(input_ids)
                ret.append(attention_mask)
            else:
                ret.append(torch.tensor(samples))
        # input_ids, attention_mask, label_ids, seq_lens
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)

def count_len(data_path):
    lens = []
    cnt = {'real':0, 'fake':0}
    for data in record_files_gen(data_path):
        label = data[0]
        text = read_text(data[1])
        if text is None or len(text.split()) == 0:
            continue
        cnt[label] += 1  

        lens.append(len(text.split()))
    
    print(sum(lens) / len(lens))
    print(max(lens))
    print(min(lens))
    print(cnt)

if __name__ == "__main__":
    data_dir = "/home/jgy/FakeNewsNet/code/fakenewsnet_dataset"
    data_path = data_dir + "/" + "gossipcop" 

    count_len(data_path)

