
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

        # self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

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


class FakeNewsNetWithEntity(Dataset):
    def __init__(self, data_path, corpus) -> None:
        super().__init__()

        self.data_path = data_path

        self.texts, self.entities, self.labels, self.seq_lens = self.read_data(data_path, corpus)

        self.max_len = 512

    def read_data(self, data_path, corpus):# TODO tokenize
        max_len = 512
        labels, texts, seq_lens, weights, entities = [], [], [], [], []
        for i, data_file in data_file_gen(data_path, corpus):
            label = data_file.split('/')[-2]
            label = label2id.get(label, -1)

            text, entity_seq = read_text_with_entity(data_file)  # type: ignore
            if text is None or len(text.split()) == 0:  
                continue
            text_split = text.split()
            if len(text_split) > max_len:
                text = " ".join(text_split[:max_len])

            labels.append(label)
            texts.append(text)
            seq_lens.append(len(text))
            entities.append(entity_seq)

        return texts, entities, labels, seq_lens
    
    def __getitem__(self, idx):
        return self.texts[idx], self.entities[idx], self.labels[idx], self.seq_lens[idx]

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


def read_text_with_entity(data_file):
    try:
        with open(data_file, 'r') as f:
            line = f.readline()
            text = line.strip()

            entity_seq = []
            for line in f.readlines():
                try:
                    item = eval(line.strip())
                except SyntaxError:
                    continue
                
                entity_seq.extend(item['entity_name'].split())
            # # sort by index
            # entity_seq, entity_dct = [], {}
            # for line in f.readlines():
            #     try:
            #         item = eval(line.strip())
            #     except SyntaxError:
            #         continue
                
            #     entity_dct[item['entity_name']] = item['index'][0]

            #     entity_sort = sorted(entity_dct.items(), key=lambda x:x[1])

            # for e in entity_sort:
            #     entity_seq.extend(e.split())

            return text, ' '.join(entity_seq)
    except FileNotFoundError:
        return


def data_file_gen(data_path, corpus=None):
    if corpus is not None:
        for label in os.listdir(f'{data_path}/'):
            for i, file_name in enumerate(os.listdir(f'{data_path}/{label}')):
                data_file = f'{data_path}/{label}/{file_name}'
                yield i, data_file


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
    def __init__(self, tokenizer, token_idx, entity_idx, label_idx, sort_key):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 
        self.entity_idx = entity_idx

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
            elif i == self.entity_idx:
                inputs = self.tokenizer(samples,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").values()
                if len(inputs) == 2:  # roberta
                    entity_ids, _ = inputs
                elif len(inputs) == 3:  # bert
                    entity_ids, _, _ = inputs
                else:
                    raise RuntimeError
                ret.append(entity_ids)
            else:
                ret.append(torch.tensor(samples))
        # input_ids, attention_mask, label_ids, seq_lens
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)


class TokenizedWithPromptCollator():
    def __init__(self, tokenizer, token_idx, label_idx, sort_key, only_mask=False, use_learnable_token=True, using_prefix=True, using_postfix=False):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer
        self.mask_ids = tokenizer.mask_token_id

        self.only_mask = only_mask
        self.use_learnable_token = use_learnable_token

        if self.only_mask:
            self.prefix_prompt = "<mask>"
        else:
            self.prefix_prompt = "Here is a piece of news with <mask> information . "
            # self.prefix_prompt = "Here is a piece of news with [MASK] information . "
        self.postfix_prompt = " This article is <mask> news ."
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        self.postfix_ids = self.tokenizer(self.postfix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.postfix_ids = self.postfix_ids[0][1:]  # ignore <cls>

        # the last id is <mask>, we use the last but one token as unused token
        if self.use_learnable_token:
            self.unused_ids = torch.tensor([-1])
        else:
            self.unused_ids = torch.tensor([], dtype=torch.int) 
        self.cls_id = torch.tensor([self.prefix_ids[0]])
        self.eos_id = torch.tensor([self.postfix_ids[-1]])

        # add learnable token ids, example(prefix): 
        # <cls> <learnable 0> Here is a piece of news with <mask> information . <learnable 1>
        # ==> [cls_id, learnable_ids, prompt_ids ..., learnable_ids]
        if using_prefix:
            self.prefix_ids = torch.cat([self.cls_id, self.unused_ids, self.prefix_ids[1:], self.unused_ids], dim=0)
        else:
            self.prefix_ids = self.cls_id
        # all learnable
        # self.prefix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.prefix_ids[10//2] = self.mask_ids 
        # self.prefix_ids = torch.cat([self.cls_id, self.prefix_ids], dim=0)

        if using_postfix:
            self.postfix_ids = torch.cat([self.unused_ids, self.postfix_ids[:-1], self.unused_ids, self.eos_id], dim=0)
        else:
            self.postfix_ids = self.eos_id
        # all learnable
        # self.postfix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.postfix_ids[10//2] = self.mask_ids
        # self.postfix_ids = torch.cat([self.postfix_ids, self.eos_id], dim=0)

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
                    inputs = self.tokenizer(sample,
                                                                padding=False,
                                                                truncation=False,
                                                                return_tensors="pt").values()
                    if len(inputs) == 2:  # roberta
                        input_ids, attention_mask = inputs
                    elif len(inputs) == 3:  # bert
                        input_ids, _, attention_mask = inputs
                    else:
                        raise RuntimeError
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


class PromptTokenzierWithEntityCollator():
    def __init__(self, tokenizer, token_idx, entity_idx, label_idx, sort_key, only_mask=False, use_learnable_token=True, using_prefix=True, using_postfix=False):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 
        self.entity_idx = entity_idx

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer
        self.mask_ids = tokenizer.mask_token_id

        self.only_mask = only_mask
        self.use_learnable_token = use_learnable_token

        if self.only_mask:
            self.prefix_prompt = "<mask>"
        else:
            self.prefix_prompt= "Here is a piece of news with <mask> information . "
            # self.prefix_prompt = "Here is <mask> news. "
            # self.prefix_prompt = "Detect <mask> news: "
        self.postfix_prompt = " This article is <mask> news ."
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        self.postfix_ids = self.tokenizer(self.postfix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.postfix_ids = self.postfix_ids[0][1:]  # ignore <cls>

        # the last id is <mask>, we use the last but one token as unused token
        if self.use_learnable_token:
            self.unused_ids = torch.tensor([-1])
        else:
            self.unused_ids = torch.tensor([], dtype=torch.int) 
        self.cls_id = torch.tensor([self.prefix_ids[0]])
        self.eos_id = torch.tensor([self.postfix_ids[-1]])

        # add learnable token ids, example(prefix): 
        # <cls> <learnable 0> Here is a piece of news with <mask> information . <learnable 1>
        # ==> [cls_id, learnable_ids, prompt_ids ..., learnable_ids]
        if using_prefix:
            self.prefix_ids = torch.cat([self.cls_id, self.unused_ids, self.prefix_ids[1:], self.unused_ids], dim=0)
        else:
            self.prefix_ids = self.cls_id
        # all learnable
        # self.prefix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.prefix_ids[10//2] = self.mask_ids 
        # self.prefix_ids = torch.cat([self.cls_id, self.prefix_ids], dim=0)

        if using_postfix:
            self.postfix_ids = torch.cat([self.unused_ids, self.postfix_ids[:-1], self.unused_ids, self.eos_id], dim=0)
        else:
            self.postfix_ids = self.eos_id
        # all learnable
        # self.postfix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.postfix_ids[10//2] = self.mask_ids
        # self.postfix_ids = torch.cat([self.postfix_ids, self.eos_id], dim=0)

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
                    inputs = self.tokenizer(sample,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    if len(inputs) == 2:  # roberta
                        input_ids, attention_mask = inputs
                    elif len(inputs) == 3:  # bert
                        input_ids, _, attention_mask = inputs
                    else:
                        raise RuntimeError
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
            elif i == self.entity_idx:
                inputs = self.tokenizer(samples,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").values()
                if len(inputs) == 2:  # roberta
                    entity_ids, _ = inputs
                elif len(inputs) == 3:  # bert
                    entity_ids, _, _ = inputs
                else:
                    raise RuntimeError
                ret.append(entity_ids)
            else:
                ret.append(torch.tensor(samples))
        # input_ids, attention_mask, label_ids, seq_lens
        return ret

    def __call__(self, batch):
        return self._collate_fn(batch)


class PTWELightWeightCollator():
    def __init__(self, tokenizer, token_idx, entity_idx, label_idx, sort_key, only_mask=False, use_learnable_token=True, using_prefix=True, using_postfix=False):
        self.token_idx = token_idx  # the index of data should be tokenized
        self.label_idx = label_idx  # the index of label 
        self.entity_idx = entity_idx

        self.sort_key = sort_key  # sort key

        self.tokenizer = tokenizer
        self.mask_ids = tokenizer.mask_token_id

        self.only_mask = only_mask
        self.use_learnable_token = use_learnable_token

        if self.only_mask:
            self.prefix_prompt = "<mask>"
        else:
            self.prefix_prompt = "Here is a piece of news with <mask> information . "
            # self.prefix_prompt = "Here is a piece of news with [MASK] information . "
        self.postfix_prompt = " This article is <mask> news ."
        self.prefix_ids = self.tokenizer(self.prefix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.prefix_ids = self.prefix_ids[0][:-1]  # ignore <\s>
        self.postfix_ids = self.tokenizer(self.postfix_prompt, padding=False, return_tensors="pt")['input_ids']
        self.postfix_ids = self.postfix_ids[0][1:]  # ignore <cls>

        # the last id is <mask>, we use the last but one token as unused token
        if self.use_learnable_token:
            self.unused_ids = torch.tensor([-1])
        else:
            self.unused_ids = torch.tensor([], dtype=torch.int) 
        self.cls_id = torch.tensor([self.prefix_ids[0]])
        self.eos_id = torch.tensor([self.postfix_ids[-1]])

        # add learnable token ids, example(prefix): 
        # <cls> <learnable 0> Here is a piece of news with <mask> information . <learnable 1>
        # ==> [cls_id, learnable_ids, prompt_ids ..., learnable_ids]
        if using_prefix:
            self.prefix_ids = torch.cat([self.cls_id, self.unused_ids, self.prefix_ids[1:], self.unused_ids, self.eos_id], dim=0)
        else:
            self.prefix_ids = self.cls_id
        # all learnable
        # self.prefix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.prefix_ids[10//2] = self.mask_ids 
        # self.prefix_ids = torch.cat([self.cls_id, self.prefix_ids], dim=0)

        if using_postfix:
            self.postfix_ids = torch.cat([self.unused_ids, self.postfix_ids[:-1], self.unused_ids, self.eos_id], dim=0)
        else:
            self.postfix_ids = self.eos_id
        # all learnable
        # self.postfix_ids = torch.cat([self.unused_ids]*10, dim=0)
        # self.postfix_ids[10//2] = self.mask_ids
        # self.postfix_ids = torch.cat([self.postfix_ids, self.eos_id], dim=0)

        self.add_len = int(len(self.postfix_ids))
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
                    inputs = self.tokenizer(sample,
                                            padding=False,
                                            truncation=False,
                                            return_tensors="pt").values()
                    if len(inputs) == 2:  # roberta
                        input_ids, attention_mask = inputs
                    elif len(inputs) == 3:  # bert
                        input_ids, _, attention_mask = inputs
                    else:
                        raise RuntimeError
                    input_ids = input_ids[0][0:-1]
                    attention_mask = attention_mask[0][0:-1]
                    if len(input_ids) > self.max_len:
                        input_ids = input_ids[:self.max_len]    
                        attention_mask = attention_mask[:self.max_len]                                    
                    input_ids = torch.cat([input_ids, self.postfix_ids], dim=0)
                    attention_mask = torch.cat([attention_mask, self.add_attention_mask], dim=0)
                        
                    input_ids_lst.append(input_ids)
                    attention_mask_lst.append(attention_mask)
                # input_ids = torch.tensor(input_ids_lst)
                # attention_mask = torch.tensor(attention_mask_lst)

                input_ids = rnn_utils.pad_sequence(input_ids_lst, batch_first=True)
                attention_mask = rnn_utils.pad_sequence(attention_mask_lst, batch_first=True)
                # max_len = input_ids.shape[1]
                ret.append([input_ids, self.prefix_ids.unsqueeze(0).repeat(input_ids.shape[0], 1)] ) 
                ret.append(attention_mask)
            elif i == self.entity_idx:
                inputs = self.tokenizer(samples,
                                        padding=True,
                                        truncation=True,
                                        return_tensors="pt").values()
                if len(inputs) == 2:  # roberta
                    entity_ids, _ = inputs
                elif len(inputs) == 3:  # bert
                    entity_ids, _, _ = inputs
                else:
                    raise RuntimeError
                ret.append(entity_ids)
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

