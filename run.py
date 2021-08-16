
from re import I
from torch.optim import Adam
from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import KFold
from transformers import RobertaTokenizer

from data import FakeNewNet, TokenizedCollator
from trainer import Trainer
from utils import load_config, set_seed, train_val_split
from model import BertFinetune


if __name__ == "__main__":
    data_dir = r"E:\FakeNewsNet\code\fakenewsnet_dataset"
    dataset = "politifact"
    data_path = data_dir + "\\" + dataset

    config = load_config("config/finetune.ini")
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    set_seed(trainer_config["seed"])

    data = FakeNewNet(data_path)

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    tokenized_collator = TokenizedCollator(tokenizer, token_idx=0, label_idx=1, sort_key=lambda x:x[2])

    trainer = Trainer(trainer_config)
    model = BertFinetune(model_config)

    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        bound = int(len(train_ids)*0.1)
        train_ids, val_ids = train_ids[:bound], train_ids[bound:]

        # Sample elements randomly from a given list of ids, no replacement.
        train_subsampler = SubsetRandomSampler(train_ids)
        val_subsampler = SubsetRandomSampler(val_ids)
        test_subsampler = SubsetRandomSampler(test_ids)
        
        # Define data loaders for training and testing data in this fold
        train_iter = DataLoader(dataset=data, 
                                 batch_size=data_config["batch_size"], 
                                 sampler=train_subsampler,
                                 collate_fn=tokenized_collator)

        val_iter = DataLoader(dataset=data, 
                              batch_size=data_config["batch_size"], 
                              sampler=val_subsampler,
                              collate_fn=tokenized_collator)

        test_iter = DataLoader(dataset=data,
                                batch_size=data_config["batch_size"], 
                                sampler=test_subsampler,
                                collate_fn=tokenized_collator)

        trainer.train(train_iter=train_iter, val_iter=val_iter, model=model)