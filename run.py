
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset
from sklearn.model_selection import KFold
from transformers import RobertaTokenizer

from data import FakeNewsNet, TokenizedCollator, TokenizedWithPromptCollator
from trainer import Trainer
from utils import load_config, set_seed, train_val_split
from model import BertFineTune, BertPromptTune, TextCNN
from transformers import RobertaConfig


if __name__ == "__main__":
    config = load_config("config/gossipcop.ini")
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    bert_config = RobertaConfig()

    set_seed(trainer_config["seed"])

    mode = "prompt-tune"

    positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'exact', 'correct', 'fact', 'truth']  
    negative_words = ['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
    prompt_words = positive_words + negative_words

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
    mask_token = tokenizer(tokenizer.mask_token)['input_ids'][1]  # type: ignore
    if mode == "fine-tune":
        tokenized_collator = TokenizedCollator(tokenizer, token_idx=0, label_idx=1, sort_key=lambda x:x[2])
    elif mode == "prompt-tune":
        tokenized_collator = TokenizedWithPromptCollator(tokenizer, token_idx=0, label_idx=1, sort_key=lambda x:x[2])  # type: ignore
    else:
        raise RuntimeError
    # ignore the first and last token
    pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1]  # type: ignore
    neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1]  # type: ignore

    datasets = ["gossipcop"]
    for dataset in datasets:
        data_path = data_config['data_dir'] + "/" + dataset
        # data = FakeNewNetWithPrompt(data_path)
        data = FakeNewsNet(data_path)

        data, val_data = train_val_split(data, val_ratio=0.1, shuffle=True)
        val_iter = DataLoader(dataset=val_data, 
                                batch_size=data_config["batch_size"], 
                                collate_fn=tokenized_collator)

        kfold = KFold(n_splits=5, shuffle=True)
        avg_loss = 0.0
        avg_metrics = {"precision":0.0, "accuracy":0.0, "recall":0.0, "f1":0.0}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            if trainer_config['cuda'] and torch.cuda.is_available:
                torch.cuda.empty_cache()

            # Sample elements randomly from a given list of ids, no replacement.
            # train_subsampler = SubsetRandomSampler(train_ids)
            # test_subsampler = SubsetRandomSampler(test_ids)

            random.shuffle(train_ids)
            random.shuffle(test_ids)

            # train_ids = train_ids[:int(len(train_ids) * 0.1)]
            # train_ids = train_ids[:100]

            train_data = Subset(data, train_ids)
            test_data = Subset(data, test_ids)

            # Define data loaders for training and testing data in this fold
            train_iter = DataLoader(dataset=train_data, 
                                    batch_size=data_config["batch_size"], 
                                    # sampler=weight_sampler,
                                    # shuffle=False,
                                    collate_fn=tokenized_collator)

            test_iter = DataLoader(dataset=test_data,
                                   batch_size=data_config["batch_size"], 
                                #    sampler=test_subsampler,
                                   collate_fn=tokenized_collator)
            
            if mode == "fine-tune":
                model = BertFineTune(config=model_config)
            elif mode == "prompt-tune":
                model = BertPromptTune(model_config,   # type: ignore
                                       bert_config=bert_config, 
                                       mask_token=mask_token, 
                                       positive_tokens = pos_tokens, 
                                       negative_tokens = neg_tokens)
            else:
                raise RuntimeError
            
            # model = TextCNN(config=model_config, vocab_size=tokenizer.vocab_size)

            trainer = Trainer(trainer_config)

            best_res, best_model = trainer.train(train_iter=train_iter, val_iter=val_iter, model=model, trainset_size=len(train_ids))
            torch.save(best_model, f"ckpt/{dataset}-finetune-fold{fold}.pt")  # type: ignore
            model.load_state_dict(best_model)  # type: ignore
            test_loss, test_metrics = trainer.evaluate(model, test_iter)  # type: ignore

            print("-Test Loss: {:.4f}  Precision: {:4f}  Recall: {:4f}  Accuracy: {:4f}  F1: {:4f}  \n"
            .format(test_loss, test_metrics['precision'], test_metrics['recall'], test_metrics['accuracy'], test_metrics['f1']))

            for k in avg_metrics.keys():
                avg_metrics[k] += test_metrics[k]
            avg_loss += test_loss

            model, trainer, best_model = None, None, None  # type: ignore

        avg_metrics = {k:v/5 for k, v in avg_metrics.items()}  # type: ignore
        avg_loss /= 5  # type: ignore
        print("-K-Fold Avg Test Loss: {:.4f}  Precision: {:4f}  Recall: {:4f}  Accuracy: {:4f}  F1: {:4f}  \n"
            .format(avg_loss, avg_metrics['precision'], avg_metrics['recall'], avg_metrics['accuracy'], avg_metrics['f1']))

        with open("result.csv", 'a+') as f:
            f.write(f"roberta-{mode},{dataset},{avg_loss},{avg_metrics['precision']},{avg_metrics['recall']},{avg_metrics['accuracy']},{avg_metrics['f1']}\n")