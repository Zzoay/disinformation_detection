
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
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

    dataset = "gossipcop"
    mode = "fine-tune"
    few_shot = 100
    if isinstance(trainer_config["seed"], int):
        seeds = [trainer_config["seed"]]
    else:
        seeds = trainer_config["seed"]

    positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']  
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

    for seed in seeds:
        set_seed(seed)

        data_path = data_config['data_dir'] + "/" + dataset
        # data = FakeNewNetWithPrompt(data_path)
        data = FakeNewsNet(data_path)

        data, val_data = train_val_split(data, val_ratio=0.1, shuffle=True)
        if few_shot is not None:
            # val_ids = len(val_data.indices)
            # val_data = Subset(val_data, val_ids)  # set the size of val data equal to the train data
            _, val_data = random_split(val_data, [len(val_data.indices) - few_shot, few_shot])
        val_iter = DataLoader(dataset=val_data, 
                                batch_size=data_config["batch_size"], 
                                collate_fn=tokenized_collator)

        num_fold = 5
        kfold = KFold(n_splits=num_fold, shuffle=True)
        avg_loss = 0.0
        avg_metrics = {"accuracy": 0, 
                        "bi_precision": 0, "bi_recall": 0, "bi_f1": 0, 
                        "micro_precision": 0, "micro_recall": 0, "micro_f1": 0, 
                        "macro_precision": 0, "macro_recall": 0, "macro_f1": 0, 
                        "weighted_precision": 0, "weighted_recall": 0, "weighted_f1": 0}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            if trainer_config['cuda'] and torch.cuda.is_available:
                torch.cuda.empty_cache()

            random.shuffle(train_ids)
            random.shuffle(test_ids)

            if few_shot is not None:
                train_ids = train_ids[:few_shot]

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

            print("-Test Loss: {:.4f}  Accuracy: {:4f}  \n" \
                  " Binary:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
                  " Micro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
                  " Macro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
                  " Weighted:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n"
                .format(best_res[0], best_res[1]['accuracy'],                               # type: ignore
                        best_res[1]['bi_precision'], best_res[1]['bi_recall'], best_res[1]['bi_f1'],  # type: ignore
                        best_res[1]['micro_precision'], best_res[1]['micro_recall'], best_res[1]['micro_f1'],  # type: ignore
                        best_res[1]['macro_precision'], best_res[1]['macro_recall'], best_res[1]['macro_f1'],  # type: ignore
                        best_res[1]['weighted_precision'], best_res[1]['weighted_recall'], best_res[1]['weighted_f1']))  # type: ignore
            for k in avg_metrics.keys():
                avg_metrics[k] += test_metrics[k]
            avg_loss += test_loss

            model, trainer, best_model = None, None, None  # type: ignore

        avg_metrics = {k:v/num_fold for k, v in avg_metrics.items()}  # type: ignore
        avg_loss /= num_fold  # type: ignore
        print("-K-Fold Avg Test Loss: {:.4f}  Accuracy: {:4f}  \n" \
              " Binary:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
              " Micro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
              " Macro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" \
              " Weighted:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n"
            .format(avg_loss, avg_metrics['accuracy'],                               # type: ignore
                    avg_metrics['bi_precision'], avg_metrics['bi_recall'], avg_metrics['bi_f1'],  # type: ignore
                    avg_metrics['micro_precision'], avg_metrics['micro_recall'], avg_metrics['micro_f1'],  # type: ignore
                    avg_metrics['macro_precision'], avg_metrics['macro_recall'], avg_metrics['macro_f1'],  # type: ignore
                    avg_metrics['weighted_precision'], avg_metrics['weighted_recall'], avg_metrics['weighted_f1']))  # type: ignore
        with open("result.csv", 'a+') as f:
            save_str = ",".join([str(x) for x in avg_metrics.values()])
            f.write(f"roberta-{mode},{dataset},{avg_loss}," + save_str +"\n")