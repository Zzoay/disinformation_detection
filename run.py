
import random

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import RobertaConfig, RobertaTokenizer, BertTokenizer, BertConfig

from data import FakeNewsNet, TokenizedCollator, TokenizedWithPromptCollator, FakeNewsNetWithEntity, PromptTokenzierWithEntityCollator
from trainer import Trainer
from utils import load_config, set_seed, train_val_split, print_measures
from model import BertFineTune, BertPromptTune, BertPTWithEntity
from transformers import RobertaConfig


if __name__ == "__main__":
    shot = 'full'
    dataset = "politifact"
    fine_tune_all = True
    use_learnable_token = True
    with_answer_weights = True
    only_mask = False
    using_prefix = True
    using_postfix = False

    config = load_config(f"config/{dataset}.ini")

    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    bert_config = RobertaConfig()

    if isinstance(trainer_config["seed"], int):
        seeds = [trainer_config["seed"]]
    else:
        seeds = trainer_config["seed"]

    positive_words = ['true', 'real', 'actual', 'substantial',
                      'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']
    negative_words = ['false', 'fake', 'unreal', 'misleading',
                      'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
    prompt_words = positive_words + negative_words

    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

    mask_token = tokenizer(tokenizer.mask_token)[
        'input_ids'][1]  # type: ignore

    # ignore the first and last token
    pos_tokens = tokenizer(" ".join(positive_words))[
        'input_ids'][1:-1]  # type: ignore
    neg_tokens = tokenizer(" ".join(negative_words))[
        'input_ids'][1:-1]  # type: ignore

    for mode in ["pt-with-entity"]:
        assert len(seeds) == 1
        set_seed(seeds[0])

        if mode == "fine-tune":
            tokenized_collator = TokenizedCollator(
                tokenizer, token_idx=0, entity_idx=1, label_idx=2, sort_key=lambda x: x[3])
        elif mode == "prompt-tune":
            # tokenized_collator = TokenizedWithPromptCollator(tokenizer,
            #                                                  token_idx=0,
            #                                                  label_idx=1,
            #                                                  sort_key=lambda x:x[2],
            #                                                  use_learnable_token = True)  # type: ignore
            tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer,
                                                                   token_idx=0,
                                                                   entity_idx=1,
                                                                   label_idx=2,
                                                                   sort_key=lambda x: x[3],
                                                                   use_learnable_token=use_learnable_token)  # type: ignore
        elif mode == "pt-with-entity":
            tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer,
                                                                   token_idx=0,
                                                                   entity_idx=1,
                                                                   label_idx=2,
                                                                   sort_key=lambda x: x[3],
                                                                   use_learnable_token=use_learnable_token,
                                                                   using_prefix=using_prefix, 
                                                                   using_postfix=using_postfix,
                                                                   only_mask = only_mask)  # type: ignore
        else:
            raise RuntimeError

        data_path = data_config['data_dir'] + "/" + dataset
        data = FakeNewsNetWithEntity(data_path, corpus=dataset)  # type: ignore

        sub_data, val_data = train_val_split(data, val_ratio=0.1, shuffle=True)
        labels = [data.labels[idx] for idx in sub_data.indices]
        data = sub_data

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
                       "weighted_precision": 0, "weighted_recall": 0, "weighted_f1": 0,
                       "auc": 0}
        for fold, (train_ids, test_ids) in enumerate(kfold.split(data)):
            print(f'FOLD {fold}')
            print('--------------------------------')

            if trainer_config['cuda'] and torch.cuda.is_available:
                torch.cuda.empty_cache()

            random.shuffle(train_ids)
            random.shuffle(test_ids)

            if shot != 'full':
                train_ids = train_ids[:shot]  # type: ignore

            train_data = Subset(data, train_ids)
            test_data = Subset(data, test_ids)

            # Define data loaders for training and testing data in this fold
            train_iter = DataLoader(dataset=train_data,
                                    batch_size=data_config["batch_size"],
                                    collate_fn=tokenized_collator)

            test_iter = DataLoader(dataset=test_data,
                                   batch_size=data_config["batch_size"],
                                   collate_fn=tokenized_collator)

            if mode == "fine-tune":
                model = BertFineTune(config=model_config)
            elif mode == "prompt-tune":
                model = BertPromptTune(model_config,   # type: ignore
                                       bert_config=bert_config,
                                       mask_token_id=mask_token,
                                       positive_token_ids=pos_tokens,
                                       negative_token_ids=neg_tokens,
                                       with_learnable_emb=use_learnable_token,
                                       with_position_weights=False,
                                       with_answer_weights=with_answer_weights,
                                       zero_shot=(shot == 0),
                                       fine_tune_all=fine_tune_all)
            elif mode == "pt-with-entity":
                model = BertPTWithEntity(model_config,   # type: ignore
                                         bert_config=bert_config,
                                         mask_token_id=mask_token,
                                         positive_token_ids=pos_tokens,
                                         negative_token_ids=neg_tokens,
                                         with_learnable_emb=use_learnable_token,
                                         with_position_weights=False,
                                         with_answer_weights=with_answer_weights,
                                         zero_shot=(shot == 0),
                                         fine_tune_all=fine_tune_all)
            else:
                raise RuntimeError

            trainer = Trainer(trainer_config)

            best_res, best_model = trainer.train(train_iter=train_iter,
                                                 val_iter=val_iter,
                                                 model=model,
                                                 trainset_size=len(train_ids),
                                                 batch_size=data_config['batch_size'],
                                                 class_balance=(dataset == 'gossipcop'))
            # torch.save(best_model, f"ckpt/{dataset}-{mode}-fold{fold}.pt")  # type: ignore
            model.load_state_dict(best_model)  # type: ignore
            test_loss, test_metrics = trainer.evaluate(
                model, test_iter)  # type: ignore

            print("------------------------------------------")
            print("-Test: ")
            print_measures(test_loss, test_metrics)
            
            # if mode in ['prompt-tune', 'pt-with-entity']:
            #     positive_weights = model.positive_weights.data.cpu()  # type: ignore
            #     negative_weights = model.negative_weights.data.cpu()  # type: ignore
            #     answer_weights = torch.cat(
            #         [positive_weights, negative_weights], dim=0).numpy()  # type: ignore
            #     # output answer words weight
            #     with open("answer_weights.csv", "a+") as f:
            #         save_weights = ",".join([str(x) for x in answer_weights])
            #         f.write(f"# roberta,full,{dataset},{save_weights}" + "\n")

            # save results of each fold
            with open("result.csv", "a+") as f:
                save_str = ",".join([str(x) for x in test_metrics.values()])
                f.write(
                    f"{shot},roberta-{mode},{dataset},{test_loss}," + save_str + "\n")

            for k in avg_metrics.keys():
                avg_metrics[k] += test_metrics[k]
            avg_loss += test_loss

            model, trainer, best_model = None, None, None  # type: ignore

        avg_metrics = {k: v/num_fold for k,   # type: ignore
                       v in avg_metrics.items()} 
        avg_loss /= num_fold  # type: ignore
        print("==========================================")
        print("-K-Fold AVG: ")
        print_measures(avg_loss, avg_metrics)
        with open("merge_res.csv", 'a+') as f:
            save_str = ",".join([str(x) for x in avg_metrics.values()])
            f.write(
                f"{shot},roberta-{mode},{dataset},{avg_loss}," + save_str + "\n")
