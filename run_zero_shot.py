
import random

import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from transformers import RobertaConfig, RobertaTokenizer, BertTokenizer, BertConfig

from data import FakeNewsNet, TokenizedCollator, FakeNewsNetWithEntity, PromptTokenzierWithEntityCollator
from trainer import Trainer
from utils import load_config, set_seed, train_val_split, print_measures
from model import BertFineTune, BertPromptTune, BertPTWithEntity


if __name__ == "__main__":
    note = "# roberta"
    if note == "# prefix & postfix":
        using_prefix = False
        using_postfix = True
        num_learnable_token = 2
    elif note == "# prefix & postfix":
        using_prefix = True
        using_postfix = True
        num_learnable_token = 2
    else:
        using_prefix = True
        using_postfix = False
        num_learnable_token = 2
    few_shot = 0
    
    config = load_config("config/zero_shot.ini")
    
    data_config = config["data"]
    model_config = config["model"]
    trainer_config = config["trainer"]

    # bert_config = RobertaConfig()
    bert_config = RobertaConfig()

    # dataset = "gossipcop"
    mode = "pt-with-entity"
    # for mode in ["prompt-tune", "fine-tune"]:
    for dataset in ["politifact", "gossipcop"]:
        if isinstance(trainer_config["seed"], int):
            seeds = [trainer_config["seed"]]
        else:
            seeds = trainer_config["seed"]

        positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']  
        negative_words = ['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
        prompt_words = positive_words + negative_words

        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        mask_token = tokenizer(tokenizer.mask_token)['input_ids'][1]  # type: ignore
        if mode == "fine-tune":
            tokenized_collator = TokenizedCollator(tokenizer, 
                                                    token_idx=0, 
                                                    entity_idx=1,
                                                    label_idx=2, 
                                                    sort_key=lambda x:x[3])
        elif mode == "prompt-tune":
            tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer, 
                                                                    token_idx=0, 
                                                                    entity_idx=1,
                                                                    label_idx=2, 
                                                                    sort_key=lambda x:x[3], 
                                                                    use_learnable_token = False,
                                                                    only_mask=False) # type: ignore   
        elif mode == "pt-with-entity":                                                                     
            tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer, 
                                                                    token_idx=0, 
                                                                    entity_idx=1,
                                                                    label_idx=2, 
                                                                    sort_key=lambda x:x[3], 
                                                                    use_learnable_token = True,
                                                                    only_mask=False) # type: ignore   
        else:
            raise RuntimeError
        # ignore the first and last token
        pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1]  # type: ignore
        neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1]  # type: ignore

        for seed in seeds:
            set_seed(seed)

            data_path = data_config['data_dir'] + "/" + dataset
            data = FakeNewsNetWithEntity(data_path, corpus=dataset)  # type: ignore


            if trainer_config['cuda'] and torch.cuda.is_available:
                torch.cuda.empty_cache()

            test_data = data
            test_iter = DataLoader(dataset=test_data,
                                    batch_size=data_config["batch_size"], 
                                #    sampler=test_subsampler,
                                    collate_fn=tokenized_collator)
            
            if mode == "fine-tune":
                model = BertFineTune(config=model_config)
            elif mode == "prompt-tune":
                model = BertPromptTune(model_config,   # type: ignore
                                        bert_config=bert_config, 
                                        mask_token_id=mask_token, 
                                        positive_token_ids = pos_tokens, 
                                        negative_token_ids = neg_tokens,
                                        with_learnable_emb = False,
                                        with_answer_weights = False,
                                        num_learnable_token = num_learnable_token,
                                        zero_shot=(few_shot == 0))
            elif mode == "pt-with-entity":
                model = BertPTWithEntity(model_config,   # type: ignore
                                        bert_config=bert_config, 
                                        mask_token_id=mask_token, 
                                        positive_token_ids = pos_tokens, 
                                        negative_token_ids = neg_tokens,
                                        with_learnable_emb = True,
                                        with_answer_weights = False,
                                        with_position_weights = False,
                                        num_learnable_token = num_learnable_token,
                                        zero_shot=True,
                                        fine_tune_all=False)
            else:
                raise RuntimeError
            
            # model = TextCNN(config=model_config, vocab_size=tokenizer.vocab_size)

            trainer = Trainer(trainer_config)

            test_loss, test_metrics = trainer.evaluate(model, test_iter) 

            print("------------------------------------------")
            print("-Test: ")
            print_measures(test_loss, test_metrics)

            with open("result.csv", 'a+') as f:
                save_str = ",".join([str(x) for x in test_metrics.values()])
                f.write(f"{note},0,roberta-{mode},{dataset},{test_loss}," + save_str +"\n")