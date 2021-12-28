
import random
import itertools

import numpy as np
import torch
from torch.optim import Adam
from torch.utils.data import DataLoader, WeightedRandomSampler, Subset, random_split
from sklearn.model_selection import KFold, StratifiedKFold
from torch.utils.data.dataset import T
from transformers import RobertaTokenizer, RobertaConfig, BertTokenizer, BertConfig

from data import FakeNewsNet, TokenizedCollator, TokenizedWithPromptCollator, FakeNewsNetWithEntity, PromptTokenzierWithEntityCollator
from trainer import Trainer
from utils import load_config, set_seed, train_val_split, print_measures, get_label_blance
from model import BertFineTune, BertPromptTune, BertPTWithEntity, PTWELightWeight
# from transformers import RobertaConfig


if __name__ == "__main__":
    fine_tune_all = True
    use_learnable_token = True
    with_answer_weights = True
    only_mask = False

    for note in ["# roberta"]:
        # note = "# postfix"
        if note == "# postfix":
            using_prefix = False
            using_postfix = True
            num_learnable_token = 2
        elif note == "# prefix & postfix":
            using_prefix = True
            using_postfix = True
            num_learnable_token = 4
        else:
            using_prefix = True
            using_postfix = False
            num_learnable_token = 2
        for shot in [16]:
            if shot in [100, 64, 50, 32]:
                config = load_config("config/few_shot.ini")
            elif shot in [16, 10, 8, 5, 4, 2, 1]:
                config = load_config("config/10_shot.ini")
            else:
                raise ValueError()

            data_config = config["data"]
            model_config = config["model"]
            trainer_config = config["trainer"]

            bert_config = RobertaConfig()

            for dataset in ["politifact"]:

                for mode in ["pt-with-entity"]:
                # mode = "fine-tune"
                    if isinstance(trainer_config["seed"], int):
                        seeds = [trainer_config["seed"]]
                    else:
                        seeds = trainer_config["seed"]

                    positive_words = ['true', 'real', 'actual', 'substantial', 'authentic', 'genuine', 'factual', 'correct', 'fact', 'truth']  
                    negative_words = ['false', 'fake', 'unreal', 'misleading', 'artificial', 'bogus', 'virtual', 'incorrect', 'wrong', 'fault']
                    # positive_words = ['real']
                    # negative_words = ['fake']
                    prompt_words = positive_words + negative_words

                    tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
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
                                                                                use_learnable_token = use_learnable_token)  # type: ignore   
                    elif mode == "pt-with-entity":
                        tokenized_collator = PromptTokenzierWithEntityCollator(tokenizer, 
                                                                                token_idx=0, 
                                                                                entity_idx=1,
                                                                                label_idx=2, 
                                                                                sort_key=lambda x:x[3], 
                                                                                use_learnable_token = use_learnable_token,
                                                                                only_mask=only_mask) # type: ignore   
                    else:
                        raise RuntimeError
                    # ignore the first and last token
                    pos_tokens = tokenizer(" ".join(positive_words))['input_ids'][1:-1]  # type: ignore
                    neg_tokens = tokenizer(" ".join(negative_words))['input_ids'][1:-1]  # type: ignore

                    res = []
                    for seed in seeds:
                        set_seed(seed)

                        data_path = data_config['data_dir'] + "/" + dataset
                        data = FakeNewsNetWithEntity(data_path, corpus=dataset)  # type: ignore

                        if trainer_config['cuda'] and torch.cuda.is_available:
                            torch.cuda.empty_cache()

                        ids = [i for i in range(len(data))]  # type: ignore
                        random.shuffle(ids)

                        train_ids_pool, val_ids_pool = get_label_blance(data, ids, shot)
                        
                        train_ids = train_ids_pool
                        val_ids = val_ids_pool 
                        test_ids = ids.copy()
                        for i in itertools.chain(train_ids_pool, val_ids_pool):
                            test_ids.remove(i)
                        
                        # train_ids = ids[:shot]
                        # val_ids = ids[shot:shot*2]
                        # test_ids = ids[shot*2:]

                        train_data = Subset(data, train_ids)
                        val_data = Subset(data, val_ids)
                        test_data = Subset(data, test_ids)

                        train_iter = DataLoader(dataset=train_data, 
                                                batch_size=data_config["batch_size"], 
                                                collate_fn=tokenized_collator)
                        val_iter = DataLoader(dataset=val_data, 
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
                                                    positive_token_ids = pos_tokens, 
                                                    negative_token_ids = neg_tokens,
                                                    with_learnable_emb = use_learnable_token,
                                                    with_answer_weights = with_answer_weights,
                                                    with_position_weights = False,
                                                    num_learnable_token = num_learnable_token,
                                                    zero_shot=(shot == 0),
                                                    fine_tune_all=fine_tune_all)
                        elif mode == "pt-with-entity":
                            model = BertPTWithEntity(model_config,   # type: ignore
                                                    bert_config=bert_config, 
                                                    mask_token_id=mask_token, 
                                                    positive_token_ids = pos_tokens, 
                                                    negative_token_ids = neg_tokens,
                                                    with_learnable_emb = use_learnable_token,
                                                    with_answer_weights = with_answer_weights,
                                                    with_position_weights = False,
                                                    num_learnable_token = num_learnable_token,
                                                    zero_shot=(shot == 0),
                                                    fine_tune_all=fine_tune_all)
                        else:
                            raise RuntimeError

                        trainer = Trainer(trainer_config)

                        best_res, best_model = trainer.train(train_iter = train_iter, 
                                                             val_iter = val_iter,
                                                             model = model, 
                                                             trainset_size = len(train_ids), 
                                                             batch_size = data_config['batch_size'], 
                                                             class_balance = (dataset == 'gossipcop'))
                        # torch.save(best_model, f"ckpt/few_shot/{dataset}-{mode}-{shot}.pt")  # type: ignore
                        model.load_state_dict(best_model)  # type: ignore
                        test_loss, test_metrics = trainer.evaluate(model, test_iter)  

                        # positive_weights = model.positive_weights.data.cpu()  # type: ignore
                        # negative_weights = model.negative_weights.data.cpu()  # type: ignore
                        # answer_weights = torch.cat([positive_weights, negative_weights], dim=0).numpy()  # type: ignore
                        # # output answer words weight
                        # with open("answer_weights.csv", "a+") as f:
                        #     save_weights = ",".join([str(x) for x in answer_weights])
                        #     f.write(f"{note},{shot},{dataset},{save_weights}" + "\n")

                        print("------------------------------------------")
                        print("-Test: ")
                        print_measures(test_loss, test_metrics)

                        r = [test_loss.item()]
                        r.extend([x for x in test_metrics.values()])
                        res.append(r)

                        with open("result.csv", 'a+') as f:
                            save_str = ",".join([str(x) for x in test_metrics.values()])
                            f.write(f"{note},{shot},roberta-{mode},{dataset},{test_loss}," + save_str +"\n")

                    res = np.array(res).T.tolist()  # type: ignore
                    for r in res:
                        r.remove(max(r))
                        r.remove(min(r))
                    
                    merge_res = [np.mean(x) for x in res]
                    with open("merge_res.csv", 'a+') as f:
                        save_str = ",".join([str(x) for x in merge_res])
                        f.write(f"{note},{shot},roberta-{mode},{dataset}," + save_str +"\n")