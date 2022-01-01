
from typing import ContextManager, List

import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from utils import to_cuda, compute_acc, compute_measures, print_measures
from forward_calculator import FinetuneFoward, FinetuneFowardWithEntity


class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.loss_fn = cross_entropy
        self.metrics_fn = compute_measures

        # self.foward_calculator = FinetuneFoward(loss_fn=cross_entropy, metrics_fn=compute_measures)
        self.foward_calculator = FinetuneFowardWithEntity(loss_fn=cross_entropy, metrics_fn=compute_measures)

    def train(self, 
              model: nn.Module, 
              train_iter: DataLoader, 
              val_iter: DataLoader,
              trainset_size: int,
              batch_size: int,
              class_balance: bool = False):
        model.train()
        if self.config["cuda"] and torch.cuda.is_available():
            model.cuda()
        
        self.optim = Adam(model.parameters(), 
                          lr=self.config['lr'],
                          weight_decay=self.config['weight_decay']
                          )

        best_res = [0, {"accuracy": 0, 
                        "bi_precision": 0, "bi_recall": 0, "bi_f1": 0, 
                        "micro_precision": 0, "micro_recall": 0, "micro_f1": 0, 
                        "macro_precision": 0, "macro_recall": 0, "macro_f1": 0, 
                        "weighted_precision": 0, "weighted_recall": 0, "weighted_f1": 0,
                        "auc": 0}]
        best_model = None
        early_stop_cnt = 0
        step = 0
        label_cnt = {"real": 0, "fake":0}
        logits, labels = [], []  # for print
        for epoch in range(self.config["epochs"]):
            for batch in train_iter:
                logit, loss = self.foward_calculator(batch, model, cuda=self.config['cuda'], class_balance=class_balance)
  
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config['clip'])
                self.optim.step()
                self.optim.zero_grad()
                
                logits.append(logit)  # type: ignore
                # labels.append(batch[2])  # type: ignore
                labels.append(batch[3])  # type: ignore

                if step > 0 and step % self.config['print_every'] == 0:
                    print_logits = torch.cat(logits, dim=0) # type: ignore
                    print_labels = torch.cat(labels, dim=0).cuda() # type: ignore
                    print_loss, print_metrics = self.loss_fn(print_logits, print_labels), self.metrics_fn(print_logits, print_labels)  # type: ignore
                    print(f"--Epoch {epoch}, Step {step}, Loss {print_loss}")
                    print_measures(print_loss, print_metrics)
                    logits, labels = [], []
                
                if epoch > 0 and step % self.config['eval_every'] == 0:
                    avg_loss, avg_metrics = self.evaluate(model, val_iter)
                    res = [avg_loss, avg_metrics]
                    if avg_metrics['bi_f1'] > best_res[1]['bi_f1']:   # type: ignore
                        best_res = res
                        best_model = model.cpu().state_dict()
                        model.cuda()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    print("--Best Evaluation: ")
                    print_measures(best_res[0], best_res[1])
                    model.train()
                
                # if epoch > 10 and early_stop_cnt >= self.config['early_stop']:
                #     print("--early stopping, training finished.")
                #     return best_res, best_model

                step += 1
            # print(label_cnt)
        if best_model is not None:  # type: ignore
            return best_res, best_model
        return best_res, model.cpu().state_dict()  # type: ignore

    # eval func
    def evaluate(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        logits, labels = [], []
        for step, batch in enumerate(eval_iter):
            logit = self.foward_calculator(batch, model, cuda=self.config['cuda'], evaluate=True)
            logits.append(logit)

            # labels.append(batch[2])
            labels.append(batch[3])

        logits = torch.cat(logits, dim=0) # type: ignore
        labels = torch.cat(labels, dim=0).cuda()  # type: ignore
        loss, metrics = self.loss_fn(logits, labels), self.metrics_fn(logits, labels)  # type: ignore
        print("--Evaluation:")

        print_measures(loss, metrics)
        if save_file != "":
            results = [save_title, avg_loss, avg_metrics.values()]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return loss, metrics  # type: ignore