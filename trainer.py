
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn.functional import cross_entropy

from optim import ScheduledOptim
from utils import to_cuda, compute_acc
from forward_calculator import FinetuneFoward


class Trainer():
    def __init__(self, config) -> None:
        self.config = config
        self.foward_calculator = FinetuneFoward(loss_fn=cross_entropy, metrics_fn=compute_acc)

    def train(self, 
              model: nn.Module, 
              train_iter: DataLoader, 
              val_iter: DataLoader):
        model.train()
        if self.config["cuda"] and torch.cuda.is_available():
            model.cuda()
        
        self.optim = Adam(model.parameters(), 
                          lr=self.config['lr'])
        self.optim_schedule = ScheduledOptim(optimizer=self.optim, 
                                             d_model=400,  # a num of model hidden size average
                                             n_warmup_steps=self.config['warmup_step'])    

        best_res = [0, 0, 0]
        best_state_dict = None
        early_stop_cnt = 0
        step = 0
        for epoch in range(1, self.config["epochs"]+1):
            for batch in train_iter:
                loss, metrics = self.foward_calculator(batch, model, cuda=self.config['cuda'])

                self.optim_schedule.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(filter(lambda p: p.requires_grad, model.parameters()), max_norm=self.config['clip'])
                self.optim_schedule.step_and_update_lr()

                if step % self.config['print_every'] == 0:
                    print(f"--epoch {epoch}, step {step}, loss {loss}")
                    print(f"  {metrics}")

                if val_iter and step % self.config['eval_every'] == 0:
                    avg_loss, avg_uas, avg_las = self.evaluate(model, val_iter)
                    res = [avg_loss, avg_uas, avg_las]
                    if avg_uas > best_res[1]:  # uas
                        best_res = res
                        best_state_dict = model.state_dict()
                        early_stop_cnt = 0
                    else:
                        early_stop_cnt += 1
                    print("--Best Evaluation: ")
                    print("-loss: {}  UAS: {}  LAS: {} \n".format(*best_res))
                    # back to train mode
                    model.train()
                
                if early_stop_cnt >= self.config['early_stop']:
                    print("--early stopping, training finished.")
                    return best_res, best_state_dict

                step += 1
        print("--training finished.")
        return best_res, best_state_dict

    # eval func
    def evaluate(self, model: nn.Module, eval_iter: DataLoader, save_file: str = "", save_title: str = ""):
        model.eval()

        avg_loss, avg_uas, avg_las, step = 0.0, 0.0, 0.0, 0
        for step, batch in enumerate(eval_iter):
            chars, words, rand_chars, labels, offsets, domain_labels, mlm_masks, pad_masks, seq_lens, POS_tags, heads, rel_tags = batch

            if self.config["cuda"] and torch.cuda.is_available():  # type: ignore
                chars, words, rand_chars, labels, offsets, \
                domain_labels, mlm_masks, pad_masks,\
                POS_tags, heads, rel_tags = to_cuda(data=(chars, words, rand_chars, labels, offsets, \
                                                            domain_labels, mlm_masks, pad_masks, \
                                                            POS_tags, heads, rel_tags))

            with torch.no_grad():
                # arc_logits, rel_logits = model(chars, words, domain_labels, POS_tags, heads, offsets, seq_lens)
                # arc_logits, rel_logits = model(chars, words, POS_tags, heads, offsets, seq_lens)
                arc_logits, rel_logits = model(chars, words, POS_tags, heads, seq_lens)
                
            loss = self.loss_fn_dct["parse_loss"](arc_logits, rel_logits, heads, rel_tags, pad_masks)
            avg_loss += loss * len(words)   # times the batch size of data

            metrics = self.metrics_fn_dct["parse_metrics"](arc_logits, rel_logits, heads, rel_tags, pad_masks)

            avg_uas += metrics['UAS'] * len(words)
            avg_las += metrics['LAS'] * len(words)

        # size = eval_iter.data_size 
        avg_loss /= len(eval_iter.dataset)  # type: ignore
        avg_uas /= len(eval_iter.dataset)  # type: ignore
        avg_las /= len(eval_iter.dataset)  # type: ignore
        print("--Evaluation:")
        print("-loss: {}  UAS: {}  LAS: {} \n".format(avg_loss, avg_uas, avg_las))

        if save_file != "":
            results = [save_title, avg_loss.item(), avg_uas, avg_las]  # type: ignore
            results = [str(x) for x in results]
            with open(save_file, "a+") as f:
                f.write(",".join(results) + "\n")  # type: ignore

        return avg_loss.item(), avg_uas, avg_las  # type: ignore