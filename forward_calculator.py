
from typing import Any
from collections import Counter

import torch

from utils import to_cuda, seq_mask_by_lens


class FinetuneFoward():
    def __init__(self, loss_fn, metrics_fn) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def compute_forward(self, batch, model, cuda:bool = False, evaluate:bool = False):
        input_ids, attention_mask, labels, seq_lens = batch

        if cuda and torch.cuda.is_available():  # type: ignore
            input_ids, attention_mask, labels = to_cuda(data=(input_ids, attention_mask, labels))

        if evaluate:
            with torch.no_grad():
                logits = model(input_ids, attention_mask)
                return logits
        else:
            logits = model(input_ids, attention_mask)
        # prediction = logits.max(1)[1]

        cnt = Counter(labels.cpu().tolist())  # type: ignore
        weight = [1, 1]
        weight[0] = 1 - cnt[0] / sum(cnt.values())  # type: ignore
        weight[1] = 1 - cnt[1] / sum(cnt.values())  # type: ignore
        loss = self.loss_fn(input=logits, target=labels, weight=torch.tensor(weight).cuda())
        # loss = self.loss_fn(input=logits, target=labels)
        metrics = self.metrics_fn(logits, labels)

        return logits, loss, metrics
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_forward(*args, **kwds)


class TextCNNFoward():
    def __init__(self, loss_fn, metrics_fn) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def compute_forward(self, batch, model, cuda:bool = False, evaluate:bool = False):
        input_ids, attention_mask, labels, seq_lens = batch

        if cuda and torch.cuda.is_available():  # type: ignore
            input_ids, attention_mask, labels = to_cuda(data=(input_ids, attention_mask, labels))

        if evaluate:
            with torch.no_grad():
                logits = model(input_ids)
        else:
            logits = model(input_ids)
        # prediction = logits.max(1)[1]

        loss = self.loss_fn(logits, labels)
        metrics = self.metrics_fn(logits, labels)

        return loss, metrics
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_forward(*args, **kwds)