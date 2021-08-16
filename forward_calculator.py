
from typing import Any

import torch

from utils import to_cuda, seq_mask_by_lens


class FinetuneFoward():
    def __init__(self, loss_fn, metrics_fn) -> None:
        self.loss_fn = loss_fn
        self.metrics_fn = metrics_fn

    def compute_forward(self, batch, model, cuda:bool = False):
        input_ids, attention_mask, labels, seq_lens = batch

        if cuda and torch.cuda.is_available():  # type: ignore
            input_ids, attention_mask, labels = to_cuda(data=(input_ids, attention_mask, labels))

        logits = model(input_ids, attention_mask)
        prediction = logits.max(2)[1]

        loss = self.loss_fn(logits, labels)
        metrics = self.metrics_fn(prediction, labels)

        return loss, metrics
    
    def __call__(self, *args: Any, **kwds: Any) -> Any:
        return self.compute_forward(*args, **kwds)