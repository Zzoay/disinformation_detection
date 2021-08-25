
import math
from typing import *

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import init
from transformers import RobertaModel


def set_requires_grad(module: nn.Module, status: bool = False):
    for param in module.parameters():
        param.requires_grad = status


class AdapterBertModel(nn.Module):
    def __init__(self,
                 adapter_size: int = 192,
                 external_param: Union[bool, List[bool]] = False,
                 trainsets_len: int = -1,
                 adapter_layer: int = 9,
                 begin_layer: int = 12,
                 **kwargs):
        super().__init__()
        self.last_pos = None
        self.adapter_layer = adapter_layer
        self.begin_layer = begin_layer
        self.trainsets_len = trainsets_len
        print(f'adapter_layer begin at: {adapter_layer}')

        self.bert = RobertaModel.from_pretrained('roberta-base')

        set_requires_grad(self.bert, False)

        if isinstance(external_param, bool):
            param_place = [external_param for _ in range(
                self.bert.config.num_hidden_layers)]
        elif isinstance(external_param, list):
            param_place = [False for _ in range(
                self.bert.config.num_hidden_layers)]
            for i, e in enumerate(external_param, 1):
                param_place[-i] = e

        self.adapters = nn.ModuleList()
        self.adapter_lambda = None

        for i, e in enumerate(param_place):
            if i < self.adapter_layer - 1:
                self.adapters.extend([None])
            elif i < self.begin_layer - 1:
                self.adapters.extend([nn.ModuleList([
                    Adapter(self.bert.config.hidden_size, adapter_size),
                    Adapter(self.bert.config.hidden_size, adapter_size)
                    ])
                ])
            else:
                self.adapters.extend([nn.ModuleList([
                    Adapter(self.bert.config.hidden_size, adapter_size),
                    Adapter(self.bert.config.hidden_size, adapter_size)
                    ])
                ])

        for i, layer in enumerate(self.bert.encoder.layer):
            if i < self.adapter_layer - 1:
                continue
            layer.output = AdapterBertOutput(
                layer.output, self.adapters[i][0].forward)
            set_requires_grad(layer.output.base.LayerNorm, True)
            layer.attention.output = AdapterBertOutput(
                layer.attention.output, self.adapters[i][1].forward)
            set_requires_grad(layer.attention.output.base.LayerNorm, True)

        self.output_dim = self.bert.config.hidden_size

    def forward(self,  
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                **kwargs) -> torch.Tensor:
        bert_output = self.bert(input_ids, attention_mask)
        return bert_output


class AdapterBertOutput(nn.Module):
    """
    Replace BertOutput and BertSelfOutput
    """
    def __init__(self, base, adapter_forward):
        super().__init__()
        self.base = base
        self.adapter_forward = adapter_forward

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.base.dense(hidden_states)
        hidden_states = self.base.dropout(hidden_states)
        hidden_states = self.adapter_forward(hidden_states)
        hidden_states = self.base.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class Adapter(nn.Module):
    def __init__(self, in_features, bottleneck_size):
        super().__init__()
        self.in_features = in_features
        self.bottleneck_size = bottleneck_size

        self.down_project = nn.Linear(in_features, bottleneck_size)

        self.activation = nn.GELU()
        self.up_project = nn.Linear(bottleneck_size, in_features)
    
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.down_project.weight, a=math.sqrt(5))
        init.kaiming_uniform_(self.up_project.weight, a=math.sqrt(5))

    def forward(self, hidden_states: torch.Tensor):
        x = self.down_project(hidden_states)
        x = self.activation(x)
        x = self.up_project(x)
        return x + hidden_states