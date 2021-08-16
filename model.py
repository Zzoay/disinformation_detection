
import torch
from torch import nn
from transformers import RobertaModel


class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], config["hidden_size"])
        self.dropout = nn.Dropout(config["dropout"])
        self.out_proj = nn.Linear(config["hidden_size"], config["num_label"])

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertFinetune(nn.Module):
    def __init__(self, config):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.classifier = Classifier(config)
    
    def forward(self, input_ids, attention_mask):
        # bert
        roberta_outputs = self.roberta(input_ids, attention_mask)
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.classifier(sequence_output)

        return logits