
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaModel

from adapter import AdapterBertModel

# reference: https://huggingface.co/transformers/model_doc/roberta.html

class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""
    def __init__(self, config, hidden_size: int = 768):  # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(config["hidden_size"], hidden_size)  
        self.dropout = nn.Dropout(config["dropout"])
        self.out_proj = nn.Linear(hidden_size, config["num_labels"])

    def forward(self, features, **kwargs):
        x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class BertFineTune(nn.Module):
    def __init__(self, config):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        for param in self.roberta.base_model.parameters(): # type: ignore
            param.requires_grad = True
        # self.roberta = AdapterBertModel()
        self.classifier = Classifier(config)
    
    def forward(self, input_ids, attention_mask):
        # bert
        roberta_outputs = self.roberta(input_ids, attention_mask)  # type: ignore
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.classifier(sequence_output)

        return logits


class BertPromptTune(nn.Module):
    def __init__(self, config, bert_config, mask_token, positive_tokens, negative_tokens):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        for param in self.roberta.base_model.parameters(): # type: ignore
            param.requires_grad = True
        self.masklm = RobertaLMHead(bert_config)

        self.vocab_size = bert_config.vocab_size
        self.mask_token = mask_token

        self.positive_tokens = positive_tokens
        self.negative_tokens = negative_tokens

        # assume weights follow a uniform distribution
        self.positive_weights = nn.Parameter(torch.rand(len(positive_tokens)), requires_grad = True)
        self.negative_weights = nn.Parameter(torch.rand(len(negative_tokens)), requires_grad = True)

        self.position_weights = nn.Parameter(torch.rand(2), requires_grad = True)

        self.learnable_tokens = - 1
        self.num_learnable_token = 2
        self.learnable_token_emb = nn.Embedding(num_embeddings=self.num_learnable_token, embedding_dim=300)
        # self.learnable_token_lstm = nn.LSTM(input_size=300, hidden_size=768//2, batch_first=True, bidirectional=True, dropout=0.33)
        self.learnable_token_ffn = nn.Linear(in_features=300, out_features=768)
        # self.learnable_token_emb = None 
    
    def forward(self, input_ids, attention_mask):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)

        if self.learnable_token_emb is not None:
            add_ids = (input_ids == self.learnable_tokens).nonzero(as_tuple=True)
            input_ids[add_ids] = self.mask_token
            
            # add learnable token embeddings
            replace_embeds  = self.learnable_token_emb(torch.arange(self.num_learnable_token).cuda())  # num_learnable_token, embed_dim
            replace_embeds = replace_embeds.unsqueeze(0).repeat(batch_size, 1, 1)  # batch_size, num_learnable_token, embed_dim
            replace_embeds  = self.learnable_token_ffn(replace_embeds)  # batch_size, num_learnable_token, hidden_size
            replace_embeds = replace_embeds.reshape(batch_size*self.num_learnable_token, -1)  # batch_size * num_learnable_token, hidden_size
            
            # replace the corresponding token embeddings
            input_emb = self.roberta.embeddings.word_embeddings(input_ids)  # type: ignore
            input_emb[add_ids] = replace_embeds
            input_emb = input_emb.view(batch_size, seq_len, -1)  # batch_size, seq_len, embed_dim
            # roberta
            roberta_outputs = self.roberta(inputs_embeds=input_emb, attention_mask=attention_mask)  # type: ignore
        else:
            # roberta
            roberta_outputs = self.roberta(input_ids, attention_mask)  # type: ignore
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.masklm(sequence_output)
        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1) 
        mask_logits = mask_logits.view(batch_size, -1, vocab_size) # batch_size, mask_num, vocab_size
        _, mask_num, _ = mask_logits.size()

        # batch_size, mask_num, vocab_size
        mask_logits = (mask_logits.transpose(1, 2) * self.position_weights[:mask_num]).transpose(1, 2)

        mask_logits = mask_logits.sum(dim=1).squeeze(1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        positive_logits = mask_logits[:, self.positive_tokens] * positive_weight  # batch_size, len(positive_tokens)
        negative_logits = mask_logits[:, self.negative_tokens] * negative_weight # batch_size, len(negative_tokens)

        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([positive_logits, negative_logits], dim=1)

        return cls_logits


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""
    def __init__(self, config, hidden_size: int = 768):   # TODO: configuration
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(hidden_size, config.vocab_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = nn.functional.gelu(x)  # type: ignore
        x = self.layer_norm(x)

        # project back to size of vocabulary with bias
        x = self.decoder(x)

        return x

    def _tie_weights(self):
        # To tie those two weights if they get disconnected (on TPU or when the bias is resized)
        self.bias = self.decoder.bias


class TextCNN(nn.Module):
    def __init__(self, config, vocab_size):
        super(TextCNN, self).__init__()
        
        embed_dim = config["embed_dim"]
        kernel_num = config["kernel_num"]
        kernel_sizes = config["kernel_sizes"]
        dropout = config["dropout"]
        class_num = config["class_num"]
        
        self.emb = nn.Embedding(vocab_size, embed_dim)
        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, embed_dim)) for K in kernel_sizes])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(kernel_num*len(kernel_sizes), class_num)

    def forward(self, x):
        x = self.emb(x)  # [batch_size, sentence_len, embed_dim]
        
        x = x.unsqueeze(1)  # [batch_size, 1, sentence_len, embed_dim]
        
        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs] 
        x = [F.max_pool1d(i, i.size(2)).squeeze(2) for i in x]  # [(N, Co), ...]*len(Ks)
        x = torch.cat(x, 1)
        
        x = self.dropout(x)
        logit = self.fc(x)

        return logit