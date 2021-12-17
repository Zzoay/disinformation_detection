
import torch
from torch import nn
from torch.nn import functional as F
from transformers import RobertaModel, BertModel, RobertaForMaskedLM


# reference: https://huggingface.co/transformers/model_doc/roberta.html
class Classifier(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config, hidden_size: int = 400):  # TODO: configuration
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
        # self.roberta = BertModel.from_pretrained('bert-base-uncased')
        for param in self.roberta.base_model.parameters():  # type: ignore
            param.requires_grad = True
        # self.roberta = AdapterBertModel()
        self.classifier = Classifier(config)

    def forward(self, input_ids, attention_mask, entity_ids):
        # bert
        roberta_outputs = self.roberta(
            input_ids, attention_mask)  # type: ignore
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.classifier(sequence_output)

        return logits


class BertPromptTune(nn.Module):
    def __init__(self,
                 config,
                 bert_config,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_learnable_emb=True,
                 with_answer_weights=True,
                 with_position_weights=False,
                 num_learnable_token=2,
                 zero_shot=False,
                 fine_tune_all=True):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # self.roberta = BertModel.from_pretrained('bert-base-uncased')
        if not fine_tune_all:  # freeze the pretrained encoder
            for param in self.roberta.base_model.parameters():  # type: ignore
                param.requires_grad = False
            self.roberta.embeddings.word_embeddings.requires_grad = True

        self.masklm = RobertaLMHead(bert_config)

        self.vocab_size = bert_config.vocab_size
        self.mask_token_id = mask_token_id

        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids

        # when in zero shot condition, simply sum over all ids
        if zero_shot:
            with_learnable_emb = False
            with_answer_weights = False

        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        if with_position_weights:
            self.position_weights = nn.Parameter(
                torch.rand(2), requires_grad=True)
        else:
            self.position_weights = nn.Parameter(
                torch.ones(2), requires_grad=False)

        self.learnable_tokens = - 1
        self.num_learnable_token = num_learnable_token
        if with_learnable_emb:
            self.learnable_token_emb = nn.Embedding(
                num_embeddings=self.num_learnable_token, embedding_dim=300)
        else:
            self.learnable_token_emb = None
        # self.learnable_token_lstm = nn.LSTM(input_size=300, hidden_size=768//2, batch_first=True, bidirectional=True, dropout=0.33)
        self.learnable_token_ffn = nn.Linear(in_features=300, out_features=768)
        # self.learnable_token_ffn = nn.Linear(in_features=300, out_features=1024)

    def forward(self, input_ids, attention_mask, entity_ids):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)

        if self.learnable_token_emb is not None:
            add_ids = (input_ids == self.learnable_tokens).nonzero(
                as_tuple=True)
            input_ids[add_ids] = self.mask_token_id

            # add learnable token embeddings
            replace_embeds = self.learnable_token_emb(torch.arange(
                self.num_learnable_token).cuda())  # num_learnable_token, embed_dim
            replace_embeds = replace_embeds.unsqueeze(0).repeat(
                batch_size, 1, 1)  # batch_size, num_learnable_token, embed_dim
            # batch_size, num_learnable_token, hidden_size
            replace_embeds = self.learnable_token_ffn(replace_embeds)
            # batch_size * num_learnable_token, hidden_size
            replace_embeds = replace_embeds.reshape(
                batch_size*self.num_learnable_token, -1)

            # replace the corresponding token embeddings
            input_emb = self.roberta.embeddings.word_embeddings(
                input_ids)  # type: ignore
            input_emb[add_ids] = replace_embeds
            # batch_size, seq_len, embed_dim
            input_emb = input_emb.view(batch_size, seq_len, -1)
            # roberta
            roberta_outputs = self.roberta(
                inputs_embeds=input_emb, attention_mask=attention_mask)  # type: ignore
        else:
            # roberta
            roberta_outputs = self.roberta(
                input_ids, attention_mask)  # type: ignore
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.masklm(sequence_output)
        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()

        # batch_size, mask_num, vocab_size
        mask_logits = (mask_logits.transpose(1, 2) *
                       self.position_weights[:mask_num]).transpose(1, 2)

        mask_logits = mask_logits.sum(dim=1).squeeze(
            1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                                      self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                                      self.negative_token_ids] * negative_weight

        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([positive_logits, negative_logits], dim=1)

        return cls_logits


class BertPTWithEntity(nn.Module):
    def __init__(self,
                 config,
                 bert_config,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_learnable_emb=True,
                 with_answer_weights=True,
                 with_position_weights=False,
                 num_learnable_token=2,
                 zero_shot=False,
                 fine_tune_all=True):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # self.roberta = BertModel.from_pretrained('bert-base-uncased')

        self.masklm = RobertaLMHead(bert_config)

        if not fine_tune_all:  # freeze the pretrained encoder
            for param in self.roberta.base_model.parameters():  # type: ignore
                param.requires_grad = False
            self.roberta.embeddings.word_embeddings.requires_grad = True

        self.vocab_size = bert_config.vocab_size
        self.mask_token_id = mask_token_id

        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids

        # when in zero shot condition, simply sum over all ids
        self.zero_shot = zero_shot
        if zero_shot:
            # with_learnable_emb = False
            with_answer_weights = False

        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        if with_position_weights:
            self.position_weights = nn.Parameter(
                torch.rand(2), requires_grad=True)
        else:
            self.position_weights = nn.Parameter(
                torch.ones(2), requires_grad=False)

        self.learnable_tokens = - 1
        self.num_learnable_token = num_learnable_token
        # self.num_learnable_token = 4
        if with_learnable_emb:
            self.learnable_token_emb = nn.Embedding(
                num_embeddings=self.num_learnable_token, embedding_dim=768)
        else:
            self.learnable_token_emb = None
        # self.learnable_token_ffn = nn.Linear(in_features=300, out_features=768)

        # self.entity_project = nn.Linear(in_features=768, out_features=300)
        self.entity_conv1 = nn.Conv2d(1, 768, (3, 768))
        # self.entity_conv2 = nn.Conv2d(1, 300, (5, 300))
        # self.entity_conv3 = nn.Conv2d(1, 768, (3, 300))

        self.entity_gru = nn.GRU(
            input_size=768, hidden_size=768//2, bidirectional=True, batch_first=True)

        self.norm = nn.LayerNorm(normalized_shape=768)
        self.dropout = nn.Dropout(0.33)

    def encode_entities(self, input_ids, entity_ids):
        batch_size, seq_len = input_ids.size()

        add_ids = (input_ids == self.learnable_tokens).nonzero(
            as_tuple=True)
        input_ids[add_ids] = self.mask_token_id

        # # add learnable token embeddings
        replace_embeds = self.learnable_token_emb(torch.arange(
            self.num_learnable_token).cuda())  # num_learnable_token, embed_dim

        replace_embeds = replace_embeds.unsqueeze(0).repeat(
            batch_size, 1, 1)  # batch_size, num_learnable_token, embed_dim

        # learn learnable token with entity
        entity_lens = (entity_ids != 1).sum(1) - 2  # ignore cls, eos, pad
        entity_emb = self.roberta.embeddings.word_embeddings(
            entity_ids)  # type: ignore

        entity_reprs = self.entity_conv1(
            entity_emb.unsqueeze(1)).squeeze(-1).transpose(1, 2)

        # _, e_len, _ = entity_reprs.size()  # e_len is the conved entity seq's max length
        # len_scale = (entity_lens * (e_len / entity_lens.max())).long()

        # entity_reprs[:, 0, :] += replace_embeds[:, 0]
        # entity_reprs[torch.arange(batch_size),
        #             len_scale-1, :] += replace_embeds[:, 1]
        entity_reprs = self.dropout(self.norm(entity_reprs))
        entity_reprs, _ = self.entity_gru(entity_reprs)
        entity_reprs = entity_reprs.transpose(1, 2)

        _, _, e_len = entity_reprs.size()  # e_len is the encoded entity seq's max length
        len_scale = (entity_lens * (e_len / entity_lens.max())).long()

        eo1, eo2 = replace_embeds[:, 0], replace_embeds[:, 1]
        e1, e2 = entity_reprs[:, :, 0], entity_reprs[torch.arange(
            batch_size), :, len_scale-1]

        e1, e2 = e1 + eo1, e2 + eo2

        replace_embeds = torch.cat(
            [e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
        replace_embeds = self.dropout(self.norm(replace_embeds))

        # replace the corresponding token embeddings
        input_emb = self.roberta.embeddings.word_embeddings(
            input_ids)  # type: ignore
        input_emb[add_ids] = replace_embeds.view(-1, 768)
        # batch_size, seq_len, embed_dim
        input_emb = input_emb.view(batch_size, seq_len, -1)
                    
        return input_emb
    
    def encode_entiies_zero(self, input_ids, entity_ids):
        batch_size, seq_len = input_ids.size()
        add_ids = (input_ids == self.learnable_tokens).nonzero(
            as_tuple=True)
        input_ids[add_ids] = self.mask_token_id

        replace_embeds = self.roberta.embeddings.word_embeddings(
            entity_ids).sum(1).repeat(2, 1)

        # replace the corresponding token embeddings
        input_emb = self.roberta.embeddings.word_embeddings(
            input_ids)  # type: ignore
        input_emb[add_ids] = replace_embeds.view(-1, 768)
        # batch_size, seq_len, embed_dim
        input_emb = input_emb.view(batch_size, seq_len, -1)   
        return input_emb

    def forward(self, input_ids, attention_mask, entity_ids):
        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)

        if self.learnable_token_emb is None:
            # roberta
            roberta_outputs = self.roberta(
                input_ids, attention_mask)  # type: ignore
        elif entity_ids.size()[1] == 2:  # entity list is empty
            add_ids = (input_ids == self.learnable_tokens).nonzero(
                as_tuple=True)
            input_ids[add_ids] = self.mask_token_id

            # add learnable token embeddings
            replace_embeds = self.learnable_token_emb(torch.arange(
                self.num_learnable_token).cuda())  # num_learnable_token, embed_dim
            # replace_embeds = self.dropout(replace_embeds)
            replace_embeds = replace_embeds.unsqueeze(0).repeat(
                batch_size, 1, 1)  # batch_size, num_learnable_token, embed_dim

            input_emb = self.roberta.embeddings.word_embeddings(
                input_ids)  # type: ignore
            input_emb[add_ids] = replace_embeds.view(-1, 768)
            # batch_size, seq_len, embed_dim
            input_emb = input_emb.view(batch_size, seq_len, -1)
            # roberta
            roberta_outputs = self.roberta(
                inputs_embeds=input_emb, attention_mask=attention_mask)  # type: ignore
        else:  # if self.learnable_token_emb is not None
            input_emb = self.encode_entities(input_ids, entity_ids)
            # input_emb = self.encode_entiies_zero(input_ids, entity_ids)
            # roberta
            roberta_outputs = self.roberta(
                inputs_embeds=input_emb, attention_mask=attention_mask)  # type: ignore
        sequence_output = roberta_outputs.last_hidden_state

        logits = self.masklm(sequence_output)
        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()

        # batch_size, mask_num, vocab_size
        mask_logits = (mask_logits.transpose(1, 2) *
                       self.position_weights[:mask_num]).transpose(1, 2)

        mask_logits = mask_logits.sum(dim=1).squeeze(
            1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                                      self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                                      self.negative_token_ids] * negative_weight

        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([positive_logits, negative_logits], dim=1)

        return cls_logits


class PTWELightWeight(nn.Module):   # only tune the prompt
    def __init__(self,
                 config,
                 bert_config,
                 mask_token_id,
                 positive_token_ids,
                 negative_token_ids,
                 with_learnable_emb=True,
                 with_answer_weights=True,
                 with_position_weights=False,
                 num_learnable_token=2,
                 zero_shot=False):
        super().__init__()
        # encoder
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        # self.roberta = BertModel.from_pretrained('bert-base-uncased')
        # for param in self.roberta.base_model.parameters(): # type: ignore
        #     param.requires_grad = False
        # self.roberta.embeddings.word_embeddings.requires_grad = True

        self.masklm = RobertaLMHead(bert_config)

        self.vocab_size = bert_config.vocab_size
        self.mask_token_id = mask_token_id

        self.positive_token_ids = positive_token_ids
        self.negative_token_ids = negative_token_ids

        # when in zero shot condition, simply sum over all ids
        if zero_shot:
            with_learnable_emb = False
            with_answer_weights = False

        if with_answer_weights:
            # assume weights follow a uniform distribution
            self.positive_weights = nn.Parameter(torch.rand(
                len(positive_token_ids)), requires_grad=True)
            self.negative_weights = nn.Parameter(torch.rand(
                len(negative_token_ids)), requires_grad=True)
        else:
            self.positive_weights = nn.Parameter(torch.ones(
                len(positive_token_ids)), requires_grad=False)
            self.negative_weights = nn.Parameter(torch.ones(
                len(negative_token_ids)), requires_grad=False)

        if with_position_weights:
            self.position_weights = nn.Parameter(
                torch.rand(2), requires_grad=True)
        else:
            self.position_weights = nn.Parameter(
                torch.ones(2), requires_grad=False)

        self.learnable_tokens = - 1
        self.num_learnable_token = num_learnable_token
        # self.num_learnable_token = 4
        if with_learnable_emb:
            self.learnable_token_emb = nn.Embedding(
                num_embeddings=self.num_learnable_token, embedding_dim=768)
        else:
            self.learnable_token_emb = None
        # self.learnable_token_ffn = nn.Linear(in_features=300, out_features=768)

        # self.entity_project = nn.Linear(in_features=768, out_features=300)
        self.entity_conv1 = nn.Conv2d(1, 768, (3, 768))
        # self.entity_conv2 = nn.Conv2d(1, 300, (5, 300))
        # self.entity_conv3 = nn.Conv2d(1, 768, (3, 300))

        self.entity_gru = nn.GRU(
            input_size=768, hidden_size=768//2, bidirectional=True, batch_first=True)

        self.norm = nn.LayerNorm(normalized_shape=768)
        self.dropout = nn.Dropout(0.33)

    def forward(self, input_ids, attention_mask, entity_ids):
        prefix_ids, input_ids = input_ids

        batch_size, seq_len = input_ids.size()
        mask_ids = (input_ids == self.mask_token_id).nonzero(as_tuple=True)
        # mask_ids = mask_ids.expand(batch_size, seq_len, self.vocab_size)

        if self.learnable_token_emb is not None:
            add_ids = (prefix_ids == self.learnable_tokens).nonzero(
                as_tuple=True)
            prefix_ids[add_ids] = self.mask_token_id

            # # add learnable token embeddings
            replace_embeds = self.learnable_token_emb(torch.arange(
                self.num_learnable_token).cuda())  # num_learnable_token, embed_dim
            # # replace_embeds = self.dropout(replace_embeds)
            replace_embeds = replace_embeds.unsqueeze(0).repeat(
                batch_size, 1, 1)  # batch_size, num_learnable_token, embed_dim

            # learn learnable token with entity
            entity_lens = (entity_ids != 1).sum(1) - 2  # ignore cls, eos, pad
            entity_emb = self.roberta.embeddings.word_embeddings(
                entity_ids)  # type: ignore

            # entity_reprs = self.dropout(self.entity_conv1(entity_emb.unsqueeze(1)).squeeze(-1).transpose(1, 2))
            entity_reprs = self.entity_conv1(
                entity_emb.unsqueeze(1)).squeeze(-1).transpose(1, 2)

            _, e_len, _ = entity_reprs.size()  # e_len is the conved entity seq's max length
            len_scale = (entity_lens * (e_len / entity_lens.max())).long()

            entity_reprs[:, 0, :] += replace_embeds[:, 0]
            entity_reprs[torch.arange(batch_size),
                         len_scale-1, :] += replace_embeds[:, 1]
            entity_reprs = self.dropout(self.norm(entity_reprs))

            # entity_reprs = self.dropout(self.entity_conv2(entity_reprs).squeeze(-1).transpose(1, 2))
            entity_reprs, _ = self.entity_gru(entity_reprs)
            entity_reprs = entity_reprs.transpose(1, 2)

            _, _, e_len = entity_reprs.size()  # e_len is the encoded entity seq's max length
            len_scale = (entity_lens * (e_len / entity_lens.max())).long()

            # len_middle = (len_scale // 2).long()
            eo1, eo2 = replace_embeds[:, 0], replace_embeds[:, 1]
            e1, e2 = entity_reprs[:, :, 0], entity_reprs[torch.arange(
                batch_size), :, len_scale-1]

            e1, e2 = e1 + eo1, e2 + eo2

            replace_embeds = torch.cat(
                [e1.unsqueeze(1), e2.unsqueeze(1)], dim=1)
            replace_embeds = self.dropout(self.norm(replace_embeds))

            prefix_outputs = self.roberta(
                prefix_ids, attention_mask=torch.ones_like(prefix_ids)).last_hidden_state

            with torch.no_grad():
                x_outputs = self.roberta(
                    input_ids, attention_mask=attention_mask).last_hidden_state
        else:
            # roberta
            roberta_outputs = self.roberta(
                input_ids, attention_mask)  # type: ignore

        sequence_output = torch.cat([prefix_outputs, x_outputs], dim=1)

        logits = self.masklm(sequence_output)
        _, _, vocab_size = logits.size()

        mask_logits = logits[mask_ids]  # batch_size, vocab_size
        mask_logits = F.log_softmax(mask_logits, dim=1)
        # batch_size, mask_num, vocab_size
        mask_logits = mask_logits.view(batch_size, -1, vocab_size)
        _, mask_num, _ = mask_logits.size()

        # batch_size, mask_num, vocab_size
        mask_logits = (mask_logits.transpose(1, 2) *
                       self.position_weights[:mask_num]).transpose(1, 2)

        mask_logits = mask_logits.sum(dim=1).squeeze(
            1)  # batch_size, vocab_size
        # mask_logits = mask_logits.prod(dim=1).squeeze(1)  # batch_size, vocab_size

        positive_weight = F.softmax(self.positive_weights, dim=0)
        negative_weight = F.softmax(self.negative_weights, dim=0)

        # batch_size, len(positive_token_ids)
        positive_logits = mask_logits[:,
                                      self.positive_token_ids] * positive_weight
        # batch_size, len(negative_token_ids)
        negative_logits = mask_logits[:,
                                      self.negative_token_ids] * negative_weight

        positive_logits = positive_logits.sum(1).unsqueeze(1)  # batch_size, 1
        negative_logits = negative_logits.sum(1).unsqueeze(1)  # batch_size, 1

        cls_logits = torch.cat([positive_logits, negative_logits], dim=1)

        return cls_logits


class RobertaLMHead(nn.Module):
    """Roberta Head for masked language modeling."""

    def __init__(self, config, hidden_size: int = 200):   # TODO: configuration
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
