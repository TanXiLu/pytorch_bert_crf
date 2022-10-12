# coding=utf-8
import torch
from transformers import BertModel, AutoModel
import torch.nn as nn
from torchcrf import CRF


class BertNerModel(nn.Module):
    def __init__(self, config):
        super(BertNerModel, self).__init__()
        self.config = config
        bert_dir = config['bert_dir']
        model_name = config['model_name']
        dropout_prob = config['dropout_prob']

        if 'electra' in model_name or 'albert' in model_name:
            self.encoder = AutoModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)
        else:
            self.encoder = BertModel.from_pretrained(bert_dir,
                                                     output_hidden_states=True,
                                                     hidden_dropout_prob=dropout_prob)
        self.encoder_config = self.encoder.config
        out_dims = self.encoder_config.hidden_size
        #
        self.num_layers = config['num_layers']
        self.lstm_hidden = config['lstm_hidden']
        self.num_tags = config['num_tags']
        dropout = config['dropout']
        self.criterion = nn.CrossEntropyLoss()

        if config['use_lstm'] is True:
            self.lstm = nn.LSTM(out_dims, self.lstm_hidden, self.num_layers,
                                bidirectional=True, batch_first=True, dropout=dropout)
            self.linear = nn.Linear(self.lstm_hidden*2, self.num_tags)
        else:
            self.mid_linear = nn.Sequential(
                nn.Linear(out_dims, 256),
                nn.ReLU(),
                nn.Dropout(dropout)
            )
            self.classifier = nn.Linear(256, self.num_tags)

        if config['use_crf'] is True:
            self.crf = CRF(self.num_tags, batch_first=True)

    def init_hidden(self, batch_size):
        h0 = torch.randn(self.num_layers*2, batch_size, self.lstm_hidden, requires_grad=True)
        c0 = torch.randn(self.num_layers*2, batch_size, self.lstm_hidden, requires_grad=True)
        return h0, c0

    def forward(self, data):
        encoder_outputs = self.encoder(
            input_ids=data['token_ids'],
            attention_mask=data['attention_masks'],
            token_type_ids=data['token_type_ids']
        )
        # seq_out: [batch_size, max_len, embedding_dims=768]
        seq_out = encoder_outputs[0]
        batch_size = seq_out.size(0)

        if self.config['use_lstm'] is True:
            hidden = self.init_hidden(batch_size)
            seq_out, (hn, _) = self.lstm(seq_out, hidden)
            seq_out = seq_out.contiguous().view(-1, self.lstm_hidden*2)
            seq_out = self.linear(seq_out)
            # [batch_size, max_len, num_tags]
            seq_out = seq_out.contiguous().view(batch_size, self.config["max_seq_len"], -1)
        else:
            # [batch_size, max_len, 256]
            seq_out = self.mid_linear(seq_out)
            # [batch_size, max_len, num_tags]
            seq_out = self.classifier(seq_out)

        if self.config["use_crf"] is True:
            logits = self.crf.decode(seq_out, mask=data['attention_masks'])
            if data['labels'] is None:
                return logits
            loss = -self.crf(seq_out, data["labels"], mask=data['attention_masks'], reduction='mean')
            return loss, logits
        else:
            logits = seq_out
            if data["labels"] is None:
                return logits
            active_loss = data["attention_masks"].view(-1) == 1
            active_logits = logits.view(-1, logits.size()[2])[active_loss]
            active_labels = data["labels"].view(-1)[active_loss]
            loss = self.criterion(active_logits, active_labels)
            return loss, logits








