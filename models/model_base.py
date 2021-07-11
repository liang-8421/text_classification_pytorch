# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 16:56
# @Author  : miliang
# @FileName: model_base.py
# @Software: PyCharm

import torch
from torch import nn
from transformers import BertModel


class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.pre_model = BertModel.from_pretrained(config.pretrain_model_path)
        self.dropout = torch.nn.Dropout(config.dropout_rate)
        self.use_pooling = config.use_pooling
        self.label_num = config.label_num
        self.fc = torch.nn.Linear(config.bert_hidden, self.label_num)
        self.cross_entropy = nn.functional.cross_entropy

    def forward(self, input_ids, input_mask, segment_ids, labels=None):
        sequence_out, ori_pooled_output, encoded_layers = self.pre_model(input_ids=input_ids,
                                                                         attention_mask=input_mask,
                                                                         token_type_ids=segment_ids,
                                                                         )
        sequence_out = self.dropout(sequence_out)

        if self.use_pooling == "max":
            pass
        elif self.use_pooling == "avg":
            pass
        else:
            sequence_out = sequence_out[:, 0, :]

        logits = self.fc(sequence_out)

        if labels is not None:
            # label_one_hot = torch.nn.functional.one_hot(labels, self.label_num)
            loss = self.cross_entropy(logits, labels)
            return loss
        else:
            prob = torch.softmax(logits, dim=-1)
            pred = torch.argmax(logits, dim=-1)
            return prob, pred
