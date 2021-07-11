# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 17:10
# @Author  : miliang
# @FileName: train_utils.py
# @Software: PyCharm

from models.model_base import Model
import os
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import torch
import copy
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score


def train(config, train_iter, dev_iter):
    model = Model(config).to(config.device)
    model.train()

    bert_param_optimizer = list(model.pre_model.named_parameters())
    # linear_param_optimizer = list(model.hidden2label.named_parameters())
    # crf_param_optimizer = list(model.crf_layer.named_parameters())

    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        # pre-train model
        {'params': [p for n, p in bert_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,
         "lr": config.bert_learning_rate},
        {'params': [p for n, p in bert_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,
         "lr": config.bert_learning_rate},

        # linear layer
        # {'params': [p for n, p in linear_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01,"lr": config.bert_learning_rate},
        # {'params': [p for n, p in linear_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0,"lr": config.bert_learning_rate},

        # crf,单独设置学习率
        # {'params': [p for n, p in crf_param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01, "lr": config.crf_learning_rate},
        # {'params': [p for n, p in crf_param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0, "lr": config.crf_learning_rate}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      betas=(0.9, 0.98),  # according to RoBERTa paperbetas=(0.9, 0.98),  # according to RoBERTa paper
                      lr=config.bert_learning_rate,
                      eps=1e-8
                      )
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(
        len(train_iter) * config.train_epoch * config.warmup_prop),
                                                num_training_steps=len(train_iter) * config.train_epoch)

    cum_step, epoch_step, best_f1 = 0, 0, 0
    for i in range(config.train_epoch):
        model.train()
        for input_ids, input_mask, segment_ids, labels, tokens_cpu in tqdm(train_iter, position=0, ncols=80,
                                                                           desc='训练中'):
            loss = model.forward(input_ids, input_mask, segment_ids, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()  # Update learning rate schedule
            model.zero_grad()
            cum_step += 1

        f1, p, r = set_test(config, model, dev_iter)

        # lr_scheduler学习率递减 step
        # print('dev set : step_{},precision_{}, recall_{}, f1_{}, loss_{}'.format(cum_step, p, r, f1, loss))
        # 保存模型
        model_to_save = model.module if hasattr(model, 'module') else model  # Only save the model it-self
        output_model_file = os.path.join(
            os.path.join(config.model_save_path, 'model_{:.4f}_{:.4f}_{:.4f}_{}.bin'.format(p, r, f1, str(cum_step))))
        torch.save(model_to_save, output_model_file)


def set_test(config, model, dev_iter):
    model.eval()
    true_doc_label_list, pred_doc_label_list = [], []
    for input_ids, input_mask, segment_ids, labels, tokens_cpu in tqdm(dev_iter, position=0, ncols=80, desc='验证中'):
        prob, pred = model.forward(input_ids, input_mask, segment_ids)

        labels = labels.cpu().numpy()
        pred = pred.cpu().numpy()
        true_doc_label_list.extend(labels)
        pred_doc_label_list.extend(pred)

    report = classification_report(y_true=true_doc_label_list, y_pred=pred_doc_label_list,
                                   target_names=config.class_list)
    f1 = f1_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    p = precision_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    r = recall_score(y_true=true_doc_label_list, y_pred=pred_doc_label_list, average='macro')
    config.logger.info(report)
    config.logger.info('precision: {}, recall {}, f1 {}'.format(p, r, f1))

    return f1, p, r
