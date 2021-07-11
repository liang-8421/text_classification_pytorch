# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 20:44
# @Author  : miliang
# @FileName: predict_utils.py.py
# @Software: PyCharm

import torch
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score


def predict(config, test_iter):
    model = torch.load(config.read_model_file)
    print("read model from {}".format(config.read_model_file))
    model.to(config.device)
    model.eval()
    true_doc_label_list, pred_doc_label_list = [], []
    for input_ids, input_mask, segment_ids, labels, tokens_cpu in tqdm(test_iter, position=0, ncols=80, desc='验证中'):
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
    print(report)
    print('precision: {}, recall {}, f1 {}'.format(p, r, f1))
