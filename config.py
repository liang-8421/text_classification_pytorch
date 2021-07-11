# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 10:18
# @Author  : miliang
# @FileName: config.py
# @Software: PyCharm
import torch
from transformers import BertTokenizer
import torch
import numpy as np
import os
import datetime
from total_utils.common import get_logger

class Config(object):
    def __init__(self):
        self.gpu_id = 0
        self.use_multi_gpu = False
        self.use_pooling = ["max", "avg", "None"][2]

        # train device selection
        if self.use_multi_gpu:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            torch.cuda.set_device(self.gpu_id)
            print('current device:', torch.cuda.current_device())  # watch for current device
            n_gpu = 1
            self.n_gpu = n_gpu

        # 基本参数
        self.bert_hidden = 768
        self.train_epoch = 10
        self.random_seed = 2021
        self.batch_size = 120
        self.sequence_length = 128
        self.random_seed = 2021
        self.dropout_rate = 0.1
        self.warmup_prop = 0.1
        self.clip_grad = 2.0
        self.bert_learning_rate = 5e-5



        self.origin_data_dir = "/home/wangzhili/LiangZ/text_classification/2_classication_pytorch/datasets/origin_data/"
        self.source_data_dir = "/home/wangzhili/LiangZ/text_classification/2_classication_pytorch/datasets/source_data/"
        self.model_save_path = "/home/wangzhili/LiangZ/text_classification/2_classication_pytorch/model_save/"
        self.config_file_path = "/home/wangzhili/LiangZ/text_classification/2_classication_pytorch/config.py"
        self.read_model_file = None

        self.class_list = ['finance', 'realty', 'stocks', 'education', 'science', 'society',
                           'politics', 'sports', 'game', 'entertainment']
        self.class2id = {'finance': 0, 'realty': 1, 'stocks': 2, 'education': 3, 'science': 4, 'society': 5,
                         'politics': 6, 'sports': 7, 'game': 8, 'entertainment': 9}
        self.id2class = {0: 'finance', 1: 'realty', 2: 'stocks', 3: 'education', 4: 'science', 5: 'society',
                         6: 'politics', 7: 'sports', 8: 'game', 9: 'entertainment'}
        self.label_num = len(self.class2id)

        self.pretrain_model_path = "/home/wangzhili/pretrained_model/Torch_model/pytorch_bert_chinese_L-12_H-768_A-12/"
        self.tokenizer = BertTokenizer(vocab_file=self.pretrain_model_path + "/vocab.txt", do_lower_case=True)

    def train_init(self):
        np.random.seed(self.random_seed)
        torch.manual_seed(self.random_seed)
        torch.cuda.manual_seed_all(self.random_seed)
        torch.backends.cudnn.deterministic = True  # 保证每次结果一样
        self.get_save_path()

    def get_save_path(self):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H_%M_%S")
        self.model_save_path = self.model_save_path + "bert_use_pooling_{}_{}".format(self.use_pooling, timestamp)


        if not os.path.exists(self.model_save_path):
            os.makedirs(self.model_save_path)

        # 将config文件写入文件夹中
        with open(self.model_save_path + "/config.test", "w", encoding="utf8") as fw:
            with open(self.config_file_path, "r", encoding="utf8") as fr:
                content = fr.read()
                fw.write(content)

        self.logger = get_logger(self.model_save_path + "/log.log")
        self.logger.info('current device:{}'.format(torch.cuda.current_device()))  # watch for current device

