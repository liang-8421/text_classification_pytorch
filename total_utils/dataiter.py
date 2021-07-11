# -*- coding: utf-8 -*-
# @Time    : 2021/6/14 18:51
# @Author  : miliang
# @FileName: dataiter.py
# @Software: PyCharm

from total_utils.dataload import create_example
import numpy as np
import pandas as pd
from config import Config
import torch


class DataIterator(object):
    """
	数据迭代器
	"""

    def __init__(self, config, data_file, is_test=False):
        # config 参数传递
        self.batch_size = config.batch_size
        self.seq_length = config.sequence_length
        self.tokenizer = config.tokenizer
        self.class2id = config.class2id
        self.device = config.device

        # 数据的操作
        self.data = create_example(data_file)
        self.num_records = len(self.data)  # 数据的个数
        self.idx = 0  # 数据索引
        self.all_idx = list(range(self.num_records))  # 全体数据索引
        self.is_test = is_test

        if not self.is_test:
            self.shuffle()

        print("样本个数：", self.num_records)

    def shuffle(self):
        np.random.shuffle(self.all_idx)

    def convert_single_example(self, example_idx):
        text_list = list(self.data[example_idx].text)
        label = self.class2id[self.data[example_idx].label]
        if len(text_list) > self.seq_length - 2:
            text_list = text_list[:(self.seq_length - 2)]

        tokens = []
        for index, token in enumerate(text_list):
            # ntokens.append(self.tokenizer.tokenize(token.lower())[0])  # 全部转换成小写, 方便BERT词典
            char_list = self.tokenizer.tokenize(token.lower())
            if char_list:
                tokens.append(char_list[0])
            else:
                # 解析不到的特殊字符用 §替代
                tokens.append("§")

        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)

        input_ids_cpu = np.zeros((1, self.seq_length), dtype=np.int64)
        input_mask_cpu = np.zeros((1, self.seq_length), dtype=np.int64)
        segment_ids_cpu = np.zeros((1, self.seq_length), dtype=np.int64)
        label_cpu = np.zeros((1,), dtype=np.int64)
        tokens_cpu = np.array(["*NULL*"] * self.seq_length)
        tokens_cpu = np.expand_dims(tokens_cpu, axis=0)

        label_cpu[0] = label
        input_ids_cpu[0, :len(input_ids)] = input_ids
        input_mask_cpu[0, :len(input_ids)] = [1] * len(input_ids)
        tokens_cpu[0, :len(tokens)] = tokens

        return input_ids_cpu, input_mask_cpu, segment_ids_cpu, label_cpu, tokens_cpu

    def __iter__(self):
        return self

    def __next__(self):
        if self.idx >= self.num_records:  # 迭代停止条件
            self.idx = 0
            if not self.is_test:
                self.shuffle()
            raise StopIteration
        input_ids_cpu, input_mask_cpu, segment_ids_cpu = [], [], []
        label_cpu, tokens_cpu = [], []

        batch_count = 0
        while batch_count < self.batch_size:  # 每次返回batch_size个数据
            idx = self.all_idx[self.idx]
            res = self.convert_single_example(idx)

            single_input_ids, single_input_mask, single_segment_ids, single_labels, single_tokens = res

            if batch_count == 0:
                input_ids_cpu, input_mask_cpu, segment_ids_cpu, label_cpu, tokens_cpu = \
                    single_input_ids, single_input_mask, single_segment_ids, single_labels, single_tokens
            else:
                input_ids_cpu = np.concatenate((input_ids_cpu, single_input_ids), axis=0)
                input_mask_cpu = np.concatenate((input_mask_cpu, single_input_mask), axis=0)
                segment_ids_cpu = np.concatenate((segment_ids_cpu, single_segment_ids), axis=0)
                label_cpu = np.concatenate((label_cpu, single_labels), axis=0)
                tokens_cpu = np.concatenate((tokens_cpu, single_tokens), axis=0)

            batch_count += 1
            self.idx += 1
            if self.idx >= self.num_records:
                break

        input_ids = torch.from_numpy(input_ids_cpu).to(self.device)
        input_mask = torch.from_numpy(input_mask_cpu).to(self.device)
        segment_ids = torch.from_numpy(segment_ids_cpu).to(self.device)
        labels = torch.from_numpy(label_cpu).to(self.device)

        return input_ids, input_mask, segment_ids, labels, tokens_cpu

    def __len__(self):
        # 返回训练的步数
        if self.num_records % self.batch_size == 0:
            return self.num_records // self.batch_size
        else:
            return self.num_records // self.batch_size + 1


if __name__ == '__main__':
    config = Config()
    test_iter = DataIterator(config, config.source_data_dir + "test.csv", is_test=True)
    print(len(test_iter))
    for input_ids, input_mask, segment_ids, labels, tokens_list in test_iter:
        print(input_ids)
        print(input_mask)
        print(segment_ids)
        print(labels)
        print(tokens_list)
        print(len(segment_ids))
