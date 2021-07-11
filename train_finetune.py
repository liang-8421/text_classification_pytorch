# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 17:09
# @Author  : miliang
# @FileName: train_finetune.py
# @Software: PyCharm

from total_utils.dataiter import DataIterator
from total_utils.train_utils import train
from config import Config

if __name__ == '__main__':
    config = Config()
    config.train_init()
    train_iter = DataIterator(config, data_file=config.source_data_dir + "train.csv", is_test=False)
    dev_iter = DataIterator(config, data_file=config.source_data_dir + "dev.csv", is_test=True)
    train(config, train_iter, dev_iter)
