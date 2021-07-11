# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 20:43
# @Author  : miliang
# @FileName: predict_utils.py
# @Software: PyCharm

from config import Config
from total_utils.dataiter import DataIterator
from total_utils.predict_utils import predict

if __name__ == '__main__':
    config = Config()
    test_iter = DataIterator(config, data_file=config.source_data_dir + "test.csv", is_test=False)
    predict(config, test_iter)
