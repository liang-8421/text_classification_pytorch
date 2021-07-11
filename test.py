# -*- coding: utf-8 -*-
# @Time    : 2021/7/2 16:53
# @Author  : miliang
# @FileName: test.py
# @Software: PyCharm
from total_utils.common import fenge
import torch

temp = torch.randn((2, 2), dtype=torch.float32, device="cuda:0")

print(temp.cpu().numpy())

