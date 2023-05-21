"""
测试文件
"""
import numpy as np

import torch
from sklearn.model_selection import KFold
from dataset import UserDataset

# q_list = [1, 2, 45, 78, 1000]
# ones = torch.ones(size=[1, len(q_list)], dtype=torch.int)
# c_list = model(
#     question=torch.unsqueeze(torch.tensor(q_list), dim=0), # 问题id列表
#     response=ones,  # 推荐时回答全设置为1
#     mask=ones  # 题目都是有效的，maks也全为1
# ).squeeze(dim=0).tolist()

k_fold = KFold(shuffle=True) # 默认五折

# 循环进行五折交叉验证
dataset = UserDataset()
for fold, (train_indices, val_indices) in enumerate(k_fold.split(dataset)):
    pass