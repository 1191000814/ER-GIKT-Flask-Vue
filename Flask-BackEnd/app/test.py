"""
测试文件
"""
import heapq
import json
import random
import numpy as np
import torch

model = torch.load('alg/model/2023_05_13#16_29_18.pt')

q_list = [1, 2, 45, 78, 1000]
ones = torch.ones(size=[1, len(q_list)], dtype=torch.int)
c_list = model(
    question=torch.unsqueeze(torch.tensor(q_list), dim=0), # 问题id列表
    response=ones,  # 推荐时回答全设置为1
    mask=ones  # 题目都是有效的，maks也全为1
).squeeze(dim=0).tolist()