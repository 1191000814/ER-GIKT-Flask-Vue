"""
数据设置
"""
import torch

# 计算设备: 优先使用gpu
DEVICE = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")

LOG_B = '\033[1;34m' # 蓝色
LOG_Y = '\033[1;33m' # 黄色
LOG_G = '\033[1;36m' # 深绿色
LOG_END = '\033[m' # 结束标记

# 数据超参数
size_q_feature = 8
size_embedding = 200 # 问题和技能的嵌入向量维度