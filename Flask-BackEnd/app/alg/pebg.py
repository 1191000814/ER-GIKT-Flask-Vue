"""
PEBG问题向量预训练模型
仅仅用于提取问题和技能之间的关系,不涉及具体用户
"""
import numpy as np
import torch.nn as nn
from icecream import ic
from params import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
ic(torch.cuda.is_available())
ic(device)

class PEBG(nn.Module):

    def __init__(self, qs_table, qq_table, ss_table, emb_dim=100):
        super().__init__()  # 父类初始化是必要的
        num_q, num_s = qs_table.shape[0], qs_table.shape[1]
        # 定义已知(对比)变量
        self.qs_target = torch.as_tensor(qs_table, dtype=torch.float, device=DEVICE) # [num_q, num_s] 问题-技能表
        self.qq_target = torch.as_tensor(qq_table, dtype=torch.float, device=DEVICE) # [num_q, num_s] 问题-问题表
        self.ss_target = torch.as_tensor(ss_table, dtype=torch.float, device=DEVICE) # [num_s, num_s] 技能-技能表
        q_feature = torch.tensor(np.load('data/q_feature.npy')).float().to(device)  # [num_q, size_q_feature] 问题属性特征,已生成
        self.q_diff = q_feature[:, :-1]  # [num_q, size_q_feature-1] 属性特征除了最后一位都可以作为难度特征 []
        self.d_target = q_feature[:, -1]  # [num_q, 1] 属性特征最后一维就是正确率
        # 需要训练的参数
        self.q_embedding = nn.Parameter(torch.randn(size=[num_q, emb_dim])).to(device)  # 问题顶点特征
        self.s_embedding = nn.Parameter(torch.randn(size=[num_s, emb_dim])).to(device)  # 技能顶点特征(只有123个数据,不区分批量与否)
        # self.q_embedding = nn.Parameter(torch.concat([self.qs_target, self.q_diff], dim=1)) # [num_q, num_s + 7]
        # self.s_embedding = nn.Parameter(self.ss_target) # [num_s, num_s]
        # 定义网络层
        self.fc_q = nn.Linear(self.q_embedding.shape[1], emb_dim) # [num_q, emb_dim]
        self.fc_s = nn.Linear(self.s_embedding.shape[1], emb_dim) # [num_q. emb_dim]
        self.relu_q = nn.ReLU()
        self.relu_s = nn.ReLU()
        self.sigmoid = [nn.Sigmoid().to(device) for _ in range(3)]
        ic('PEBG model built')

    def forward(self, q_embedding, s_embedding):
        # 接收已经确定好批次的输入向量
        q_embedding_fc = self.relu_q(self.fc_q(q_embedding)) # [batch_size_q, emb_dim]
        s_embedding_fc = self.relu_s(self.fc_s(s_embedding)) # [num_s, emb_dim]
        qs_logit = self.sigmoid[0](torch.matmul(q_embedding_fc, s_embedding_fc.T))  # [batch_size_q, num_s] 公式1
        qq_logit = self.sigmoid[1](torch.matmul(q_embedding_fc, q_embedding_fc.T))  # [batch_size_q, size_batch] 公式5
        ss_logit = self.sigmoid[2](torch.matmul(s_embedding_fc, s_embedding_fc.T))  # [num_s, num_s] 公式6
        return qs_logit, qq_logit, ss_logit