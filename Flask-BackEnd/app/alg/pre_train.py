"""
预训练问题和技能向量
"""
import math

import torch
import torch.nn as nn
from scipy import sparse
from params import DEVICE
from pebg import PEBG
from icecream import ic
from torch.utils.tensorboard import SummaryWriter

# 批量训练, 每次训练batch_size个问题,并值分析这些问题相关的技能
# 只取数据的批量的问题, 每次批量都将所有的技能考虑进计算
# model内的变量都是全部变量, 不带的是批量变量

qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
qq_table = torch.tensor(sparse.load_npz('data/qq_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
ss_table = torch.tensor(sparse.load_npz('data/ss_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
num_q = qs_table.shape[0]
num_s = qs_table.shape[1]
batch_size = 256
num_batch = math.ceil(num_q / batch_size)

model = PEBG(qs_table, qq_table, ss_table, emb_dim=200)
ic('开始训练模型')
# optimizer = torch.optim.Adam(params=list(model.parameters()) + list(), lr=0.001) # 优化器
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)
mse_loss = [nn.MSELoss().to(DEVICE) for _ in range(3)]
writer = SummaryWriter(log_dir='logs')
for epoch in range(20):
    train_loss = 0  # 总损失
    for idx_batch in range(num_batch):
        optimizer.zero_grad()  # 梯度清零
        # 计算索引
        idx_start = idx_batch * batch_size
        idx_end = min((idx_batch + 1) * batch_size, num_q)  # 结束索引,注意是否超过了问题数
        # 通过索引计算相应的批量
        q_embedding = model.q_embedding[idx_start: idx_end]  # [问题]的顶点特征(批量) [num_batch, size_embed]
        s_embedding = model.s_embedding  # 与批量无关 [num_s, embed_size]
        qs_target = model.qs_target[idx_start: idx_end]  # [size_batch, num_s]
        qq_target = model.qq_target[idx_start: idx_end, idx_start: idx_end]  # [size_batch, size_batch]
        ss_target = model.ss_target  # 与批量无关 [nums, num_s]
        # 计算logit
        qs_logit, qq_logit, ss_logit = model.forward(q_embedding, s_embedding)
        # 计算损失
        loss_qs = torch.sqrt(mse_loss[0](qs_logit, qs_target)).sum()  # L1
        loss_qq = torch.sqrt(mse_loss[1](qq_logit, qq_target)).sum()  # L2
        loss_ss = torch.sqrt(mse_loss[2](ss_logit, ss_target)).sum()  # L3
        loss = loss_qs + loss_qq + loss_qq  # 总损失
        train_loss += loss.item()
        loss.backward()  # 反向传播
        optimizer.step()  # 参数优化
        # if idx_batch % 10 == 0:
        #     print(f'epoch: {epoch}, idx_batch: {idx_batch}, loss: {loss.item()}, loss_qs: {loss_qs.item()}, '
        #           f'loss_qq: {loss_qq.item()}, loss_ss: {loss_ss.item()}')
    print(f'----------epoch: {epoch + 1}, train_loss: {train_loss}')
    writer.add_scalar(tag='pebg_loss_dim200', scalar_value=train_loss, global_step=epoch)
ic(model.q_embedding, model.s_embedding)
torch.save(model.q_embedding, f='data/q_embedding.pt')
torch.save(model.s_embedding, f='data/s_embedding.pt')
# writer.add_graph(model, input_to_model=[model.q_embedding[0], model.s_embedding[0]])
writer.close()