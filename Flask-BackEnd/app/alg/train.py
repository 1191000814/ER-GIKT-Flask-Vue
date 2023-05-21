"""
仅训练模型
"""
from gikt import GIKT
from utils import gen_gikt_graph, build_adj_list
from params import *
from scipy import sparse
from dataset import UserDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score
from data_process import min_seq_len, max_seq_len
from datetime import datetime
import numpy as np
import torch
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
time_now = datetime.now().strftime('%Y_%m_%d#%H_%M_%S')
output_path = os.path.join('output', time_now)
output_file = open(output_path, 'w')
# 训练时的超参数
params = {
    'max_seq_len': max_seq_len,
    'min_seq_len': min_seq_len,
    'epochs': 1,
    'lr': 0.005,
    'lr_gamma': 0.95,
    'batch_size': 32,
    'size_q_neighbors': 4,
    'size_s_neighbors': 10,
    'num_workers': 16,
    'prefetch_factor': 4,
    'agg_hops': 3,
    'emb_dim': 100,
    'hard_recap': True,
    'dropout': (0.2, 0.4),
    'rank_k': 10
}
# 打印并写超参数
output_file.write(str(params) + '\n')
print(params)
# 构建模型需要的数据数据结构, 全部转化为正确类型tensor再输入模型中
qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE) # [num_q, num_c]
num_question = torch.tensor(qs_table.shape[0], device=DEVICE)
num_skill = torch.tensor(qs_table.shape[1], device=DEVICE)
q_neighbors_list, s_neighbors_list = build_adj_list()
q_neighbors, s_neighbors = gen_gikt_graph(q_neighbors_list, s_neighbors_list, params['size_q_neighbors'], params['size_s_neighbors'])
q_neighbors = torch.tensor(q_neighbors, dtype=torch.int64, device=DEVICE)
s_neighbors = torch.tensor(s_neighbors, dtype=torch.int64, device=DEVICE)

# 初始化模型
model = GIKT(
    num_question, num_skill, q_neighbors, s_neighbors, qs_table,
    agg_hops=params['agg_hops'],
    emb_dim=params['emb_dim'],
    dropout=params['dropout'],
    hard_recap=params['hard_recap'],
).to(DEVICE)

dataset = UserDataset()
data_len = len(dataset) # 数据总长度
loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) # 损失函数
if DEVICE.type == 'cpu': # cpu, 本机
    data_loader = DataLoader(dataset, batch_size=params['batch_size']) # 数据加载器
else: # gpu, 服务器
    data_loader = DataLoader(dataset, batch_size=params['batch_size'], num_workers=params['num_workers'],
                             pin_memory=True,prefetch_factor=params['prefetch_factor'])
print('model has been built')

# 开始训练
optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
torch.optim.lr_scheduler.ExponentialLR(optimizer, params['lr_gamma'])
# 在matplotlib中绘制的y轴数据，三行分别表示loss, acc, auc
y_label = np.zeros([3, params['epochs']])

for epoch in range(params['epochs']):
    print('===================' + LOG_Y + 'epoch: {}'.format(epoch + 1) + LOG_END + '====================')
    output_file.write('epoch{}: '.format(epoch + 1))
    time0 = time.time()
    train_step = 0 # 每轮训练第几个批量
    loss_val = 0 # 总损失
    num_train = 0 # 训练的真实样本个数
    num_acc = 0 # 其中正确的个数
    auc_aver = 0 # 总体的auc
    for data in data_loader:
        # 梯度清零
        optimizer.zero_grad()
        x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
        y_hat = model(x, y_target, mask)
        y_hat = torch.masked_select(y_hat, mask)
        y_pred = torch.ge(y_hat, torch.tensor(0.5))
        y_target = torch.masked_select(y_target, mask)
        loss = loss_fun(y_hat, y_target.to(torch.float32))
        loss_val += loss.item()
        # 计算acc
        acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
        num_acc += torch.sum(torch.eq(y_target, y_pred))
        num_train += torch.sum(mask)
        # 计算auc
        auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
        auc_aver += auc * len(x) / data_len
        loss.backward()
        optimizer.step()
        train_step += 1
        print(f'train_step: {train_step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
    time1 = time.time()
    run_time = time1 - time0
    loss_aver, acc_aver = loss_val / train_step, num_acc / num_train
    # 保存至数组，之后用matplotlib画图
    y_label[0][epoch], y_label[1][epoch], y_label[2][epoch] = loss_aver, acc_aver, auc_aver
    # 输出至终端
    print(LOG_B + f'epoch: {epoch}, loss: {loss_aver:.4f}, acc: {acc_aver:.4f}, auc: {auc_aver: .4f}' + LOG_END)
    print(LOG_B + f'time: {run_time:.2f}s, average batch time: {(run_time / train_step):.2f}s' + LOG_END)
    # 保存输出至本地文件
    output_file.write(f'epoch: {epoch}, loss: {loss_aver:.4f}, acc: {acc_aver:.4f}, auc: {auc_aver: 4f}\n')
    output_file.write(f'time: {run_time:.2f}s, average batch time: {(run_time / train_step):.2f}s\n')
output_file.close()
torch.save(model, f=f'model/{time_now}.pt')
np.savetxt(f'chart_data/{time_now}.txt', y_label)