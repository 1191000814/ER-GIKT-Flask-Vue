"""
训练并测试模型
"""
import math
import os
import time
from datetime import datetime
import numpy as np
from scipy import sparse
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Subset
from data_process import min_seq_len, max_seq_len
from dataset import UserDataset
from gikt import GIKT
from params import *
from utils import gen_gikt_graph, build_adj_list

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
time_now = datetime.now().strftime('%Y_%m_%d#%H_%M_%S')
output_path = os.path.join('output', time_now)
output_file = open(output_path, 'w')
# 训练时的超参数
params = {
    'max_seq_len': max_seq_len,
    'min_seq_len': min_seq_len,
    'epochs': 2, # 每折训练的轮数
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
    'rank_k': 10,
    'k_fold': 5  # 几折交叉验证
}

# 打印并写超参数
output_file.write(str(params) + '\n')
print(params)
batch_size = params['batch_size']
# 构建模型需要的数据结构, 全部转化为正确类型tensor再输入模型中
qs_table = torch.tensor(sparse.load_npz('data/qs_table.npz').toarray(), dtype=torch.int64, device=DEVICE)  # [num_q, num_c]
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

loss_fun = torch.nn.BCEWithLogitsLoss().to(DEVICE) # 损失函数
dataset = UserDataset()  # 数据集
data_len = len(dataset)  # 数据总长度
print('model has been built')

# 优化器
epoch_total = 0
optimizer = torch.optim.Adam(params=model.parameters(), lr=params['lr'])
torch.optim.lr_scheduler.ExponentialLR(optimizer, params['lr_gamma'])
# 在matplotlib中绘制的y轴数据，三行分别表示loss, acc, auc
y_label = np.zeros([3, params['epochs']])

k_fold = KFold(n_splits=params['k_fold'] ,shuffle=True) # 交叉验证
for fold, (train_indices, test_indices) in enumerate(k_fold.split(dataset)):
    train_set = Subset(dataset, train_indices)  # 训练集
    test_set = Subset(dataset, test_indices)  # 测试集
    if DEVICE.type == 'cpu':  # Cpu(本机)
        train_loader = DataLoader(train_set, batch_size=batch_size)  # 训练数据加载器
        test_loader = DataLoader(test_set, batch_size=batch_size)  # 测试数据加载器
    else:  # Gpu(服务器)
        train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=params['num_workers'],
                                  pin_memory=True, prefetch_factor=params['prefetch_factor'])
        test_loader = DataLoader(test_set, batch_size=batch_size, num_workers=params['num_workers'],
                                 pin_memory=True, prefetch_factor=params['prefetch_factor'])
    train_data_len, test_data_len = len(train_set), len(test_set)
    for epoch in range(params['epochs']):
        # 使用五折交叉验证（手动实现），每次的训练集和测试集都不相同
        print('===================' + LOG_Y + 'epoch: {}'.format(epoch_total + 1) + LOG_END + '====================')
        print('-------------------training------------------')
        output_file.write(f'epoch{epoch_total + 1}: ')
        time0 = time.time()
        step = 0  # 每轮训练第几个批量
        train_loss = 0  # 总损失
        num_total = 0  # 训练的真实样本个数
        num_right = 0  # 其中正确的个数
        train_auc_aver = 0  # 总体训练的auc
        # 训练阶段
        for data in train_loader:
            # 梯度清零
            optimizer.zero_grad()
            x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
            # 从data取出来时，mask的类型是int而不是bool
            y_hat = model(x, y_target, mask)
            y_hat = torch.masked_select(y_hat, mask)
            y_pred = torch.ge(y_hat, torch.tensor(0.5))
            y_target = torch.masked_select(y_target, mask)
            loss = loss_fun(y_hat, y_target.to(torch.float32))
            train_loss += loss.item()
            # 计算acc
            acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
            num_right += torch.sum(torch.eq(y_target, y_pred))
            num_total += torch.sum(mask)
            # 计算auc
            auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
            train_auc_aver += auc * len(x) / train_data_len
            loss.backward()
            optimizer.step()
            step += 1
            print(f'step: {step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
        train_loss_aver, train_acc_aver = train_loss / step, num_right / num_total
        print('-------------------testing------------------')
        step = 0  # 每轮训练第几个批量
        test_loss = 0  # 总损失
        num_total = 0  # 训练的真实样本个数
        num_right = 0  # 其中正确的个数
        test_auc_aver = 0  # 总体的auc
        # 测试阶段，只有前向传递，没有反向传播阶段
        for data in test_loader:
            x, y_target, mask = data[:, :, 0].to(DEVICE), data[:, :, 1].to(DEVICE), data[:, :, 2].to(torch.bool).to(DEVICE)
            y_hat = model(x, y_target, mask)
            y_hat = torch.masked_select(y_hat, mask.to(torch.bool))
            y_pred = torch.ge(y_hat, torch.tensor(0.5))
            y_target = torch.masked_select(y_target, mask.to(torch.bool))
            loss = loss_fun(y_hat, y_target.to(torch.float32))
            test_loss += loss.item()
            # 计算acc
            acc = torch.sum(torch.eq(y_target, y_pred)) / torch.sum(mask)
            num_right += torch.sum(torch.eq(y_target, y_pred))
            num_total += torch.sum(mask)
            # 计算auc
            auc = roc_auc_score(y_target.cpu(), y_pred.cpu())
            test_auc_aver += auc * len(x) / test_data_len
            step += 1
            print(f'step: {step}, loss: {loss.item():.4f}, acc: {acc.item():.4f}, auc: {auc:.4f}')
        epoch_total += 1
        test_loss_aver, test_acc_aver = test_loss / step, num_right / num_total
        time1 = time.time()
        run_time = time1 - time0
        print(LOG_B + f'epoch: {epoch_total}' + LOG_END)
        print(LOG_B + f'training: loss: {train_loss_aver:.4f}, acc: {train_acc_aver:.4f}, auc: {train_auc_aver: .4f}' + LOG_END)
        print(LOG_B + f'testing: loss: {test_loss_aver:.4f}, acc: {test_acc_aver:.4f}, auc: {test_auc_aver: .4f}' + LOG_END)
        print(LOG_B + f'time: {run_time:.2f}s, average batch time: {(run_time / step):.2f}s' + LOG_END)
        # 保存输出至本地文件
        # output_file.write(f'epoch: {epoch}, loss: {loss_aver:.4f}, acc: {acc_aver:.4f}, auc: {auc_aver: 4f}\n')
        # output_file.write(f'time: {run_time:.2f}s, average batch time: {(run_time / step):.2f}s\n')
        # 保存至数组，之后用matplotlib画图
        # y_label[0][epoch], y_label[1][epoch], y_label[2][epoch] = loss_aver, acc_aver, auc_aver
output_file.close()
torch.save(model, f=f'model/{time_now}.pt')
np.savetxt(f'chart_data/{time_now}.txt', y_label)