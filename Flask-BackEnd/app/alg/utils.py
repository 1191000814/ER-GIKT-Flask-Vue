"""
工具类
"""
import numpy as np
from scipy import sparse

def build_adj_list():
    # 返回每个问题的所有邻居, 每个技能的所有邻居
    qs_table = sparse.load_npz('data/qs_table.npz').toarray() # get qs_table ==> tensor(num_q, num_s)
    num_question = qs_table.shape[0]
    num_skill = qs_table.shape[1]
    q_neighbors_list = [[] for _ in range(num_question)] # 每个问题的邻居(技能)
    s_neighbors_list = [[] for _ in range(num_skill)] # 每个技能的邻居(问题)
    for q_id in range(num_question):
        s_ids = np.reshape(np.argwhere(qs_table[q_id] > 0), [-1]).tolist() # 每个问题涉及的技能id
        q_neighbors_list[q_id] += s_ids
        for s_id in s_ids:
            s_neighbors_list[s_id].append(q_id)
    return q_neighbors_list, s_neighbors_list

def gen_gikt_graph(q_neighbors_list, s_neighbors_list, q_neighbor_size=4, s_neighbor_size=10):
    # 每个问题的固定数量的邻居(随机挑选)构成的矩阵, 每个技能也是同理
    num_question = len(q_neighbors_list)
    num_skill = len(s_neighbors_list)
    q_neighbors = np.zeros([num_question, q_neighbor_size], dtype=np.int32) # 问题的邻居矩阵
    s_neighbors = np.zeros([num_skill, s_neighbor_size], dtype=np.int32) # 技能的邻居矩阵
    for i, neighbors in enumerate(q_neighbors_list):
        if len(neighbors) == 0: # 没有邻居
            continue
        if len(neighbors) >= q_neighbor_size: # 邻居数量超过或等于最大限制,随机选取一些, 但[无重复]
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0: # 邻居数量不够, [有重复]地选取
            q_neighbors[i] = np.random.choice(neighbors, q_neighbor_size, replace=True)
    for i, neighbors in enumerate(s_neighbors_list):
        if len(neighbors) == 0:
            continue
        if len(neighbors) >= s_neighbor_size:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=False)
            continue
        if len(neighbors) > 0:
            s_neighbors[i] = np.random.choice(neighbors, s_neighbor_size, replace=True)
    return q_neighbors, s_neighbors