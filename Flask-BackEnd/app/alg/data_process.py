"""
预处理文件
"""

import pandas as pd
import numpy as np
import os
from scipy import sparse
from icecream import ic

min_seq_len = 20 # 最短用户的答题序列
max_seq_len = 200 # 最长用户答题序列

# 数据处理主要策略：
# 本地存在文件：直接加载
# 本地不存在文件：计算、保存
if __name__ == '__main__':
    data = pd.read_csv(filepath_or_buffer='data/assist09_origin.csv', encoding="ISO-8859-1")
    # 原始数据
    ic(data.shape)

    data = data.sort_values(by='user_id', ascending=True)
    ic('按用户id排序完成')

    # 清洗一些无用数据
    data = data.drop(data[data['skill_id'] == 'NA'].index) # 删除技能为NA的行
    data = data.dropna(subset=['skill_id']) # 清除技能为NAN的行
    data = data.drop(data[data['original'] == 0].index) # 删除origin类型为0

    # 统计每个学生回答的问题数量, 并创建布尔索引，标记回答问题不超过5个的学生
    is_valid_user = data.groupby('user_id').size() >= min_seq_len
    # 使用布尔索引过滤数据，删除回答问题不超过5个的学生的数据
    data = data[data['user_id'].isin(is_valid_user[is_valid_user].index)]
    ic('数据行清洗完成')

    data = data.loc[:, ['order_id', 'user_id', 'problem_id', 'correct', 'skill_id', 'skill_name',
                        'ms_first_response', 'answer_type', 'attempt_count']] # 只保留有用的列
    ic('数据列清洗完成')
    ic(data.shape)

    # 统计工作
    num_answer = data.shape[0] # 数据集的总长度
    questions = set()
    skills = set()
    users = set()

    # 不加索引的时候, row[1]是user_id, row[2]是problem_id, row[4]是skill_id
    # 这里并没有在df数据集中修改, 因为时间开销太大
    for row in data.itertuples(index=False):
        users.add(row[1])
        questions.add(row[2])
        if isinstance(row[4], (int, float)):
            skills.add(int(row[4]))
        else:
            skill_add = set(int(s) for s in row[4].split('_'))
            # 按'_'分隔, 并转化为整数, 再加入skills中
            skills = skills.union(skill_add)

    data.to_csv('data/assist09_processed.csv', sep=',', index=False)

    num_q = len(questions)
    num_s = len(skills)
    num_user = len(users)

    # 本地的字典
    if not os.path.exists('data/question2idx.npy'):
        # 转化为列表
        questions = list(questions)
        skills = list(skills)
        users = list(users)

        # 原id-新id(从0开始递增)
        question2idx = {questions[i]: i + 1 for i in range(num_q)} # 从1开始，0留给空白
        question2idx[0] = 0 # 空白问题，用于padding
        skill2idx = {skills[i]: i for i in range(num_s)}
        user2idx = {users[i]: i for i in range(num_user)}
        num_q += 1
        ic(num_q, num_s, num_user)

        # 新id-原id
        idx2question = {question2idx[q]: q for q in question2idx}
        idx2skill = {skill2idx[s]: s for s in skill2idx}
        idx2user = {user2idx[u]: u for u in user2idx}
        # 保存字典
        np.save('data/question2idx.npy', question2idx)
        np.save('data/skill2idx.npy', skill2idx)
        np.save('data/user2idx.npy', user2idx)
        np.save('data/idx2question.npy', idx2question)
        np.save('data/idx2skill.npy', idx2skill)
        np.save('data/idx2user.npy', idx2user)
    else:
        question2idx = np.load('data/question2idx.npy', allow_pickle=True).item()
        skill2idx = np.load('data/skill2idx.npy', allow_pickle=True).item()
        user2idx = np.load('data/user2idx.npy', allow_pickle=True).item()
        idx2question = np.load('data/idx2question.npy', allow_pickle=True).item()
        idx2skill = np.load('data/idx2skill.npy', allow_pickle=True).item()
        idx2user = np.load('data/idx2user.npy', allow_pickle=True).item()

    # 本地的邻接矩阵
    if not os.path.exists('data/qs_table.npz'):
        qs_table = np.zeros([num_q, num_s], dtype=int)  # 问题-技能矩阵
        q_set = data['problem_id'].drop_duplicates() # 以每个问题为单位
        q_samples = pd.concat([data[data['problem_id'] == q_id].sample(1) for q_id in q_set])  # 每个问题选一行
        # 构建问题-技能表
        for row in q_samples.itertuples(index=False):
            # row[2]: 问题原id, row[4]: 技能原id
            if isinstance(row[4], (int, float)): # 单一个技能
                qs_table[question2idx[row[2]], skill2idx[int(row[4])]] = 1
            else: # 用'_'连接的技能
                skill_add = [int(s) for s in row[4].split('_')]
                for s in skill_add:
                    qs_table[question2idx[row[2]], skill2idx[s]] = 1
        ic('问题-技能矩阵构建完成')

        # 构建问题-问题表, 技能-技能表
        qq_table = np.matmul(qs_table, qs_table.T) # 问题-问题矩阵 [num_q, num_q]
        ss_table = np.matmul(qs_table.T, qs_table) # 技能-技能矩阵 [num_s, num_s]

        ic('三个邻接矩阵全部构建完成')

        # 转化为稀疏矩阵保存至本地
        qs_table = sparse.coo_matrix(qs_table)
        qq_table = sparse.coo_matrix(qq_table)
        ss_table = sparse.coo_matrix(ss_table)

        sparse.save_npz('data/qs_table.npz', qs_table)
        sparse.save_npz('data/qq_table.npz', qq_table)
        sparse.save_npz('data/ss_table.npz', ss_table)

    else:
        qs_table = sparse.load_npz('data/qs_table.npz').toarray()
        qq_table = sparse.load_npz('data/qq_table.npz').toarray()
        ss_table = sparse.load_npz('data/ss_table.npz').toarray()

    # 构建不同用户的问答序列, 记录每个用户回答的问题id
    # 一个用户的最长序列为200, 不足的补零, 超过的舍去
    # 同时记录mask (哪些是真正存在的答题序列:1, 哪些是补零的数据: 0)
    if not os.path.exists('data/user_seq.npy'):
        user_seq = np.zeros([num_user, max_seq_len]) # 问题序列
        user_res = np.zeros([num_user, max_seq_len]) # 回答结果{0, 1}序列
        num_seq = [0 for _ in range(num_user)] # 当前用户长度
        user_mask = np.zeros([num_user, max_seq_len])
        # 不加索引的时候, row[1]是user_id, row[2]是problem_id, row[3]是回答正确与否, row[4]是skill_id
        # 当前data是按照user_id排序的
        for row in data.itertuples(index=False):
            user_id = user2idx[row[1]] # 用户id
            if num_seq[user_id] < max_seq_len - 1: # 已经是最大长度了
                user_seq[user_id, num_seq[user_id]] = question2idx[row[2]]
                user_res[user_id, num_seq[user_id]] = row[3]
                user_mask[user_id, num_seq[user_id]] = 1
                num_seq[user_id] += 1
        np.save('data/user_seq.npy', user_seq)
        np.save('data/user_res.npy', user_res)
        np.save('data/user_mask.npy', user_mask)
    ic('字典加载成功')

    # 构建问题的属性(attribute)特征向量, 公式: q = [平均反应时间, 平均回答次数, 问题类型(5, one-hot), 平均正确率]
    if not os.path.exists('data/q_feature.npy'):
        q_type_list = data['answer_type'].unique() # 所有的问题类型（只有五种）
        num_types = len(q_type_list) # 问题类型数量
        q_type_dict = dict(zip(q_type_list, range(num_types)))
        q_feature = np.zeros([num_q, num_types + 3]) # 属性特征

        for idx in idx2question: # 每个问题
            if idx == 0: # 第一个是空白问题
                q_feature[idx] = np.zeros([8,])
                continue
            q_temp = data[data['problem_id'] == idx2question[idx]] # 该id的问题
            _mean_time = np.atleast_1d(q_temp['ms_first_response'].mean())
            _mean_count = np.atleast_1d(q_temp['attempt_count'].mean())
            _type = np.zeros([num_types, ], float)
            _type[q_type_dict[q_temp.iloc[0]['answer_type']]] = 1
            _accuracy = np.atleast_1d(q_temp['correct'].mean())
            q_feature[idx] = np.concatenate([_mean_time, _mean_count, _type, _accuracy])

        ic('问题属性特征构建完成')
        np.save('data/q_feature.npy', q_feature)

    ic('数据全部保存至本地')