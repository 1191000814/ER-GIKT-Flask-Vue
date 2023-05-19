"""
创建初始数据
"""
import traceback
import numpy as np
import pandas as pd
from icecream import ic
from scipy import sparse
from . import db
from .entity import Question, Skill, Answer

# 通过assist09数据集中的数据初始化mysql数据库
def create_data():
    data = pd.read_csv('app/alg/data/assist09_processed.csv', encoding='ISO-8859-1')
    question2idx = np.load('app/alg/data/question2idx.npy', allow_pickle=True).item()
    skill2idx = np.load('app/alg/data/skill2idx.npy', allow_pickle=True).item()
    user2idx = np.load('app/alg/data/user2idx.npy', allow_pickle=True).item()
    qs_table = sparse.load_npz('app/alg/data/qs_table.npz').toarray()
    questions, skills = set(), set() # 已经加载到数据库中id的集合
    # 0:'order_id', 1:'user_id', 2:'problem_id', 3:'correct', 4:'skill_id',
    # 5:'skill_name',6:'ms_first_response', 7:'answer_type', 8:'attempt_count'
    print('开始创建初始数据')
    row_i = 0
    for row in data.itertuples(index=False):
        try:
            # 先处理一下每行的数据
            q_id = question2idx[row[2]]
            user_id = user2idx[row[1]]
            skills_str = None # 技能的字符串表示
            if isinstance(row[4], (int, float)): # 只有一个技能（整数/浮点数）
                s_id = skill2idx[row[4]]
                skills_str = str(s_id)
                if row[4] not in skills:
                    skills.add(int(row[4]))
                    db.session.add(Skill(id=s_id, name=row[5], q_num=np.sum(qs_table[:, s_id])))
            else: # 有多个技能（用下划线_连接的字符串）
                skill_add = set(int(float(s)) for s in row[4].split('_'))
                skills_str = ' '.join([str(skill2idx[s_id]) for s_id in skill_add])
                # 用空格代替下划线'_'连接新的技能id
                for skill in skill_add:
                    if skill not in skills:
                        s_id = skill2idx[skill]
                        skills.add(skill)
                        db.session.add(Skill(id=s_id, name=None if not isinstance(row[5], str) else row[5], num_q=np.sum(qs_table[:, s_id])))
            if row[2] not in questions:
                db.session.add(Question(id=q_id, type=row[7], skills=skills_str))
                questions.add(row[2])
            db.session.add(Answer(user_id=user_id, q_id=q_id, correct=row[3]))
            row_i += 1
        except Exception:
            traceback.print_exc()
            ic(row_i)
            print('创建数据失败，开始回滚')
            db.session.rollback()
            print('回滚完成')
            return
    print('创建数据成功，开始提交')
    db.session.commit()
    print('提交完成')
    return