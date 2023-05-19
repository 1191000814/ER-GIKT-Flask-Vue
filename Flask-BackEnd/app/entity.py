"""
实体类模型
"""
from . import db

class User(db.Model):
    """用户类"""
    # 继承db.Model已经自动是实现了__init__方法
    __tablename__ = 'user'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    username = db.Column(db.String(255), nullable=False)
    password = db.Column(db.String(255), nullable=False)
    sex = db.Column(db.Integer, nullable=False) # 性别：男1/女0
    age = db.Column(db.String(255), nullable=False)
    role = db.Column(db.Integer, nullable=False) # 角色: 普通用户1/管理员0
    # 关联属性history_answers: 所有历史问题列表

    def __repr__(self):
        return f'User[id: {self.id}, username: {self.username}, password: {self.password}, sex: {self.sex} ' \
               f'age: {self.age} role: {self.role}]'

    # 实现下面两个方法就可以使用函数python内置函数：dict()
    def keys(self):
        return 'id', 'username', 'password', 'sex', 'age', 'role'

    def __getitem__(self, item):
        return getattr(self, item)

    # 从字典中获取属性创建对象
    @classmethod
    def from_dict(cls, object_dict):
        user = cls()
        for attr in object_dict:
            if hasattr(user, attr):
                user.__setattr__(attr, object_dict[attr])
        return user

class Question(db.Model):
    __tablename__ = 'question'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    type = db.Column(db.String(255), nullable=True)
    skills = db.Column(db.String(255), nullable=True) # 技能id的字符串拼接
    # 关联属性related_answers: 所有相关的问题列表

    def __repr__(self):
        return f'Question[id: {self.id}, type: {self.type}, skills: {self.skills}]'

    # 实现下面两个方法就可以使用函数python内置函数：dict()
    def keys(self):
        return 'id', 'type', 'skills'

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def from_dict(cls, object_dict):
        question = cls()
        for attr in object_dict:
            if hasattr(question, attr):
                question.__setattr__(attr, object_dict[attr])
        return question

class Skill(db.Model):
    __tablename__ = 'skill'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    name = db.Column(db.String(255), nullable=True)
    num_q = db.Column(db.Integer, nullable=False) # 相关的问题数量

    def __repr__(self):
        return f'Skill[id: {self.id} name: {self.s_name}, num_q: {self.num_q}]'

    # 实现下面两个方法就可以使用函数python内置函数：dict()
    def keys(self):
        return 'id', 'name', 'num_q'

    def __getitem__(self, item):
        return getattr(self, item)

    @classmethod
    def from_dict(cls, object_dict):
        skill = cls()
        for attr in object_dict:
            if hasattr(skill, attr):
                skill.__setattr__(attr, object_dict[attr])
        return skill

class Answer(db.Model):
    __tablename__ = 'answer'
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    q_id = db.Column(db.Integer, db.ForeignKey('question.id', ondelete='RESTRICT'), nullable=False) # 外键1
    user_id = db.Column(db.Integer, db.ForeignKey('user.id', ondelete='RESTRICT'), nullable=False) # 外键2
    correct = db.Column(db.Integer, nullable=False) # 1为正确，0位错误
    question = db.relationship('Question', backref='related_answers') # 外键1关联属性对象
    user = db.relationship('User', backref='history_answers') # 外键2关联属性对象

    def __repr__(self):
        return f'Answer[id: {self.id}, q_id: {self.q_id}, user_id: {self.user_id}, correct: {self.correct}]'

    # 实现下面两个方法就可以使用函数python内置函数：dict()
    def keys(self):
        return 'id', 'q_id', 'user_id', 'correct'

    def __getitem__(self, item):
        return getattr(self, item)