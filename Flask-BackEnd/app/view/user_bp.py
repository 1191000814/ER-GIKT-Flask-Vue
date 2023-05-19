"""
登录认证蓝图
"""
from flask import Blueprint
from flask import request
from icecream import ic

from app.entity import User
from .. import db

user_bp = Blueprint('user', __name__, url_prefix='/user')
# 建议养成习惯, 最后不写'/', 都在前面写'/'

# 登录
@user_bp.route('/login', methods=['POST'])
def login():
    user_dict = request.get_json()
    ic(user_dict)
    user = User.query.filter_by(username=user_dict['username'], password=user_dict['password']).first()
    if user is None: # 登录失败
        return {
            'code': 1
        }
    else: # 登录成功
        return {
            'code': 0,
            'data': dict(user)
        }

# 注册
@user_bp.route('/add', methods=['POST'])
@user_bp.route('/register', methods=['POST'])
def register():
    user_dict = request.get_json()
    ic(user_dict)
    register_user = User.from_dict(user_dict)
    ic(register_user)
    users = User.query.filter_by(username=register_user.username).all()
    if len(users) > 0: # 用户名已经存在
        return {
            'code': 1
        }
    else:
        db.session.add(register_user)
        db.session.commit()
        # 添加新用户
        return {
            'code': 0
        }

# 显示全部用户（学生）
@user_bp.route('', methods=['GET'])
def load():
    page_index = int(request.args.get('pageIndex'))
    page_size = int(request.args.get('pageSize'))
    search = request.args.get('search')
    users = User.query.filter(User.username.like(f'%{search}%')).all()
    # ic(users)
    users_dict = [dict(user) for user in users][(page_index - 1) * page_size : page_index * page_size]
    # 编码为json, 并取相应页面
    return {
        'data': users_dict, # 返回的数据仅仅是当前页面的数据
        'num': len(users) # 返回数量还是全部的数量
    }

# 更新数据
@user_bp.route('update', methods=['POST'])
def update():
    user_dict = request.get_json()
    user = User.query.get(user_dict['id'])
    ic(user)
    for key, val in user_dict.items():
        ic(key, val)
        user.__setattr__(key, val)
    try:
        db.session.commit()
    except Exception:
        return {
            'code': 1,
            'msg': '由于完整性约束，无法更新'
        }
    return {
        'code': 0
    }

@user_bp.route('delete', methods=['POST'])
def delete():
    user_id = request.get_json()['user_id']
    ic(user_id)
    user = User.query.get(user_id)
    ic(user)
    ic(user.history_answers)
    try:
        db.session.delete(user)
        db.session.commit()
    except Exception:
        return {
            'code': 1,
            'msg': '由于完整性约束，无法删除'
        }
    return {
        'code': 0
    }