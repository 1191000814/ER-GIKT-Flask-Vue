"""
关于技能的蓝图
"""
from flask import Blueprint, request
from ..entity import Skill

skill_bp = Blueprint('skill', __name__, url_prefix='/skill')

# 显示全部技能
@skill_bp.route('')
def show():
    page_index = int(request.args.get('pageIndex'))
    page_size = int(request.args.get('pageSize'))
    search = request.args.get('search')
    if search == '':
        skills = Skill.query.all()
    else:
        try:
            search = int(request.args.get('search')) # 知识点id
            skills = Skill.query.filter_by(id=search).all()
        except Exception:
            print('输入格式有误')
            return {
                'msg': '输入格式有误'
            }
    skills_dict = [dict(skill) for skill in skills][(page_index - 1) * page_size : page_index * page_size]
    return {
        'data': skills_dict,
        'num': len(skills)
    }