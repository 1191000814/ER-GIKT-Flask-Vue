"""
有关问题的蓝图
"""

from flask import Blueprint, request
from ..entity import Question

question_bp = Blueprint('question', __name__, url_prefix='/question')

# 显示全部问题
@question_bp.route('', methods=['GET'])
def show():
    page_index = int(request.args.get('pageIndex'))
    page_size = int(request.args.get('pageSize'))
    search = request.args.get('search')
    if search == '':
        questions = Question.query.all()
    else:
        try:
            search = int(request.args.get('search'))  # 知识点id
            questions = Question.query.filter_by(id=search).all()
        except Exception:
            print('输入格式有误')
            return {
                'msg': '输入格式有误'
            }
    q_dict = [dict(q) for q in questions][(page_index - 1) * page_size : page_index * page_size]
    return {
        'data': q_dict,
        'num': len(questions)
    }