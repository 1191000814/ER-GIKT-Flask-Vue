"""
启动文件
"""

from . import create_app

if __name__ == '__main__':
    app = create_app()
    app.run()