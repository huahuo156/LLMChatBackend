from flask import Blueprint
from .view import *

chat_bp = Blueprint('chat_bp', __name__,url_prefix='/llm_chat')

# 普通对话
chat_bp.add_url_rule(rule='/chat', methods=['POST'], view_func=chat)

# 带有图片的对话
chat_bp.add_url_rule(rule='/chat_with_image', methods=['POST'], view_func=chat_with_image)

# 带有文件的对话
chat_bp.add_url_rule(rule='/chat_with_file', methods=['POST'], view_func=chat_with_file)

# 健康检查接口
chat_bp.add_url_rule(rule='/health',methods=['GET'],view_func=health_check)

