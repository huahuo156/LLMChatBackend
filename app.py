from flask import Flask
from dotenv import load_dotenv, find_dotenv

# 将 .env 文件中对应的键值对，加载至环境变量中(仅在本次运行过程有效)
_ = load_dotenv(find_dotenv())

from Config import Config
from llm_chat import chat_bp



def create_app():

    app = Flask(__name__)
    app.config.from_object(Config)

    # 注册 Blueprint
    app.register_blueprint(chat_bp)

    return app


if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
