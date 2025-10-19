import os


class Config:
    # LLM api 的配置
    LLM_API_KEY: str = ""
    LLM_MODEL_NAME: str = "deepseek-reasoner"
    LLM_BASE_URL: str = "https://api.deepseek.com/v1"

    # embedding api 的配置
    DASH_EMBEDDINGS_MODEL_NAME: str = 'text-embedding-v1'
    DASHSCOPE_API_KEY: str = ''
    EMBEDDINGS_PATH:str = '.\\embedding'

    # vision llm api 的配置
    VISION_MODEL_API_KEY:str = ''
    VISION_MODEL_NAME:str = 'qwen3-vl-plus'
    VISION_MODEL_BASE_URL:str = 'https://dashscope.aliyuncs.com/compatible-mode/v1'

    # 搜索引擎的密钥
    TAVILY_API_KEY=""

    # redis 存储的配置
    REDIS_HOST = os.environ.get('REDIS_HOST') or 'localhost'  # Redis 服务器地址
    REDIS_PORT = int(os.environ.get('REDIS_PORT') or 6379)  # Redis 服务器端口
    REDIS_DB = int(os.environ.get('REDIS_DB') or 0)  # Redis 数据库索引 (0-15)
    REDIS_PASSWORD = os.environ.get('REDIS_PASSWORD') or None  # Redis 密码 (如果有的话)

    # --- VectorDB (ChromaDB) 配置 ---
    CHROMA_PERSIST_DIR = os.environ.get('CHROMA_PERSIST_DIR') or '.\\chroma_data'

    # 上传文件相关配置
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or '.\\uploads\\temp'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 限制上传文件大小为 16MB
    ALLOWED_FILE_EXTENSIONS = {'md','markdown','pdf', 'txt', 'docx', 'doc', 'pptx', 'ppt', 'c', 'java', 'py'}
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg'}

    DEBUG = True
