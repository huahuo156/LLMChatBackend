import json
import logging
import redis
from flask import current_app
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage


class RedisSessionManager:
    def __init__(self):
        # 在初始化时或每次需要连接时获取客户端
        # self.redis_client = self._get_redis_client()
        pass

    def _get_redis_client(self):
        """获取 Redis 客户端实例"""
        # 从 Flask 应用配置中获取 Redis 连接信息
        redis_host = current_app.config.get('REDIS_HOST', 'localhost')
        redis_port = current_app.config.get('REDIS_PORT', 6379)
        redis_db = current_app.config.get('REDIS_DB', 0)
        redis_password = current_app.config.get('REDIS_PASSWORD', None)
        redis_url = current_app.config.get('REDIS_URL', None)

        if redis_url:
            pool = redis.ConnectionPool.from_url(redis_url)
            r = redis.Redis(connection_pool=pool)
        else:
            pool = redis.ConnectionPool(host=redis_host, port=redis_port, db=redis_db, password=redis_password)
            r = redis.Redis(connection_pool=pool)

        # 测试连接
        try:
            r.ping()
        except redis.ConnectionError as e:
            current_app.logger.error(f"Could not connect to Redis: {e}")
            raise e

        return r

    def get_session_history(self, session_id: str, default=None):
        """从 Redis 获取会话历史"""
        if default is None:
            default = []
        redis_client = self._get_redis_client()
        key = f"chat_session:{session_id}"

        session_data = redis_client.get(key)
        if session_data:
            try:
                history_json = json.loads(session_data)
                history = []
                for msg_obj in history_json:
                    if msg_obj['type'] == 'human':
                        content = msg_obj['content']
                        history.append(HumanMessage(content=content))
                    elif msg_obj['type'] == 'ai':
                        history.append(AIMessage(content=msg_obj['content']))
                    elif msg_obj['type'] == 'system':
                        history.append(SystemMessage(content=msg_obj['content']))
                return history
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                current_app.logger.error(f"Error loading session history for {session_id}: {e}")
                return default
        else:
            return default

    def set_session_history(self, session_id: str, history: list, expire_time=3600):
        """将会话历史保存到 Redis"""
        redis_client = self._get_redis_client()
        key = f"chat_session:{session_id}"

        history_json = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_json.append({'type': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                history_json.append({'type': 'ai', 'content': msg.content})
            elif isinstance(msg, SystemMessage):
                history_json.append({'type': 'system', 'content': msg.content})

        try:
            redis_client.setex(key, expire_time, json.dumps(history_json))
        except Exception as e:
            current_app.logger.error(f"Error saving session history for {session_id}: {e}")
            # 可以选择抛出异常或静默失败，取决于你的需求
            # raise e

    def clear_session_history(self, session_id: str):
        """从 Redis 清除指定会话的历史"""
        redis_client = self._get_redis_client()
        key = f"chat_session:{session_id}"

        # Redis 的 delete 命令即使键不存在也不会报错
        # 它会返回删除的键的数量 (0 或 1)
        deleted_count = redis_client.delete(key)

        # 可以选择性地记录日志，区分是否真的删除了数据
        if deleted_count > 0:
            current_app.logger.info(f"Session {session_id} cleared from Redis.")
        else:
            current_app.logger.debug(
                f"Attempted to clear session {session_id}, but it did not exist in Redis.")  # 使用 debug 级别，避免日志过多

    def print_session_history(self, session_id: str):
        """打印指定会话的所有对话历史，处理可能的编码问题"""
        redis_client = self._get_redis_client()
        key = f"chat_session:{session_id}"

        session_data = redis_client.get(key)
        if session_data:
            try:
                history_json = json.loads(session_data)
                print(f"\n--- Session History for ID: {session_id} ---")
                for i, msg_obj in enumerate(history_json):
                    msg_type = msg_obj['type'].upper()
                    original_content = msg_obj['content']

                    # 尝试编码内容以检查是否能被当前环境的默认编码处理
                    # 如果不能，则进行转义或替换处理
                    try:
                        # 尝试用系统默认编码（通常是gbk在Windows控制台）编码内容
                        # 这会触发错误，如果内容包含无法编码的字符
                        original_content.encode('gbk')
                        # 如果成功，直接使用原始内容
                        safe_content = original_content
                    except UnicodeEncodeError:
                        # 如果失败，使用 'unicode_escape' 编码进行转义
                        # 这会将无法编码的字符显示为 \Uxxxx 的形式
                        # 或者使用 'replace' 或 'ignore' 错误处理
                        # safe_content = original_content.encode('gbk', errors='replace').decode('gbk') # 用 ? 替换
                        # safe_content = original_content.encode('gbk', errors='ignore').decode('gbk') # 忽略
                        safe_content = original_content.encode('unicode_escape').decode('ascii')  # 转义
                        # 或者使用 'latin-1' 作为中间编码（虽然不完美），但这通常不可行且可能导致乱码
                        # 更安全的方式是使用 'unicode_escape' 或 'replace'

                    print(f"Round {i + 1} - {msg_type}: {safe_content}")
                print("--- End of Session History ---\n")
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                current_app.logger.error(f"Error printing session history for {session_id}: {e}")
                print(f"Error reading session {session_id} from Redis: {e}")
        else:
            print(f"No history found for session ID: {session_id}")

    def ping(self):
        """测试 Redis 连接"""
        try:
            r = self._get_redis_client()
            r.ping()
            return True
        except Exception as e:
            logging.error(f"Redis health check failed: {e}")
            return False

# 创建一个全局实例，以便在其他模块中使用
session_manager = RedisSessionManager()