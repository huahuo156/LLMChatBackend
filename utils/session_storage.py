import json
import logging
import redis
from flask import current_app
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from .mysql_storage import session_manager as mysql_session_manager


class RedisSessionManager:
    def __init__(self):
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
        """
        从 Redis 获取会话历史。
        如果 Redis 中没有，则尝试从 MySQL 加载并存入 Redis，然后返回。
        """
        if default is None:
            default = []
        redis_client = self._get_redis_client()
        key = f"chat_session:{session_id}"

        session_data = redis_client.get(key)
        if session_data:
            # Redis 中有数据，直接返回
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
                logging.info(f"Retrieved session {session_id} from Redis.")
                return history
            except (json.JSONDecodeError, KeyError, TypeError) as e:
                current_app.logger.error(f"Error loading session history for {session_id} from Redis: {e}")
                # 如果 Redis 数据损坏，尝试从 MySQL 加载
                return self._load_from_mysql_and_cache(session_id, default)
        else:
            # Redis 中没有数据，尝试从 MySQL 加载
            logging.info(f"Session {session_id} not found in Redis, attempting to load from MySQL.")
            return self._load_from_mysql_and_cache(session_id, default)

    def _load_from_mysql_and_cache(self, session_id: str, default):
        """从 MySQL 加载会话历史并缓存到 Redis"""
        try:
            history_from_mysql = mysql_session_manager.get_session_history(session_id, default)
            # 将从 MySQL 加载的数据存入 Redis，供后续快速访问
            self.set_session_history(session_id, history_from_mysql)
            logging.info(f"Loaded session {session_id} from MySQL and cached in Redis.")
            return history_from_mysql
        except Exception as e:
            current_app.logger.error(f"Error loading session {session_id} from MySQL: {e}")
            return default

    def sync_session_to_mysql(self, session_id: str):
        """
        从 Redis 获取会话历史并同步到 MySQL。
        这个方法需要在对话结束时被调用。
        """
        # 从 Redis 获取最新的会话历史
        latest_history_from_redis = self.get_session_history(session_id, default=[])

        try:
            # 将最新的历史保存到 MySQL
            mysql_session_manager.set_session_history(session_id, latest_history_from_redis)
            logging.info(f"Synced session {session_id} from Redis to MySQL.")
        except Exception as e:
            current_app.logger.error(f"Error syncing session {session_id} to MySQL: {e}")


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
                    try:
                        original_content.encode('gbk')
                        safe_content = original_content
                    except UnicodeEncodeError:
                        safe_content = original_content.encode('unicode_escape').decode('ascii')  # 转义

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
