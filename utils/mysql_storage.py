import json
import logging
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from flask import current_app
import pymysql.cursors
import os


class MySQLSessionManager:
    def __init__(self):
        # 从 Flask 应用配置中获取 MySQL 连接信息
        self.host = os.getenv('MYSQL_HOST', 'localhost')
        self.port = int(os.getenv('MYSQL_PORT', 3306))
        self.user = os.getenv('MYSQL_USER', 'root')
        self.password = os.getenv('MYSQL_PASSWORD', '')
        self.database = os.getenv('MYSQL_DATABASE', 'chat_app')
        self.charset = os.getenv('MYSQL_CHARSET', 'utf8mb4')
        self.timezone = os.getenv('MYSQL_TIMEZONE', '+08:00')

    def _get_connection(self):
        """获取 MySQL 数据库连接"""
        try:
            connection = pymysql.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database,
                charset=self.charset,
                cursorclass=pymysql.cursors.DictCursor  # 返回字典格式的结果，方便处理
            )
            print("Connected to MySQL database")
            return connection
        except Exception as e:
            current_app.logger.error(f"Error connecting to MySQL: {e}")
            raise e

    def _create_table_if_not_exists(self):
        """检查并创建会话历史表（如果不存在）"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS chat_sessions (
            id INT AUTO_INCREMENT PRIMARY KEY,
            session_id VARCHAR(255) NOT NULL UNIQUE,
            history JSON NOT NULL, -- 使用 JSON 类型存储消息列表
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
            INDEX idx_session_id (session_id) -- 为 session_id 创建索引，提高查询速度
        ) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_unicode_ci;
        """
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                cursor.execute(create_table_sql)
            connection.commit()
            print("Table 'chat_sessions' is ready.")
        except Exception as e:
            current_app.logger.error(f"Error creating table: {e}")
            raise e
        finally:
            connection.close()

    def get_session_history(self, session_id: str, default=None):
        """从 MySQL 获取会话历史"""
        if default is None:
            default = []

        # 确保表存在
        self._create_table_if_not_exists()

        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "SELECT history FROM chat_sessions WHERE session_id = %s"
                cursor.execute(sql, (session_id,))
                result = cursor.fetchone()

                if result and result['history']:
                    history_json = result['history']
                    history = []
                    for msg_obj in history_json:
                        if msg_obj['type'] == 'human':
                            history.append(HumanMessage(content=msg_obj['content']))
                        elif msg_obj['type'] == 'ai':
                            history.append(AIMessage(content=msg_obj['content']))
                        elif msg_obj['type'] == 'system':
                            history.append(SystemMessage(content=msg_obj['content']))
                    return history
                else:
                    return default
        except Exception as e:
            current_app.logger.error(f"Error loading session history for {session_id}: {e}")
            return default
        finally:
            connection.close()

    def set_session_history(self, session_id: str, history: list):
        """将会话历史保存到 MySQL (持久化)"""
        # 确保表存在
        self._create_table_if_not_exists()

        # 将 LangChain 消息对象转换为 JSON 序列化的格式
        history_json = []
        for msg in history:
            if isinstance(msg, HumanMessage):
                history_json.append({'type': 'human', 'content': msg.content})
            elif isinstance(msg, AIMessage):
                history_json.append({'type': 'ai', 'content': msg.content})
            elif isinstance(msg, SystemMessage):
                history_json.append({'type': 'system', 'content': msg.content})

        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                # 使用 INSERT ... ON DUPLICATE KEY UPDATE 来实现 "upsert"
                # 如果 session_id 不存在则插入，存在则更新
                sql = """
                INSERT INTO chat_sessions (session_id, history) 
                VALUES (%s, %s) 
                ON DUPLICATE KEY UPDATE history = VALUES(history), updated_at = CURRENT_TIMESTAMP
                """
                cursor.execute(sql, (
                    session_id, json.dumps(history_json, ensure_ascii=False)))  # ensure_ascii=False 以支持中文等非ASCII字符
            connection.commit()
        except Exception as e:
            current_app.logger.error(f"Error saving session history for {session_id}: {e}")
            # raise e # 根据需要决定是否抛出异常
        finally:
            connection.close()

    def clear_session_history(self, session_id: str):
        """从 MySQL 清除指定会话的历史"""
        connection = self._get_connection()
        try:
            with connection.cursor() as cursor:
                sql = "DELETE FROM chat_sessions WHERE session_id = %s"
                cursor.execute(sql, (session_id,))
            connection.commit()

            # 记录日志
            if cursor.rowcount > 0:
                current_app.logger.info(f"Session {session_id} cleared from MySQL.")
            else:
                current_app.logger.debug(f"Attempted to clear session {session_id}, but it did not exist in MySQL.")
        except Exception as e:
            current_app.logger.error(f"Error clearing session history for {session_id}: {e}")
            # raise e # 根据需要决定是否抛出异常
        finally:
            connection.close()

    def ping(self):
        """测试 MySQL 连接"""
        try:
            connection = self._get_connection()
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            connection.close()
            return True
        except Exception as e:
            logging.error(f"MySQL health check failed: {e}")
            return False


# 创建一个全局实例，以便在其他模块中使用
session_manager = MySQLSessionManager()
