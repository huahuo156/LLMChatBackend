# services/chat_service.py
import base64
import datetime
import os
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.messages import AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain.memory import ConversationBufferMemory
from models.vector_db_manager import VectorDBManager
from models.llm_factory import get_llm, get_vision_llm
from utils.file_util import allowed_file, allowed_image, save_temp_file, remove_temp_file, process_file, get_image_desc
from utils.web_utils import web_search, crawl_url_content,fetch_url_content
from utils.session_storage import RedisSessionManager
from models.prompts import AGENT_SYSTEM_PROMPT


class ChatService:
    def __init__(self, session_manager: RedisSessionManager, vector_db_manager: VectorDBManager):
        self.session_manager = session_manager
        self.vector_db_manager = vector_db_manager
        self.default_tools = [web_search, crawl_url_content,fetch_url_content]

    def handle_chat(self, user_message, user_system_prompt, session_id):
        # 获取历史对话
        session = self.session_manager.get_session_history(session_id)
        self.session_manager.print_session_history(session_id)
        # _get_agent 需要访问 self.vector_db_manager
        # 构建智能体
        agent = self._get_agent(session_id, user_system_prompt, session)
        # 调用智能体
        res = agent.invoke({'input': user_message, "chat_history": session})
        # 更新历史对话
        ai_response = res.get('output', '')
        final_session_messages = agent.memory.chat_memory.messages
        self.session_manager.set_session_history(session_id, final_session_messages)
        self.session_manager.sync_session_to_mysql(session_id)

        return ai_response

    def handle_chat_with_image(self, image_file, user_message, user_system_prompt, session_id):
        if not allowed_image(image_file.filename):
            raise ValueError('File type not allowed')

        filepath = save_temp_file(image_file)
        try:
            # 读取图片并编码为 base64
            with open(filepath, "rb") as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')

            # 对用户上传的图片进行提取文字和描述的预处理，并将其加入至上下文中
            image_description = ("本轮对话中提及一张图片，关于这张图片的描述如下所示，包括但不限于图片中的文字：\n\n"
                                 + get_image_desc(get_vision_llm(), img_data))

            user_system_prompt = image_description + "\n\n" + user_system_prompt

            self.session_manager.print_session_history(session_id)
            session = self.session_manager.get_session_history(session_id)
            agent = self._get_agent(session_id, user_system_prompt, session)
            res = agent.invoke({"input": user_message})
            ai_response = res.get("output", "")

            # 将本次对话记录添加到会话历史中
            session.append(AIMessage(content=ai_response))
            self.session_manager.set_session_history(session_id, session)
            self.session_manager.sync_session_to_mysql(session_id)

            return ai_response
        finally:
            remove_temp_file(filepath)

    def handle_chat_with_file(self, uploaded_file, user_message, user_system_prompt, session_id):
        if not allowed_file(uploaded_file.filename):
            raise ValueError('File type not allowed')

        filepath = save_temp_file(uploaded_file)
        try:
            file_content = process_file(filepath)
            filename = uploaded_file.filename

            # 生成向量数据库
            self.vector_db_manager.generate_embeddings(filename, file_content, session_id)

            self.session_manager.print_session_history(session_id)
            session = self.session_manager.get_session_history(session_id)
            agent = self._get_agent(session_id, user_system_prompt, session)
            res = agent.invoke({'input': user_message})
            ai_response = res.get('output', '')

            # 保存更新后的会话历史到 Redis
            final_session_messages = agent.memory.chat_memory.messages
            self.session_manager.set_session_history(session_id, final_session_messages)
            self.session_manager.sync_session_to_mysql(session_id)

            return ai_response
        finally:
            remove_temp_file(filepath)

    def clear_session_history(self, session_id):
        self.session_manager.clear_session_history(session_id)
        # 同时清理相关的向量数据库
        self.vector_db_manager.clear_vector_db(session_id)

    def _get_agent(self, session_id: str, user_system_prompt: str, session: list):

        # 查询向量数据库工具
        @tool
        def query_vectorstore_with_session_id(query: str):
            """
            Query the vectorstore with the given query
            : param query: 必要参数，字符串类型，用于表示要查询向量数据库的具体相关的内容
            : return res: 查询数据库获得的具体内容，可能查询失败，返回“并没有查询到相关的内容”，否则，返回最佳匹配的三个“文档”，并拼接在一起
            """
            return self.vector_db_manager.query_vectorstore(query=query, session_id=session_id)

        vector_db_dir = os.path.join(self.vector_db_manager.get_embeddings_path(), session_id)
        # 可使用的工具的配置
        tools = list(self.default_tools)
        if os.path.exists(vector_db_dir):
            tools = tools.append(query_vectorstore_with_session_id)

        # 记忆的配置
        current_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        current_memory.chat_memory.messages = session

        # 获取当前系统时间
        current_time = datetime.datetime.now().strftime("%Y-%m-%d %H")

        # 提示词工程
        prompt = ChatPromptTemplate.from_messages([
            ("system", user_system_prompt),
            ("system", AGENT_SYSTEM_PROMPT.format(current_time=current_time)),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ])

        agent = create_tool_calling_agent(get_llm(), tools, prompt)
        return AgentExecutor(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=current_memory,
            handle_parsing_errors=True,
            max_iterations=5,
            max_execution_time=30,
        )
