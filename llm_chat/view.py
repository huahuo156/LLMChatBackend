import base64
import os

from flask import request, jsonify, current_app, send_file
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool

from utils.file_util import allowed_file, save_temp_file, remove_temp_file, process_file, allowed_image, get_image_desc
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from utils.session_storage import session_manager
from langchain.memory import ConversationBufferMemory
from utils.network_utils import fetch_url_content
from utils.audio_utils import dash_text_to_speech, pyttsx_text_to_speech
import datetime


# 纯文本 大模型(大多数情况下用户接入的诸如此类的模型)
def get_llm():
    """获取配置好的 LLM 实例"""
    return ChatOpenAI(
        openai_api_key=current_app.config['LLM_API_KEY'],
        model_name=current_app.config['LLM_MODEL_NAME'],
        base_url=current_app.config['LLM_BASE_URL']
    )


# 具有视觉的模型(用于提取用户上传的图片中的文字以及描述图片内容)
def get_vision_llm():
    """具有识图功能的大模型"""
    return ChatOpenAI(
        openai_api_key=current_app.config['VISION_MODEL_API_KEY'],
        model_name=current_app.config['VISION_MODEL_NAME'],
        base_url=current_app.config['VISION_MODEL_BASE_URL']
    )


# 嵌入模型(用于将用户上传的文件进行向量化，以便进行知识的召回)
def get_embeddings():
    """配置embedding模型"""
    return DashScopeEmbeddings(
        model=current_app.config.get('DASH_EMBEDDINGS_MODEL_NAME'),
        dashscope_api_key=current_app.config.get('DASHSCOPE_API_KEY')
    )


# 对用户上传的文件内容进行一个 长文档 的总结，生成 AI式 摘要，便于 AI 理解与知识召回
def get_generate_summary_chain():
    system_prompt = """
    你是一位高效的文本摘要专家。请仔细阅读用户提供的文档内容，并生成一份简洁、准确且保留核心信息的摘要。
    具体要求如下：
    1.  **长度控制**：摘要应简明扼要，长度大约为原文档的10%-20%，或控制在几句话到一小段之内，具体取决于原文长度，但需确保信息密度高。
    2.  **核心信息**：准确提取并呈现文档的关键信息、主要观点、重要结论或核心论点。确保摘要涵盖文档的主要方面。
    3.  **忠实原文**：摘要内容必须忠实于原文，不得添加原文中未提及的信息、推论或个人解读。
    4.  **语言精炼**：使用清晰、流畅、精炼的语言进行概括，避免冗余和重复。
    5.  **结构清晰**：如果文档内容有多个要点或层次，摘要中也应逻辑清晰地呈现出来。
    请直接输出生成的摘要，无需添加如“摘要如下”或“以下是摘要”之类的前缀。
    """

    generate_summary_prompt = ChatPromptTemplate.from_messages([
        ('system', system_prompt),
        ('human', '{input}')
    ])
    generate_summary_chain = generate_summary_prompt | get_llm() | StrOutputParser()
    return generate_summary_chain


def get_agent(session_id: str, system_prompt: str, session: list):
    # 网络查询工具
    search_internet = TavilySearchResults(max_results=2)

    # 查询向量数据库工具
    @tool
    def query_vectorstore_with_session_id(query: str):
        """
        Query the vectorstore with the given query
        : param query: 必要参数，字符串类型，用于表示要查询向量数据库的具体相关的内容
        : return res: 查询数据库获得的具体内容，可能查询失败，返回“并没有查询到相关的内容”，否则，返回最佳匹配的三个“文档”，并拼接在一起
        """
        return query_vectorstore(query=query, session_id=session_id)

    vector_db_dir = os.path.join(current_app.config.get('EMBEDDINGS_PATH'), session_id)
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H")
    if os.path.exists(vector_db_dir):
        tools = [search_internet, query_vectorstore_with_session_id, fetch_url_content]
        inner_system_prompt = \
            (  # 强调理解当前输入意图和减少调用次数
                    "你是一个高效的AI助手。你的首要任务是准确理解用户**最新输入**({input})的提问意图。"
                    "请仔细分析 {input} 中的核心问题，不要被之前的对话历史干扰。"
                    "在获取足够信息以回答 **{input}** 中的问题之前，不要进行任何工具调用。"
                    "只在必要时调用工具，并且尽量只调用一次。"
                    # 重新定义工具使用策略
                    "如果向量库能直接提供与 **{input}** 问题（非泛化）相关的信息，则优先调用 query_vectorstore_with_session_id。"
                    "如果向量库无相关信息或 **{input}** 问题涉及实时信息（如天气、新闻、股票等），则调用 tavily_search_results_json。"
                    "当用户明确要求访问某个网址时，使用 fetch_url_content 工具。该工具会尝试提取页面的主要文本内容。"
                    "**执行工具以获取信息，而非解释步骤**。"
                    "获取所需信息后，立即组织语言回答 **{input}** 中的问题，确保回答内容与 {input} 直接相关。"
                    "系统提示: 当前日期是 " + str(current_time) + "。请基于此时间回答用户问题。"
            )
    else:
        tools = [search_internet, fetch_url_content]
        inner_system_prompt = \
            (  # 强调理解当前输入意图和减少调用次数
                    "你是一个高效的AI助手。你的首要任务是准确理解用户**最新输入**({input})的提问意图。"
                    "请仔细分析 {input} 中的核心问题，不要被之前的对话历史干扰。"
                    "在获取足够信息以回答 **{input}** 中的问题之前，不要进行任何工具调用。"
                    "只在必要时调用工具，并且尽量只调用一次。"
                    # 重新定义工具使用策略
                    "如果 **{input}** 问题涉及实时信息（如天气、新闻、股票等），则调用 tavily_search_results_json。"
                    "当用户明确要求访问某个网址时，使用 fetch_url_content 工具。该工具会尝试提取页面的主要文本内容。"
                    "**执行工具以获取信息，而非解释步骤**。"
                    "获取所需信息后，立即组织语言回答 **{input}** 中的问题，确保回答内容与 {input} 直接相关。"
                    "系统提示: 当前日期是 " + str(current_time) + "。请基于此时间回答用户问题。"
            )

    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(
            content=system_prompt),
        SystemMessage(
            content=inner_system_prompt),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    current_memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    current_memory.chat_memory.messages = session

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


def chat():
    """普通对话接口"""
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({'error': 'Missing message in request body'}), 400

    if 'session_id' not in data:
        return jsonify({'error': 'Missing session_id in request body'}), 400

    user_message = data['message']
    system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
    session_id = data.get('session_id')  # 用于恢复会话

    # 从 Redis 获取或创建会话
    session = session_manager.get_session_history(session_id)

    session_manager.print_session_history(session_id)

    try:
        agent = get_agent(session_id=session_id, system_prompt=system_prompt, session=session)
        res = agent.invoke({
            'input': user_message,
            "chat_history": session
        })
        ai_response = res.get('output', '')

        # 保存更新后的会话历史到 Redis
        session_manager.set_session_history(session_id, session)

        return jsonify({
            'response': ai_response,
            'session_id': session_id  # 返回给前端，以便后续请求使用
        })

    except Exception as e:
        current_app.logger.error(f"Error in chat: {e}")
        return jsonify({'error': 'Failed to process chat request'}), 500


def chat_with_image():
    """带图片的对话接口"""
    if 'image' not in request.files or 'message' not in request.form:
        return jsonify({'error': 'Missing image file or message text'}), 400

    if 'session_id' not in request.form:
        return jsonify({'error', 'Missing session_id in request body'}), 400

    image_file = request.files['image']
    user_message = request.form['message']
    session_id = request.form['session_id']
    system_prompt = request.form.get('system_prompt', 'You are a helpful assistant.')

    if image_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_image(image_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filepath = save_temp_file(image_file)

    # 从 Redis 获取或创建会话
    session = session_manager.get_session_history(session_id)

    try:
        # 读取图片并编码为 base64
        with open(filepath, "rb") as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')

        # 对用户上传的图片进行提取文字和描述的预处理，并将其加入至上下文中
        image_description = ("本轮对话中用户提及一张图片，关于这张图片的描述如下所示，包括但不限于图片中的文字：\n\n"
                             + get_image_desc(get_vision_llm(), img_data))

        system_prompt = system_prompt + image_description

        agent = get_agent(session_id, system_prompt, session)
        res = agent.invoke({
            "input": user_message
        })
        ai_response = res.get("output", "")

        # 将本次对话记录添加到会话历史中
        session.append(AIMessage(content=ai_response))

        session_manager.set_session_history(session_id, session)

        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })

    except Exception as e:
        current_app.logger.error(f"Error in chat_with_image: {e}")
        return jsonify({'error': 'Failed to process chat with image request'}), 500
    finally:
        # 删除临时保存的图片文件
        remove_temp_file(filepath)


def chat_with_file():
    """带文件的对话接口"""
    if 'file' not in request.files or 'message' not in request.form:
        return jsonify({'error': 'Missing file or message text'}), 400

    if 'session_id' not in request.form:
        return jsonify({'error', 'Missing session_id in request body'}), 400

    uploaded_file = request.files['file']
    user_message = request.form['message']
    session_id = request.form['session_id']
    system_prompt = request.form.get('system_prompt', 'You are a helpful assistant.')

    if uploaded_file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if not allowed_file(uploaded_file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filepath = save_temp_file(uploaded_file)

    # 从 Redis 获取或创建会话
    session = session_manager.get_session_history(session_id)

    try:
        file_content = process_file(filepath)
        filename = uploaded_file.filename

        generate_embeddings(filename, file_content, session_id)

        agent = get_agent(session_id, system_prompt, session)
        res = agent.invoke({'input': user_message})
        ai_response = res.get('output', '')

        # 将本次对话记录添加到会话历史中
        session_manager.set_session_history(session_id, session)

        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })

    except ValueError as e:  # 捕获未实现的文件类型错误
        return jsonify({'error': str(e)}), 501
    except Exception as e:
        current_app.logger.error(f"Error in chat_with_file: {e}")
        return jsonify({'error': 'Failed to process chat with file request'}), 500
    finally:
        # 删除临时保存的文件
        remove_temp_file(filepath)


def clear_current_chat_history():
    """清除本轮对话的历史(缓存)"""
    data = request.get_json()
    if not data or 'session_id' not in data:
        return jsonify({'error': 'Missing session_id in request body'}), 400

    session_id = data['session_id']

    # 使用 session_manager 清除会话历史
    session_manager.clear_session_history(session_id)

    return jsonify(
        {'message': f'Chat history and associated vector data for session {session_id} cleared successfully.'}), 200


def text_to_speech():
    """将指定的文本转化为语音"""
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': "Missing text in request body"}), 400

    text = data['text']
    dashscope_api_key = current_app.config.get('DASHSCOPE_API_KEY', None)
    if not dashscope_api_key:
        # 配置中没有 dashscope api key
        audio_file_path = pyttsx_text_to_speech(text=text)
    else:
        # 配置中有 dashscope api key
        audio_file_path = dash_text_to_speech(
            text=text,
            dashscope_api_key=dashscope_api_key
        )

    if not audio_file_path:
        return jsonify({"error": "Can't convert the text to the speech,please try again"}), 400
    else:
        return send_file(audio_file_path, as_attachment=True)


def generate_embeddings(file_name: str, file_content: str, session_id: str):
    """
    :param file_name: 是临时存储上传文件的文件名称
    :param file_content: 是临时存储的上传文件的文件内容
    :param session_id: 是本轮对话的会话 ID
    :return: none
    """
    persist_dir = os.path.join(current_app.config.get('EMBEDDINGS_PATH'), session_id)
    os.makedirs(persist_dir, exist_ok=True)

    if file_content and file_name:

        chain = get_generate_summary_chain()
        summary = chain.invoke({'input': file_content})

        file_content = '本文的摘要\主要内容是：\n\n' + summary + "\n\n" + file_content

        documents = [Document(
            page_content=file_content,
            metadata={'file_name': file_name, 'session_id': session_id}
        )]

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_docs = text_splitter.split_documents(documents)

        if os.path.exists(persist_dir):
            dabs = Chroma(
                persist_directory=persist_dir,
                embedding_function=get_embeddings()
            )
            dabs.add_documents(split_docs)
        else:
            dabs = Chroma.from_documents(
                documents=split_docs,
                embedding=get_embeddings(),
                persist_directory=persist_dir
            )

        dabs.persist()
        current_app.logger.info(f'成功将 {file_name} 加载至向量数据库中')


def query_vectorstore(query: str, session_id: str):
    """
    检索向量数据库函数
    : param query: 必要参数，字符串类型，用于表示要查询向量数据库的具体相关的内容
    : param session_id: 必要参数，但在创建agent之前已绑定该形参，无需理会
    : return res: 查询数据库获得的具体内容，可能查询失败，返回“并没有查询到相关的内容”，否则，返回最佳匹配的三个“文档”，并拼接在一起
    """
    persist_dir = os.path.join(current_app.config.get('EMBEDDINGS_PATH'), session_id)

    if not os.path.exists(persist_dir):
        return "未发现向量数据库"

    vector_db = Chroma(
        persist_directory=persist_dir,
        embedding_function=get_embeddings()
    )
    results = vector_db.similarity_search(query, k=3)

    if results:
        res = [doc.page_content for doc in results]
        res = '\n'.join(res)
    else:
        res = "没有查询到相关内容"

    return res


def health_check():
    """健康检查接口"""
    # 从各自的管理器检查依赖健康状况
    redis_status = "healthy" if session_manager.ping() else "unhealthy"
    return {
        'status': 'healthy',
        'dependencies': {
            'redis': redis_status,
        }
    }, 200
