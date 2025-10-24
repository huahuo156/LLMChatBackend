# models/vector_db_manager.py
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
from models.llm_factory import get_embeddings, get_llm
from utils.file_util import get_generate_summary_chain


class VectorDBManager:
    # 接收 embeddings_path 作为参数，而不是在 __init__ 时从 current_app 获取
    def __init__(self, embeddings_path):
        self.embeddings_path = embeddings_path

    def get_embeddings_path(self):
        return self.embeddings_path

    def generate_embeddings(self, file_name: str, file_content: str, session_id: str):
        persist_dir = os.path.join(self.embeddings_path, session_id)
        os.makedirs(persist_dir, exist_ok=True)

        if file_content and file_name:

            chain = get_generate_summary_chain(get_llm())
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
                    embedding_function=get_embeddings()  # get_embeddings 需在有 app_context 时调用
                )
                dabs.add_documents(split_docs)
            else:
                dabs = Chroma.from_documents(
                    documents=split_docs,
                    embedding=get_embeddings(),  # get_embeddings 需在有 app_context 时调用
                    persist_directory=persist_dir
                )

            dabs.persist()
            # 记录日志时使用 current_app
            from flask import current_app
            current_app.logger.info(f'成功将 {file_name} 加载至向量数据库中')

    def query_vectorstore(self, query: str, session_id: str):
        persist_dir = os.path.join(self.embeddings_path, session_id)

        if not os.path.exists(persist_dir):
            return "未发现向量数据库"

        # get_embeddings 需在有 app_context 时调用
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

    def clear_vector_db(self, session_id: str):
        persist_dir = os.path.join(self.embeddings_path, session_id)
        if os.path.exists(persist_dir):
            import shutil
            shutil.rmtree(persist_dir)
            # 记录日志时使用 current_app
            from flask import current_app
            current_app.logger.info(f'向量数据库目录 {persist_dir} 已删除')
