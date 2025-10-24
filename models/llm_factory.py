# models/llm_factory.py
import os

from flask import current_app
from langchain_openai import ChatOpenAI
from langchain_community.embeddings import DashScopeEmbeddings


def get_llm():
    """获取配置好的 LLM 实例"""
    return ChatOpenAI(
        openai_api_key=os.getenv('LLM_API_KEY'),
        model_name=current_app.config['LLM_MODEL_NAME'],
        base_url=current_app.config['LLM_BASE_URL']
    )


def get_vision_llm():
    """具有识图功能的大模型"""
    return ChatOpenAI(
        openai_api_key=os.getenv('VISION_MODEL_API_KEY'),
        model_name=current_app.config['VISION_MODEL_NAME'],
        base_url=current_app.config['VISION_MODEL_BASE_URL']
    )


def get_embeddings():
    """配置embedding模型"""
    return DashScopeEmbeddings(
        model=current_app.config.get('DASH_EMBEDDINGS_MODEL_NAME'),
        dashscope_api_key=os.getenv('DASHSCOPE_API_KEY')
    )
