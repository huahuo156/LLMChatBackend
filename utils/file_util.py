import os

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from werkzeug.utils import secure_filename
from flask import current_app
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, JSONLoader, \
    UnstructuredMarkdownLoader
from langchain_community.document_loaders import (
    UnstructuredWordDocumentLoader,  # 用于 .docx 和 .doc
    UnstructuredPowerPointLoader,  # 用于 .pptx 和 .ppt
    UnstructuredFileLoader  # 通用加载器，可处理 .java, .c 等
)
import easyocr
from models.prompts import GENERATE_SUMMARY_PROMPT,IMAGE_DESC_PROMPT


def allowed_file(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_FILE_EXTENSIONS']


def allowed_image(filename):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in current_app.config['ALLOWED_IMAGE_EXTENSIONS']


def save_temp_file(file_obj):
    """保存上传的文件到临时目录并返回文件路径"""
    filename = secure_filename(file_obj.filename)
    filepath = os.path.join(current_app.config['UPLOAD_FOLDER'], filename)
    os.makedirs(current_app.config['UPLOAD_FOLDER'], exist_ok=True)
    file_obj.save(filepath)
    return filepath


def remove_temp_file(filepath):
    """删除临时文件"""
    try:
        os.remove(filepath)
    except OSError:
        current_app.logger.warning(f"Failed to remove temporary file: {filepath}")
        pass  # 忽略删除失败的情况


def process_file(filepath):
    """
    处理不同类型的文件并返回其文本内容。
    :param filepath: 文件的路径
    :return: 文件的文本内容 (str)
    :raises ValueError: 如果文件类型不支持
    """
    _, file_extension = os.path.splitext(filepath.lower())

    if file_extension == '.txt':
        loader = TextLoader(filepath, encoding='utf-8')
    elif file_extension == '.pdf':
        loader = PyPDFLoader(filepath)
    elif file_extension == '.csv':
        loader = CSVLoader(filepath)
    elif file_extension == '.json':
        loader = JSONLoader(filepath, jq_schema='.', text_content=False)  # 根据JSON结构调整jq_schema
    elif file_extension in ['.md', 'markdown']:
        loader = UnstructuredMarkdownLoader(filepath)
    elif file_extension in ['.docx', '.doc']:
        # 使用 UnstructuredWordDocumentLoader 处理 .docx 和 .doc
        loader = UnstructuredWordDocumentLoader(filepath)
    elif file_extension in ['.pptx', '.ppt']:
        # 使用 UnstructuredPowerPointLoader 处理 .pptx 和 .ppt
        loader = UnstructuredPowerPointLoader(filepath)
    elif file_extension in ['.java', '.c', '.py', '.js', '.html', '.css', '.xml']:  # 可以添加更多代码文件类型
        # 对于代码文件，最简单的方式是作为文本读取
        # 也可以使用 UnstructuredFileLoader
        try:
            # 尝试直接作为文本读取
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except UnicodeDecodeError:
            # 如果编码问题，尝试 UnstructuredFileLoader
            print(f"UTF-8 decode failed for {filepath}, trying UnstructuredFileLoader.")
            try:
                loader = UnstructuredFileLoader(filepath)
            except Exception as e:
                raise ValueError(f"File type '{file_extension}' is not supported or could not be processed. Error: {e}")
    else:
        raise ValueError(f"File type '{file_extension}' is not supported.")

    if 'loader' in locals():
        try:
            documents = loader.load()
            current_app.logger.info(documents)
            # documents 是一个 Document 对象列表，通常取第一个的 page_content
            return "\n".join([doc.page_content for doc in documents])
        except Exception as e:
            raise ValueError(f"Could not load file {filepath} using {type(loader).__name__}: {e}")

    raise RuntimeError(f"Logic error in process_file for {filepath}")


# 使用 OCR 技术尝试识别图片上的文字
def preprocess_image(filepath):
    # 支持简体中文、英文的文字识别
    reader = easyocr.Reader(['ch_sim', 'en'])
    try:
        # OCR 提取
        ocr_results = reader.readtext(filepath, detail=0)  # detail=0 只返回文本
        if ocr_results:
            file_name = os.path.basename(filepath)
            ocr_text = " ".join(ocr_results)
            ocr_text = f"在图片'{file_name}'中识别到的文字如下：\n {ocr_text} \n\n"
        else:
            ocr_text = "未在图片中识别到文字。"

        current_app.logger.info(f"OCR Extracted Text: {ocr_text}")

    except Exception as e:
        current_app.logger.error(f"Error during image preprocessing (OCR): {e}")
        # 即使预处理出错，也提供一个基础信息，避免流程中断
        ocr_text = f"图片预处理失败，错误: {str(e)}。"

    return ocr_text


# 通过 VISION LLM 获得图片的描述以及可能的文字
def get_image_desc(vision_llm, image_base64: str):
    vision_prompt_template = ChatPromptTemplate.from_messages([
        ('system', IMAGE_DESC_PROMPT),
        ("human", [
            {"type": "text", "text": "请详细描述这张图片的内容以及可能包含的文字。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ])
    vision_chain = vision_prompt_template | vision_llm | StrOutputParser()
    image_description = vision_chain.invoke({})
    return image_description


# 建立 生成文章总结的链
def get_generate_summary_chain(llm):
    generate_summary_prompt = ChatPromptTemplate.from_messages([
        ('system', GENERATE_SUMMARY_PROMPT),
        ('human', '{input}')
    ])
    generate_summary_chain = generate_summary_prompt | llm | StrOutputParser()
    return generate_summary_chain
