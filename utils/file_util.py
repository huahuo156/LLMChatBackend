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

    # 如果上面没有定义 loader (例如，代码文件直接读取的情况)，则函数应已返回内容
    # 此行理论上不应执行到
    raise RuntimeError(f"Logic error in process_file for {filepath}")


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


def get_image_desc(vision_llm,image_base64: str):
    system_prompt = """
    你是一位专业的图像内容描述专家。你的任务是接收一张图片，并生成一段清晰、准确、全面且客观的描述。
    具体要求如下：
    1.  **内容全面**：描述图片中包含的所有主要对象、场景、人物、活动、背景元素等。
    2.  **细节清晰**：尽可能地捕捉并描述重要的视觉细节，例如物体的颜色、形状、大小、纹理、位置关系，人物的衣着、表情、姿态，环境的光线、氛围等。
    3.  **逻辑连贯**：组织语言，使描述流畅、有条理，能够清晰地呈现图片的整体画面和各元素之间的关系。
    4.  **客观准确**：基于图片实际内容进行描述，避免添加主观臆断、猜测或图片中未明确显示的信息。
    5.  **语言精炼**：使用自然、流畅、精炼的语言，避免冗余和模糊不清的表述。
    6.  **聚焦核心**：如果图片内容复杂，优先描述最核心、最突出的部分。
    请直接输出对图片的详细描述，无需添加如“这张图片显示了”或“我看到”之类的前缀。
    """
    vision_prompt_template = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
        ("human", [
            {"type": "text", "text": "请详细描述这张图片的内容以及可能包含的文字。"},
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
        ])
    ])
    vision_chain = vision_prompt_template | vision_llm | StrOutputParser()
    image_description = vision_chain.invoke({})
    return image_description
