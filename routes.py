# routes.py
from flask import Blueprint, request, jsonify, send_file, current_app

from models.vector_db_manager import VectorDBManager
from services.chat_service import ChatService
from services.audio_service import AudioService
from utils.session_storage import session_manager

# 创建蓝图
main_bp = Blueprint('main', __name__)


def get_services():
    """在当前应用上下文内获取服务实例"""
    # 从 current_app.config 获取配置
    embeddings_path = current_app.config.get('EMBEDDINGS_PATH')
    # 创建 VectorDBManager 实例
    vector_db_manager = VectorDBManager(embeddings_path)
    # 创建 ChatService 实例，注入依赖
    chat_service = ChatService(session_manager, vector_db_manager)
    # 创建 AudioService 实例
    audio_service = AudioService()

    return chat_service, audio_service


@main_bp.route('/chat', methods=['POST'])
def chat():
    """普通对话接口"""
    chat_service, _ = get_services()  # 获取需要的服务
    try:
        data = request.get_json()
        if not data or 'message' not in data or 'session_id' not in data:
            return jsonify({'error': 'Missing message or session_id in request body'}), 400

        user_message = data['message']
        system_prompt = data.get('system_prompt', 'You are a helpful assistant.')
        session_id = data['session_id']

        ai_response = chat_service.handle_chat(user_message, system_prompt, session_id)

        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    except Exception as e:
        current_app.logger.error(f"Error in chat: {e}")
        return jsonify({'error': 'Failed to process chat request'}), 500


@main_bp.route('/chat_with_image', methods=['POST'])
def chat_with_image():
    """带图片的对话接口"""
    chat_service, _ = get_services()  # 获取需要的服务
    try:
        if 'image' not in request.files or 'message' not in request.form or 'session_id' not in request.form:
            return jsonify({'error': 'Missing image file, message text, or session_id'}), 400

        image_file = request.files['image']
        user_message = request.form['message']
        session_id = request.form['session_id']
        system_prompt = request.form.get('system_prompt', 'You are a helpful assistant.')

        if image_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        ai_response = chat_service.handle_chat_with_image(image_file, user_message, system_prompt, session_id)

        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    except Exception as e:
        current_app.logger.error(f"Error in chat_with_image: {e}")
        return jsonify({'error': 'Failed to process chat with image request'}), 500


@main_bp.route('/chat_with_file', methods=['POST'])
def chat_with_file():
    """带文件的对话接口"""
    chat_service, _ = get_services()  # 获取需要的服务
    try:
        if 'file' not in request.files or 'message' not in request.form or 'session_id' not in request.form:
            return jsonify({'error': 'Missing file, message text, or session_id'}), 400

        uploaded_file = request.files['file']
        user_message = request.form['message']
        session_id = request.form['session_id']
        system_prompt = request.form.get('system_prompt', 'You are a helpful assistant.')

        if uploaded_file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        ai_response = chat_service.handle_chat_with_file(uploaded_file, user_message, system_prompt, session_id)

        return jsonify({
            'response': ai_response,
            'session_id': session_id
        })
    except ValueError as e:
        return jsonify({'error': str(e)}), 501
    except Exception as e:
        current_app.logger.error(f"Error in chat_with_file: {e}")
        return jsonify({'error': 'Failed to process chat with file request'}), 500


@main_bp.route('/clear_current_chat_history', methods=['POST'])
def clear_current_chat_history():
    """清除本轮对话的历史(缓存)"""
    chat_service, _ = get_services()  # 获取需要的服务
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return jsonify({'error': 'Missing session_id in request body'}), 400

        session_id = data['session_id']
        chat_service.clear_session_history(session_id)

        return jsonify(
            {'message': f'Chat history and associated vector data for session {session_id} cleared successfully.'}), 200
    except Exception as e:
        current_app.logger.error(f"Error in clear_current_chat_history: {e}")
        return jsonify({'error': 'Failed to clear chat history'}), 500


@main_bp.route('/text_to_speech', methods=['POST'])
def text_to_speech():
    """将指定的文本转化为语音"""
    _, audio_service = get_services()  # 获取需要的服务
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': "Missing text in request body"}), 400

        text = data['text']
        audio_file_path = audio_service.convert_text_to_speech(text)

        if not audio_file_path:
            return jsonify({"error": "Can't convert the text to the speech, please try again"}), 400
        else:
            return send_file(audio_file_path, as_attachment=True)
    except Exception as e:
        current_app.logger.error(f"Error in text_to_speech: {e}")
        return jsonify({'error': 'Failed to process text to speech request'}), 500


@main_bp.route('/health', methods=['GET'])
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


def register_routes(app):
    """注册所有路由到应用"""
    app.register_blueprint(main_bp, url_prefix='/api/v1')  # 可以添加版本前缀
