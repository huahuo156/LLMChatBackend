import uuid
import pyttsx3
import dashscope
from dashscope.audio.tts import SpeechSynthesizer
from flask import current_app
import os


def pyttsx_text_to_speech(text: str, filename_hint: str = None) -> str:
    """
    Converts text to speech using pyttsx3 (local system TTS engine) and saves the audio file.
    This function can be called from other routes.

    Args:
        text (str): The text to convert to speech.
        filename_hint (str, optional): A hint to use in the generated filename. Defaults to None.

    Returns:
        str: The filename of the generated audio file, or None if an error occurs.
    """
    if not text or not text.strip():
        current_app.logger.error("text_to_speech_util: Input text is empty.")
        return None

    try:
        # Generate a unique filename for the audio file
        suffix = f"_{filename_hint}" if filename_hint else ""
        audio_filename = f"tts_{uuid.uuid4().hex}{suffix}.wav"
        os.makedirs(current_app.config.get('TEMP_AUDIO_PATH'), exist_ok=True)
        audio_path = os.path.join(current_app.config.get('TEMP_AUDIO_PATH'), audio_filename)

        # Initialize pyttsx3 engine
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.setProperty('pitch', 0.8)

        # Attempt to save to file using pyttsx3
        # Note: pyttsx3.save_to_file requires the engine.runAndWait() to complete before the file is written.
        # This is a blocking call.
        engine.save_to_file(text, audio_path)
        engine.runAndWait()

        # Check if the file was created successfully
        if os.path.exists(audio_path):
            current_app.logger.info(f"text_to_speech_util: TTS successful (pyttsx3), file saved as {audio_filename}")
            return audio_path
        else:
            current_app.logger.error(f"text_to_speech_util: TTS (pyttsx3) completed but file {audio_path} was not found.")
            return None

    except Exception as e:
        current_app.logger.error(f"text_to_speech_util: Exception occurred with pyttsx3 - {e}")
        return None


def dash_text_to_speech(text:str, dashscope_api_key:str,model="sambert-zhiqian-v1", voice="zhiqian"):
    dashscope.api_key = dashscope_api_key
    try:

        audio_filename = f"tts_{uuid.uuid4().hex}.wav"
        os.makedirs(current_app.config.get('TEMP_AUDIO_PATH'),exist_ok=True)
        audio_file_path = os.path.join(current_app.config.get('TEMP_AUDIO_PATH'),audio_filename)

        res = SpeechSynthesizer.call(
            model=model,
            text=text,
            sample_text=48000,
            voice=voice
        )
        current_app.logger.info('requestId: ', res.get_response()['request_id'])
        if res.get_audio_data() is not None:
            with open(audio_file_path, 'wb') as f:
                f.write(res.get_audio_data())
            current_app.logger.info(' get response: %s' % (res.get_response()))
            return audio_file_path

    except Exception as e:
        current_app.logger.error(f"文本转语音过程中发生异常: {e}")
        return None


if __name__ == "__main__":
    sample_text = ("这个世界，什么都可以安排，唯独你的心。"
                   "这个世界失去谁都不可怕不要紧，唯独失去了你自己。"
                   "以后还有很漫长很漫长的道路，都要一个人走完，都是靠自己，凭借自己的能力去完成。"
                   "这条道路，故事是昨天的瞬间，沿着长长的路，恍然如梦，到永远")
    # output_path = ".\\output_audio_tts_v2.wav"
    # model = 'sambert-zhiqian-v1'
    # voice = 'zhiqian'
    #
    # print(f"正在调用 DashScope (tts_v2) 服务将文本转换为语音...")
    # print(f"文本内容: {sample_text[:50]}...")  # 打印部分文本
    # print(f"输出路径: {output_path}")
    # print(f"模型: {model}, 音色: {voice}")
    #
    # success = dash_text_to_speech(
    #     text=sample_text,
    #     output_file_path=output_path,
    #     dashscope_api_key='your_dashscope_api_key',
    #     model="sambert-zhiqian-v1",
    #     voice="zhiqian"
    # )

    print(pyttsx_text_to_speech(text=sample_text))
