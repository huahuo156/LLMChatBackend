# services/audio_service.py
import os

from utils.audio_utils import dash_text_to_speech, pyttsx_text_to_speech


class AudioService:
    def __init__(self):
        self.dashscope_api_key = os.getenv('DASHSCOPE_API_KEY')

    def convert_text_to_speech(self, text):
        if not self.dashscope_api_key:
            # 配置中没有 dashscope api key
            audio_file_path = pyttsx_text_to_speech(text=text)
        else:
            # 配置中有 dashscope api key
            audio_file_path = dash_text_to_speech(
                text=text,
                dashscope_api_key=self.dashscope_api_key
            )
        return audio_file_path
