"""聊天窗口中的模型状态"""

from config import conf

class UserModelState:
    def __init__(self):
        self._model_states = {}
    
    def __get_model_state__(self, user_id):
        """获取用户模型状态,如果不存在则创建新的"""
        if user_id not in self._model_states:
            self._model_states[user_id] = {
                'model': conf().get('model'),
                'text_to_image': conf().get('text_to_image'),
                'image_mode': conf().get('image_mode', 'Generation'),
                'image_size': conf().get('image_create_size', '1k'),
                'image_quality': conf().get('image_create_quality', 'low'),
                'text_to_voice': conf().get('text_to_voice'),
                'text_to_video': conf().get('text_to_video'),
                'video_mode': conf().get('video_mode', 'FirstLast'),
                'watermark': conf().get('watermark', True)
            }
        return self._model_states[user_id]
    
    def toggle_basic_model(self, user_id, model):
        """切换用户的基础模型"""
        state = self.__get_model_state__(user_id)
        state['model'] = model
        return state['model']
    
    def toggle_image_model(self, user_id, image_model):
        """切换用户的图像模型"""
        state = self.__get_model_state__(user_id)
        state['text_to_image'] = image_model
        return state['text_to_image']

    def toggle_image_mode(self, user_id, image_mode):
        """切换用户的出图模式"""
        state = self.__get_model_state__(user_id)
        state['image_mode'] = image_mode
        return state['image_mode']

    def toggle_image_size(self, user_id, image_size):
        """切换用户的图片尺寸状态"""
        state = self.__get_model_state__(user_id)
        state['image_size'] = image_size
        return state['image_size']

    def toggle_image_quality(self, user_id, image_quality):
        """切换用户的图片质量状态"""
        state = self.__get_model_state__(user_id)
        state['image_quality'] = image_quality
        return state['image_quality']
    
    def toggle_voice_model(self, user_id, voice_model):
        """切换用户的语音模型"""
        state = self.__get_model_state__(user_id)
        state['text_to_voice'] = voice_model
        return state['text_to_voice']
    
    def toggle_video_model(self, user_id, video_model):
        """切换用户的视频模型"""
        state = self.__get_model_state__(user_id)
        state['text_to_video'] = video_model
        return state['text_to_video']

    def toggle_video_mode(self, user_id, video_mode):
        """切换用户的视频模式状态"""
        state = self.__get_model_state__(user_id)
        state['video_mode'] = video_mode
        return state['video_mode']

    def toggle_watermark(self, user_id, watermark):
            """切换用户视频/图片模型的水印状态"""
            state = self.__get_model_state__(user_id)
            state['watermark'] = watermark
            return state['watermark']

    def get_basic_state(self, user_id):
        """获取用户基础模型状态"""
        return self.__get_model_state__(user_id)['model']
    
    def get_image_model(self, user_id):
        """获取用户图像模型状态"""
        return self.__get_model_state__(user_id)['text_to_image']

    def get_image_mode(self, user_id):
        """获取用户出图模式状态"""
        return self.__get_model_state__(user_id).get('image_mode', conf().get('image_mode', 'Generation'))

    def get_image_size(self, user_id):
        """获取用户图片尺寸状态"""
        return self.__get_model_state__(user_id)['image_size']

    def get_image_quality(self, user_id):
        """获取用户图片质量状态"""
        return self.__get_model_state__(user_id)['image_quality']
    
    def get_voice_state(self, user_id):
        """获取用户语音状态"""
        return self.__get_model_state__(user_id)['text_to_voice']
    
    def get_video_state(self, user_id):
        """获取用户视频模型状态"""
        return self.__get_model_state__(user_id)['text_to_video']

    def get_video_mode(self, user_id):
        """获取用户视频模式状态"""
        return self.__get_model_state__(user_id)['video_mode']

    def get_watermark(self, user_id):
            """获取用户视频/图片模型的水印状态"""
            return self.__get_model_state__(user_id)['watermark']
    
    def clear_model_state(self, user_id):
        """清除用户状态"""
        if user_id in self._model_states:
            del self._model_states[user_id]

# 创建全局实例
model_state = UserModelState()
