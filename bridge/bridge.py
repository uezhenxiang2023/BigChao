from bot.bot_factory import create_bot
from bridge.context import Context
from bridge.reply import Reply
from common import const
from common.log import logger
from common.singleton import singleton
from config import conf
from image.image_factory import create_image
from video.video_factory import create_video
from translate.factory import create_translator
from voice.factory import create_voice
from common.model_status import model_state

@singleton
class Bridge(object):
    def __init__(self, session_id=None):
        self.btype = {
            "chat": model_state.get_basic_state(session_id),
            "voice_to_text": conf().get("voice_to_text", "openai"),
            "text_to_voice": conf().get("text_to_voice", "google"),
            "translate": conf().get("translate", "baidu")
        }
        self.session_id = session_id
        self.model_type = self.btype['chat']
        if self.model_type in (const.GPT35_LIST + const.GPT4_LIST + const.GPT_O1_LIST + const.GPT54_LIST):
            self.btype["chat"] = const.CHATGPT
        if self.model_type in ["text-davinci-003"]:
            self.btype["chat"] = const.OPEN_AI
        if conf().get("use_azure_chatgpt", False):
            self.btype["chat"] = const.CHATGPTONAZURE
        if self.model_type in ["wenxin", "wenxin-4"]:
            self.btype["chat"] = const.BAIDU
        if self.model_type in ["xunfei"]:
            self.btype["chat"] = const.XUNFEI
        if self.model_type in [const.QWEN]:
            self.btype["chat"] = const.QWEN
        if self.model_type in (const.GEMINI_1_PRO_LIST + const.GEMINI_15_PRO_LIST + const.GEMINI_15_FLASH_LIST + const.GEMINI_GENAI_SDK) or (isinstance(self.model_type, str) and self.model_type.lower().startswith("gemini")):
            self.btype["chat"] = const.GEMINI
        if self.model_type in [const.OPEN_AI_ASSISTANT]:
            self.btype["chat"] = const.OPEN_AI_ASSISTANT
        if self.model_type in const.DOUAO:
            self.btype["chat"] = const.ARK
        if self.model_type in const.GROK_LIST:
            self.btype["chat"] = const.GROK

        if conf().get("use_linkai") and conf().get("linkai_api_key"):
            self.btype["chat"] = const.LINKAI
            if not conf().get("voice_to_text") or conf().get("voice_to_text") in ["openai"]:
                self.btype["voice_to_text"] = const.LINKAI
            if not conf().get("text_to_voice") or conf().get("text_to_voice") in ["openai", const.TTS_1, const.TTS_1_HD]:
                self.btype["text_to_voice"] = const.LINKAI

        if self.model_type in const.CLAUDE_SDK:
            self.btype["chat"] = const.CLAUDEAI
        self.bots = {}
        self.chat_bots = {}

    def get_bot(self, typename):
        # 图片/视频每次都实时取 model_state，不走缓存
        if typename == "prompt_to_image":
            model = model_state.get_image_model(self.session_id)
            return create_image(model)
        elif typename == "prompt_to_video":
            model = model_state.get_video_state(self.session_id)
            return create_video(model)
        if self.bots.get(typename) is None:
            #logger.info("create bot {}[{}] for {}".format(self.btype[typename], self.model, typename))
            if typename == "text_to_voice":
                logger.info("create bot {}[{}] for {}".format(self.btype[typename], conf().get("text_to_voice_model"), typename))
                self.bots[typename] = create_voice(self.btype[typename])
            elif typename == "voice_to_text":
                logger.info("create bot {}[{}] for {}".format(self.btype[typename], conf().get("voice_to_text_model"), typename))
                self.bots[typename] = create_voice(self.btype[typename])
            elif typename == "chat":
                logger.info("create bot {}[{}] for {}".format(self.btype[typename], (self.model_type or "").upper(), typename))
                self.bots[typename] = create_bot(self.btype[typename])
            elif typename == "translate":
                self.bots[typename] = create_translator(self.btype[typename])
        return self.bots[typename]

    def get_bot_type(self, typename):
        return self.btype[typename]

    def fetch_reply_content(self, query, context: Context) -> Reply:
        return self.get_bot("chat").reply(query, context)

    def fetch_voice_to_text(self, voiceFile) -> Reply:
        return self.get_bot("voice_to_text").voiceToText(voiceFile)

    def fetch_text_to_voice(self, text) -> Reply:
        return self.get_bot("text_to_voice").textToVoice(text)

    def fetch_translate(self, text, from_lang="", to_lang="en") -> Reply:
        return self.get_bot("translate").translate(text, from_lang, to_lang)
    
    def fetch_image_content(self, prompt, context: Context) -> Reply:
        return self.get_bot("prompt_to_image").reply(prompt, context)

    def fetch_video_content(self, prompt, context: Context, fps=1) -> Reply:
        return self.get_bot("prompt_to_video").reply(prompt, context)

    def find_chat_bot(self, bot_type: str):
        if self.chat_bots.get(bot_type) is None:
            self.chat_bots[bot_type] = create_bot(bot_type)
        return self.chat_bots.get(bot_type)

    def reset_bot(self, session_id):
        """
        重置bot路由
        """
        self.__init__(session_id)
