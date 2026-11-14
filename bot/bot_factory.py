"""
bot factory
"""
from common import const


def create_bot(bot_type):
    """
    create a bot_type instance
    :param bot_type: bot type code
    :return: bot instance
    """
    if bot_type == const.BAIDU:
        # 替换Baidu Unit为Baidu文心千帆对话接口
        # from bot.baidu.baidu_unit_bot import BaiduUnitBot
        # return BaiduUnitBot()
        from bot.baidu.baidu_wenxin import BaiduWenxinBot
        return BaiduWenxinBot()

    elif bot_type == const.CHATGPT:
        # ChatGPT 网页端web接口
        from bot.chatgpt.chat_gpt_bot import ChatGPTBot
        return ChatGPTBot()

    elif bot_type == const.OPEN_AI:
        # OpenAI 官方对话模型API
        from bot.openai.open_ai_bot import OpenAIBot
        return OpenAIBot()

    elif bot_type == const.XUNFEI:
        from bot.xunfei.xunfei_spark_bot import XunFeiBot
        return XunFeiBot()

    elif bot_type == const.LINKAI:
        from bot.linkai.link_ai_bot import LinkAIBot
        return LinkAIBot()

    elif bot_type == const.CLAUDEAI:
        from bot.claude.claude_ai_bot import ClaudeAIBot
        return ClaudeAIBot()

    elif bot_type == const.QWEN:
        from bot.ali.ali_qwen_bot import AliQwenBot
        return AliQwenBot()

    elif bot_type == const.GEMINI or (isinstance(bot_type, str) and bot_type.lower().startswith("gemini")):
        from bot.gemini.google_gemini_bot import GoogleGeminiBot
        return GoogleGeminiBot()
    
    elif bot_type == const.OPEN_AI_ASSISTANT:
        from bot.openai.open_ai_assistant import OpenAIAssistantBot
        return OpenAIAssistantBot()
    
    elif bot_type == const.ARK:
        from bot.ark.volcengine_ark_bot import VolcengineArkBot
        return VolcengineArkBot()

    elif bot_type == const.GROK:
        from bot.grok.grok_bot import GrokBot
        return GrokBot()

    raise RuntimeError
