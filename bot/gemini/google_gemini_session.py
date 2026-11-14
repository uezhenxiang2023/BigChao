from bot.session_manager import SessionManager, Session
from common import const


class GoogleGeminiSession(Session):
    def __init__(self, session_id, system_prompt=None, model=const.GEMINI_25_FLASH):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.last_total_tokens = None
        self.last_interaction_id = None
        self.reset()

    def reset(self):
        super().reset()
        self.last_interaction_id = None

    def calc_tokens(self):
        """
        粗略估算当前 session 的 token 数。
        content 可能是字符串或多模态列表，统一提取文本部分计算。
        """
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content)
            elif isinstance(content, list):
                for block in content:
                    block_type = block.get("type")
                    if block_type == "text":
                        total += len(block.get("text", ""))
                    elif block_type == "image_url":
                        total += 1500
                    elif block_type == "video_url":
                        total += 8000
        return total

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        """
        当历史消息超出 max_tokens 时，从最早的非 system 消息开始丢弃，
        直到 token 数满足要求。
        保证至少保留最后一轮 user 消息，不会把当前消息丢掉。
        """
        total = cur_tokens if cur_tokens is not None else (
            self.last_total_tokens + self._calc_last_message_tokens()
            if self.last_total_tokens is not None
            else self.calc_tokens()
        )

        while True:
            if total <= max_tokens:
                return total

            first_non_system = next(
                (i for i, msg in enumerate(self.messages) if msg.get("role") != "system"),
                None
            )
            if first_non_system is None:
                return total

            remaining_non_system = [
                msg for msg in self.messages if msg.get("role") != "system"
            ]
            if len(remaining_non_system) <= 1:
                return total

            self.messages.pop(first_non_system)
            total = self.calc_tokens()

    def _calc_last_message_tokens(self):
        """估算最后一条消息即最新 query 消息的 token 数。"""
        if not self.messages:
            return 0

        msg = self.messages[-1]
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content)
        if isinstance(content, list):
            total = 0
            for block in content:
                block_type = block.get("type")
                if block_type == "text":
                    total += len(block.get("text", ""))
                elif block_type == "image_url":
                    total += 1500
                elif block_type == "video_url":
                    total += 8000
            return total
        return 0


_gemini_sessions = SessionManager(GoogleGeminiSession, model=const.GEMINI_25_FLASH)
