from bot.session_manager import Session


class VolcengineArkSession(Session):
    def __init__(self, session_id, system_prompt=None, model="doubao-seed-1-6-251015"):
        super().__init__(session_id, system_prompt)
        self.model = model
        self.last_total_tokens = None  # 上一轮 API 返回的准确 token 数
        self.previous_response_id = None
        self.remote_history_outdated = False
        self.pending_video_url = None
        self.reset()

    def reset(self):
        super().reset()
        self.previous_response_id = None
        self.remote_history_outdated = False
        self.pending_video_url = None

    def set_system_prompt(self, system_prompt):
        self.system_prompt = system_prompt
        self.reset()

    def mark_remote_history_outdated(self):
        self.remote_history_outdated = True

    def calc_tokens(self):
        """
        粗略估算当前 session 的 token 数。
        content 可能是字符串或多模态列表，统一提取文本部分计算。
        """
        total = 0
        for msg in self.messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                total += len(content) // 2
            elif isinstance(content, list):
                for block in content:
                    if block.get("type") == "text":
                        total += len(block.get("text", "")) // 2
                    elif block.get("type") == "image_url":
                        total += 1500
                    elif block.get("type") == "video_url":
                        total += 8000
        return total

    def discard_exceeding(self, max_tokens, cur_tokens=None):
        """
        当历史消息超出 max_tokens 时，从最早的非 system 消息开始丢弃，
        直到 token 数满足要求。
        保证至少保留最后一轮 user 消息，不会把当前消息丢掉。
        """
        # 第一轮用传入的准确值，后续截断后用估算值
        total = cur_tokens if cur_tokens is not None else (
        self.last_total_tokens + self._calc_last_message_tokens()
        if self.last_total_tokens is not None 
        else self.calc_tokens()
        )

        while True:
            if total <= max_tokens:
                return total

            first_non_system = next(
                (i for i, m in enumerate(self.messages) if m.get("role") != "system"),
                None
            )
            if first_non_system is None:
                return total

            remaining_non_system = [
                m for m in self.messages if m.get("role") != "system"
            ]
            if len(remaining_non_system) <= 1:
                return total

            self.messages.pop(first_non_system)
            total = self.calc_tokens()  # 截断后重新估算
    
    def _calc_last_message_tokens(self) -> int:
        """估算最后一条消息即最新query消息的 token 数"""
        if not self.messages:
            return 0
        msg = self.messages[-1]
        content = msg.get("content", "")
        if isinstance(content, str):
            return len(content) // 2
        elif isinstance(content, list):
            total = 0
            for block in content:
                if block.get("type") == "text":
                    total += len(block.get("text", "")) // 2
                elif block.get("type") == "image_url":
                    total += 1500
                elif block.get("type") == "video_url":
                    total += 8000
            return total
        return 0
