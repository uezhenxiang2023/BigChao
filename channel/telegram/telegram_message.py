import os
import asyncio
from bridge.context import ContextType
from channel.chat_message import ChatMessage
from common.log import logger
from common.tmp_dir import get_request_dir


def get_file(file_id):
        """
        Use this method to get basic information about a file and prepare it for downloading.
        """
        """bot_token = conf().get("telegram_bot_token")
        proxy_url = conf().get("telegram_proxy_url")
        updater = Updater(bot_token, request_kwargs={'proxy_url': proxy_url})
        bot = updater.bot
        try:
            file = bot.get_file(file_id)
            return file, None
        except Exception as e:
            return None, e"""


def get_file_name(file):
    file_path = file.file_path
    file_name_index = file_path.rfind("/")
    file_name = file_path[file_name_index+1:]
    return file_name


class TelegramMessage(ChatMessage):
    def __init__(self, telegram_message, is_group=False, main_loop=None):
        """
        TelegramMessage类的构建函数
        
        :param telegram_message: PTB 的 Message 对象
        :param is_group: 是否群聊
        :param main_loop: 【关键】主线程的事件循环，用于执行异步操作
        """
        super().__init__(telegram_message)

        # 1. 保存 Loop 和 Bot 引用
        self.loop = main_loop
        # 在 PTB v20 中，message 对象知道自己的 bot 是谁
        self.bot = telegram_message._bot

        self.msg_id = telegram_message["message_id"]
        self.create_time = telegram_message["date"]
        self.is_group = is_group
        self.from_user_id = telegram_message["from_user"]["id"]
        self.to_user_id = telegram_message["chat_id"]
        self.other_user_id = self.to_user_id
        self.user_dir = get_request_dir("telegram", self.from_user_id)
        reply_to_message = telegram_message.reply_to_message
        self.parent_id = reply_to_message.message_id if reply_to_message else None

        def _run_sync(coro):
            """
            在同步环境执行异步代码,利用 run_coroutine_threadsafe 把任务发回主线程，并阻塞等待结果
            
            :param coro: PTB定义的异步方法
            """
            if self.loop:
                future = asyncio.run_coroutine_threadsafe(coro, self.loop)
                return future.result() # 阻塞直到拿到结果
            return None

        if telegram_message["text"]:
            self.ctype = ContextType.TEXT
            self.content = telegram_message["text"]
        elif telegram_message["voice"]:
            self.ctype = ContextType.VOICE
            # 获取 file_id
            file_id = telegram_message.voice.file_id
            # 【异步转同步】获取文件信息对象
            new_file = _run_sync(self.bot.get_file(file_id))
            
            # v20 的 file_path 可能需要处理一下文件名
            if new_file and new_file.file_path:
                file_name = new_file.file_path.split('/')[-1]
            else:
                file_name = f"{file_id}.ogg"
                
            self.content = self.user_dir + file_name
            # 定义下载函数：注意 download_to_drive 也是异步的
            self._prepare_fn = lambda: _run_sync(new_file.download_to_drive(self.content))
        elif telegram_message["photo"]:
            self.ctype = ContextType.IMAGE
            # 取最大分辨率的图
            photo_obj = telegram_message.photo[-1]
            file_id = photo_obj.file_id
            try:
                # 【异步转同步】调用 get_file
                new_file = _run_sync(self.bot.get_file(file_id))
                
                if new_file:
                    file_name = new_file.file_path.split('/')[-1]
                    self.content = self.user_dir + file_name
                    # 【异步转同步】下载
                    self._prepare_fn = lambda: _run_sync(new_file.download_to_drive(self.content))
                else:
                    raise Exception("get_file return None")
                    
            except Exception as e:
                error_reply = f"[TELEGRAMBOT] Image download failed: {e}"
                logger.error(error_reply)
                raise NotImplementedError(error_reply)
            
        elif telegram_message["video"]:
            self.ctype = ContextType.VIDEO
            file_id = telegram_message.video.file_id
            file_name = telegram_message.video.file_name or f"{file_id}.mp4"
            new_file = _run_sync(self.bot.get_file(file_id))
            self.content = self.user_dir + file_name
            self._prepare_fn = lambda: _run_sync(new_file.download_to_drive(self.content))
        elif telegram_message.new_chat_members:
            self.ctype = ContextType.JOIN_GROUP
            # 获取加入者的名字
            users = ", ".join([u.full_name for u in telegram_message.new_chat_members])
            self.content = f"{users} 加入了群聊"
            self.actual_user_nickname = users
        elif telegram_message.left_chat_member:
            self.ctype = ContextType.EXIT_GROUP
            user = telegram_message.left_chat_member.full_name
            self.content = f"{user} 移出了群聊"
            self.actual_user_nickname = user
        elif telegram_message["document"]:
            self.ctype = ContextType.FILE
            file_id = telegram_message.document.file_id
            file_name = telegram_message.document.file_name
            
            try:
                new_file = _run_sync(self.bot.get_file(file_id))
                if new_file:
                    self.content = self.user_dir + file_name
                    self._prepare_fn = lambda: _run_sync(new_file.download_to_drive(self.content))
                else:
                    raise Exception("get_file failed")
            except Exception as e:
                logger.error(f"[TELEGRAMBOT] File download error: {e}")
                raise NotImplementedError(str(e))
        elif telegram_message["text_html"] and 'https://' in telegram_message["text_html"]:
            self.ctype = ContextType.SHARING
            self.content = telegram_message["text_html"]

        else:
            # 暂时忽略其他不支持的类型（如 Sticker, Location）
                logger.debug(f"[TELEGRAM] Unsupported message type: {telegram_message}")
                raise NotImplementedError("Unsupported message type")

    def _download_quoted_file(self, file_id, file_name):
        if not file_id or not file_name:
            return None
        os.makedirs(self.user_dir, exist_ok=True)
        quoted_file_path = os.path.join(self.user_dir, f"quoted_{self.parent_id}_{os.path.basename(file_name)}")
        new_file = self._run_sync(self.bot.get_file(file_id))
        if not new_file:
            return None
        self._run_sync(new_file.download_to_drive(quoted_file_path))
        return quoted_file_path

    def _run_sync(self, coro):
        if self.loop:
            future = asyncio.run_coroutine_threadsafe(coro, self.loop)
            return future.result()
        return None

    def get_quoted_image_path(self):
        reply_to_message = getattr(self._rawmsg, "reply_to_message", None)
        if not reply_to_message or not reply_to_message.photo:
            return None
        photo_obj = reply_to_message.photo[-1]
        file_name = f"{photo_obj.file_unique_id or photo_obj.file_id}.jpg"
        return self._download_quoted_file(photo_obj.file_id, file_name)

    def get_quoted_video_path(self):
        reply_to_message = getattr(self._rawmsg, "reply_to_message", None)
        if not reply_to_message or not reply_to_message.video:
            return None
        video_obj = reply_to_message.video
        file_name = video_obj.file_name or f"{video_obj.file_unique_id or video_obj.file_id}.mp4"
        return self._download_quoted_file(video_obj.file_id, file_name)

    def get_quoted_file_path(self):
        reply_to_message = getattr(self._rawmsg, "reply_to_message", None)
        if not reply_to_message or not reply_to_message.document:
            return None
        document = reply_to_message.document
        suffix = os.path.splitext(document.file_name or "")[1].lstrip(".").lower()
        if suffix not in {"pdf", "doc", "docx", "txt"}:
            logger.info(
                f"[TELEGRAMBOT] skip quoted file download, unsupported suffix={suffix}, parent_id={self.parent_id}"
            )
            return None
        file_name = document.file_name or f"{document.file_unique_id or document.file_id}"
        return self._download_quoted_file(document.file_id, file_name)
