import io
import requests
import logging
import asyncio
import html
import time
import httpx

from io import BytesIO

from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from channel.chat_channel import ChatChannel
from channel.chat_message import ChatMessage
from channel.telegram.telegram_message import TelegramMessage
from common import const
from common.model_status import model_state
from common.tool_button import tool_state
from common.log import logger
from common.singleton import singleton
from config import conf
from channel.telegram.telegram_text_util import escape

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.error import BadRequest
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters, CallbackContext, CallbackQueryHandler
from telegram.request import HTTPXRequest

@singleton
class TelegramChannel(ChatChannel):
    _MEDIA_MODEL_DICT = {
        'Seedream': const.DOUBAO_SEEDREAM_5,
        'KlingImage': const.KLING_V3_OMNI,
        'GPTImage': const.GPT_IMAGE_2,
        'NanoBanana': const.GEMINI_31_FLASH_IMAGE,
        'GrokImage': const.GROK_IMAGINE_IMAGE_QUALITY,
        'LumaImage': const.UNI_1,
        'Seedance': const.DOUBAO_SEEDANCE_20,
        'KlingVideo': const.KLING_V3_OMNI,
        'Veo': const.VEO_31,
        'GrokVideo': const.GROK_IMAGINE_VIDEO,
        'LumaVideo': const.LUMA_RAY_32
    }
    def __init__(self, session_id=None):
        super().__init__()
        self.last_update_time = time.time()
        self.logger = logging.getLogger(__name__)
        self.bot_token = conf().get("telegram_bot_token")
        self.proxy_url =conf().get("telegram_proxy_url")

        # Pre-assign placeholder menu text
        self.FIRST_MENU = "<b>Menu 1</b>\n\nA beautiful menu with a shiny inline button."
        self.SECOND_MENU = "<b>Menu 2</b>\n\nA better menu with even more shiny inline buttons."

        # Pre-assign placeholder button text
        self.NEXT_BUTTON = "Next"
        self.BACK_BUTTON = "Back"
        self.TUTORIAL_BUTTON = "Tutorial"

        # Build placeholder keyboards
        self.FIRST_MENU_MARKUP = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(self.NEXT_BUTTON, callback_data=self.NEXT_BUTTON)]
            ]
        )
        self.SECOND_MENU_MARKUP = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(self.BACK_BUTTON, callback_data=self.BACK_BUTTON)],
                [InlineKeyboardButton(self.TUTORIAL_BUTTON, url="https://core.telegram.org/bots/api")]
            ]
        )

        self.SKILL_MENU_TITLE = "<b>Producer Skill</b>\n\nSkills of Production Agent."
        self.SKILL_CALLBACK_PREFIX = "skill:"
        self.SKILL_OPTIONS = ["print", "breakdown", "search"]
        self.IMAGE_MODE_MENU_TITLE = "<b>Image Mode</b>\n\nGenerate new image or edit exiting image."
        self.IMAGE_MODE_CALLBACK_PREFIX = "image_mode:"
        self.IMAGE_MODE_OPTIONS = ["Generation", "Editing"]
        self.VIDEO_MODE_MENU_TITLE = "<b>Video Mode</b>\n\nImage role for image to video generation."
        self.VIDEO_MODE_CALLBACK_PREFIX = "video_mode:"
        self.VIDEO_MODE_OPTIONS = ["FirstLast", "Reference"]
        self.IMAGE_MODEL_MENU_TITLE = "<b>Image Model</b>\n\nPick a image model."
        self.IMAGE_MODEL_CALLBACK_PREFIX = "image_model:"
        self.IMAGE_MODEL_OPTIONS = ["Seedream", "KlingImage", "GPTImage", "NanoBanana", "GrokImage", "LumaImage"]
        self.VIDEO_MODEL_MENU_TITLE = "<b>Video Model</b>\n\nPick a video model."
        self.VIDEO_MODEL_CALLBACK_PREFIX = "video_model:"
        self.VIDEO_MODEL_OPTIONS = ["Seedance", "KlingVideo", "Veo", "GrokVideo", "LumaVideo"]

        self.SKILL_MAP = {
            'print': self.print,
            'breakdown': self.breakdown,
            'search': self.search
        }

        # 新增：用于存储主事件循环的引用
        #self.main_loop = None

    def _get_current_image_model_id(self, user_id):
        return model_state.get_image_model(user_id).upper()

    def _get_current_video_model_id(self, user_id):
        return model_state.get_video_state(user_id).upper()

    def _build_model_switch_text(self, subject_name, model_name):
        return f"[INFO]\n{subject_name}已切换为：{model_name.upper()}"
    
    def _build_skill_markup(self):
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(option, callback_data=f"{self.SKILL_CALLBACK_PREFIX}{option}")]
                for option in self.SKILL_OPTIONS
            ]
        )

    def _build_video_mode_markup(self):
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(option, callback_data=f"{self.VIDEO_MODE_CALLBACK_PREFIX}{option}")]
                for option in self.VIDEO_MODE_OPTIONS
            ]
        )

    def _build_image_mode_markup(self):
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(option, callback_data=f"{self.IMAGE_MODE_CALLBACK_PREFIX}{option}")]
                for option in self.IMAGE_MODE_OPTIONS
            ]
        )
    
    def _build_video_model_markup(self):
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(option, callback_data=f"{self.VIDEO_MODEL_CALLBACK_PREFIX}{option}")]
                for option in self.VIDEO_MODEL_OPTIONS
            ]
        )
    
    def _build_image_model_markup(self):
        return InlineKeyboardMarkup(
            [
                [InlineKeyboardButton(option, callback_data=f"{self.IMAGE_MODEL_CALLBACK_PREFIX}{option}")]
                for option in self.IMAGE_MODEL_OPTIONS
            ]
        )

    def _build_video_mode_menu_text(self, current_mode):
        return f"{self.VIDEO_MODE_MENU_TITLE}\n\nCurrent: <b>{current_mode}</b>"

    def _build_image_mode_menu_text(self, current_mode):
        return f"{self.IMAGE_MODE_MENU_TITLE}\n\nCurrent: <b>{current_mode}</b>"
    
    def _build_video_model_menu_text(self, current_model):
        return f"{self.VIDEO_MODEL_MENU_TITLE}\n\nCurrent: <b>{current_model}</b>"
    
    def _build_image_model_menu_text(self, current_model):
        return f"{self.IMAGE_MODEL_MENU_TITLE}\n\nCurrent: <b>{current_model}</b>"
    
    def _build_skill_menu_text(self, current_mode):
        return f"{self.SKILL_MENU_TITLE}\n\nCurrent:\n<b>{current_mode}</b>"

    async def _send_simple_text(self, chat_id, text):
        await self.application.bot.send_message(chat_id=chat_id, text=text)

    async def echo(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        This function would be added to the dispatcher as a handler for messages coming from the Bot API
        """
        # 每次收到消息，刷新一下时间
        self.last_update_time = time.time()
        message = update.effective_message
        chat = update.effective_chat
        user = update.effective_user
        chat_id = chat.id
        from_user_id = user.id if user else None
        parent_id = getattr(getattr(message, "reply_to_message", None), "message_id", None)
        message_type = (
            "text" if message.text else
            "photo" if message.photo else
            "document" if message.document else
            "video" if message.video else
            "voice" if message.voice else
            "status_update" if (message.new_chat_members or message.left_chat_member) else
            "unknown"
        )
        logger.info(
            "[TELEGRAM-event] event_id=%s, message_id=%s, parent_id=%s, create_time=%s, chat_type=%s, chat_id=%s, open_id=%s, message_type=%s",
            update.update_id,
            message.message_id,
            parent_id,
            message.date,
            chat.type,
            chat_id,
            from_user_id,
            message_type,
        )

        # Print tool_button stasus to console
        logger.info(
            f'[TELEGRAM-search] is {tool_state.get_search_state(chat_id)},\
            [TELEGRAM-video_mode] is {model_state.get_video_mode(chat_id)},\
            [TELEGRAM-print] is {tool_state.get_print_state(chat_id)},\
            [TELEGRAM-breakdown] is {tool_state.get_breakdown_state(chat_id)},\
            [TELEGRAM-image] is {tool_state.get_image_state(chat_id)},\
            [TELEGRAM-image_mode] is {model_state.get_image_mode(chat_id)},\
            [TELEGRAM-video] is {tool_state.get_edit_state(chat_id)},\
            [TELEGRAM-image_model] is {model_state.get_image_model(chat_id)},\
            [TELEGRAM-video_model] is {model_state.get_video_state(chat_id)},\
            requester={chat_id}'
        )
        
        # 使用 run_in_executor 将同步的业务逻辑扔到子线程
        # 这样 handler_single_msg 里的耗时操作（如 GPT 请求）就不会卡死机器人
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None, 
            self.handler_single_msg, 
            update.message
        )

    async def print(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        This function handles the /print command
        """
        chat_id = update.effective_chat.id
        if tool_state.get_print_state(chat_id):
            text = "[INFO]\n剧本排版功能已关闭，可以在消息框输入#print开启，也可以在‘skills’菜单中点击print开启。"
        else:
            text = "[INFO]\n剧本排版功能已开启,请先上传pdf格式的剧本，再发我编剧姓名。\n我会按照好莱坞编剧工会的标准格式进行排版，让您的剧本看起来更专业、读起来更舒服，大大提升获得‘绿灯’的几率。"

        tool_state.toggle_printing(chat_id)

        await context.bot.send_message(
            chat_id=chat_id,
            text=text
        )
        logger.info(f'[TELEGRAMBOT]{text} requester={chat_id}')
    
    async def breakdown(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        This function handles the /breakdown command
        """
        chat_id = update.effective_chat.id
        if tool_state.get_breakdown_state(chat_id):
            text = "[INFO]\n拆解顺分场表功能已关闭，可以在消息框输入#breakdown开启，也可以在‘skills’菜单中点击breakdown开启。"
        else:
            text = "[INFO]\n拆解顺分场表功能已开启。"

        tool_state.toggle_breakdowning(chat_id)

        await context.bot.send_message(
            chat_id=chat_id,
            text=text
        )
        logger.info(f'[TELEGRAMBOT]{text} requester={chat_id}')

    async def search(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        This function handles the /search command
        """
        chat_id = update.effective_chat.id
        if tool_state.get_search_state(chat_id):
            text = "[INFO]\n联网功能已关闭，如果需要，可以在‘skills’菜单中点击search开启。"
        else:
            text = "[INFO]\n联网搜索功能已开启，需要我帮你查询点啥？"

        tool_state.toggle_searching(chat_id)
        await context.bot.send_message(
            chat_id=chat_id,
            text=text
        )
        logger.info(f'[TELEGRAMBOT]{text} requester={chat_id}')

    async def skill(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        current_mode = (
            f"print:{tool_state.get_print_state(chat_id)}\nbreakdown:{tool_state.get_breakdown_state(chat_id)}\nsearch:{tool_state.get_search_state(chat_id)}"
        )
        menu_text = self._build_skill_menu_text(current_mode)
        await context.bot.send_message(
            chat_id=chat_id,
            text=menu_text,
            parse_mode="HTML",
            reply_markup=self._build_skill_markup(),
        )
        logger.info(f"[TELEGRAMBOT] skill menu opened, current_mode={current_mode}, requester={chat_id}")
    
    async def image_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        current_mode = model_state.get_image_mode(chat_id)
        menu_text = self._build_image_mode_menu_text(current_mode)
        await context.bot.send_message(
            chat_id=chat_id,
            text=menu_text,
            parse_mode="HTML",
            reply_markup=self._build_image_mode_markup(),
        )
        logger.info(f"[TELEGRAMBOT] image mode menu opened, current_mode={current_mode}, requester={chat_id}")

    async def video_mode(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        current_mode = model_state.get_video_mode(chat_id)
        menu_text = self._build_video_mode_menu_text(current_mode)
        await context.bot.send_message(
            chat_id=chat_id,
            text=menu_text,
            parse_mode="HTML",
            reply_markup=self._build_video_mode_markup(),
        )
        logger.info(f"[TELEGRAMBOT] video mode menu opened, current_mode={current_mode}, requester={chat_id}")

    async def video_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        current_video_model = model_state.get_video_state(chat_id)
        menu_text = self._build_video_model_menu_text(current_video_model)
        await context.bot.send_message(
            chat_id=chat_id,
            text=menu_text,
            parse_mode="HTML",
            reply_markup=self._build_video_model_markup(),
        )
        logger.info(f"[TELEGRAMBOT] video model menu opened, current_mode={current_video_model}, requester={chat_id}")
    
    async def image_model(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        chat_id = update.effective_chat.id
        current_image_model = model_state.get_image_model(chat_id)
        menu_text = self._build_image_model_menu_text(current_image_model)
        await context.bot.send_message(
            chat_id=chat_id,
            text=menu_text,
            parse_mode="HTML",
            reply_markup=self._build_image_model_markup(),
        )
        logger.info(f"[TELEGRAMBOT] image model menu opened, current_mode={current_image_model}, requester={chat_id}")

    async def menu(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.FIRST_MENU,
            parse_mode="HTML",
            reply_markup=self.FIRST_MENU_MARKUP
        )


    async def button_tap(self, update: Update, context: CallbackContext) -> None:
        """
        This handler processes the inline buttons on the menu
        """
        query = update.callback_query
        data = query.data
        chat_id = query.message.chat_id

        if data.startswith(self.VIDEO_MODE_CALLBACK_PREFIX):
            target_mode = data[len(self.VIDEO_MODE_CALLBACK_PREFIX):]
            if target_mode not in self.VIDEO_MODE_OPTIONS:
                await query.answer("Unsupported mode", show_alert=True)
                return

            model_state.toggle_video_mode(chat_id, target_mode)
            text = self._build_video_mode_menu_text(target_mode)
            await query.answer(f"Video mode: {target_mode}")
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode="HTML",
                    reply_markup=self._build_video_mode_markup(),
                )
            except BadRequest as e:
                if "Message is not modified" in str(e):
                    logger.info(
                        f"[TELEGRAMBOT] video mode message unchanged, target_mode={target_mode}, requester={chat_id}"
                    )
                else:
                    raise
            logger.info(f"[TELEGRAMBOT] video mode switched to {target_mode}, requester={chat_id}")
            return

        if data.startswith(self.IMAGE_MODE_CALLBACK_PREFIX):
            target_mode = data[len(self.IMAGE_MODE_CALLBACK_PREFIX):]
            if target_mode not in self.IMAGE_MODE_OPTIONS:
                await query.answer("Unsupported mode", show_alert=True)
                return

            model_state.toggle_image_mode(chat_id, target_mode)
            text = self._build_image_mode_menu_text(target_mode)
            await query.answer(f"Image mode: {target_mode}")
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode="HTML",
                    reply_markup=self._build_image_mode_markup(),
                )
            except BadRequest as e:
                if "Message is not modified" in str(e):
                    logger.info(
                        f"[TELEGRAMBOT] image mode message unchanged, target_mode={target_mode}, requester={chat_id}"
                    )
                else:
                    raise
            logger.info(f"[TELEGRAMBOT] image mode switched to {target_mode}, requester={chat_id}")
            return
        
        if data.startswith(self.VIDEO_MODEL_CALLBACK_PREFIX):
            target_button = data[len(self.VIDEO_MODEL_CALLBACK_PREFIX):]
            if target_button not in self.VIDEO_MODEL_OPTIONS:
                await query.answer("Unsupported mode", show_alert=True)
                return
            target_model = self._MEDIA_MODEL_DICT[target_button]
            model_state.toggle_video_model(chat_id, target_model)
            text = self._build_video_model_menu_text(target_model)
            await query.answer(f"Video model: {target_model}")
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode="HTML",
                    reply_markup=self._build_video_model_markup(),
                )
            except BadRequest as e:
                if "Message is not modified" in str(e):
                    logger.info(
                        f"[TELEGRAMBOT] video model message unchanged, target_model={model_state.get_video_state(chat_id)}, requester={chat_id}"
                    )
                else:
                    raise
            logger.info(f"[TELEGRAMBOT] video model switched to {target_model}, requester={chat_id}")
            return
        
        if data.startswith(self.IMAGE_MODEL_CALLBACK_PREFIX):
            target_button = data[len(self.IMAGE_MODEL_CALLBACK_PREFIX):]
            if target_button not in self.IMAGE_MODEL_OPTIONS:
                await query.answer("Unsupported mode", show_alert=True)
                return
            target_model = self._MEDIA_MODEL_DICT[target_button]
            model_state.toggle_image_model(chat_id, target_model)
            text = self._build_image_model_menu_text(target_model)
            await query.answer(f"Image model: {target_model}")
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode="HTML",
                    reply_markup=self._build_image_model_markup(),
                )
            except BadRequest as e:
                if "Message is not modified" in str(e):
                    logger.info(
                        f"[TELEGRAMBOT] image model message unchanged, target_model={model_state.get_image_model(chat_id)}, requester={chat_id}"
                    )
                else:
                    raise
            logger.info(f"[TELEGRAMBOT] image model switched to {target_model}, requester={chat_id}")
            return
        
        if data.startswith(self.SKILL_CALLBACK_PREFIX):
            target_button = data[len(self.SKILL_CALLBACK_PREFIX):]
            if target_button not in self.SKILL_OPTIONS:
                await query.answer("Unsupported mode", show_alert=True)
                return
            await self.SKILL_MAP[target_button](update, context)
            target_mode = (
            f"print:{tool_state.get_print_state(chat_id)}\nbreakdown:{tool_state.get_breakdown_state(chat_id)}\nsearch:{tool_state.get_search_state(chat_id)}"
        )
            text = self._build_skill_menu_text(target_mode)
            await query.answer(f"Skill mode: {target_mode}")
            try:
                await query.edit_message_text(
                    text=text,
                    parse_mode="HTML",
                    reply_markup=self._build_skill_markup(),
                )
            except BadRequest as e:
                if "Message is not modified" in str(e):
                    logger.info(
                        f"[TELEGRAMBOT] skill mode message unchanged, target_mode={target_mode}, requester={chat_id}"
                    )
                else:
                    raise
            logger.info(f"[TELEGRAMBOT] skill mode switched to {target_mode}, requester={chat_id}")
            return

        if data == self.NEXT_BUTTON:
            await query.answer()
            await query.edit_message_text(
                text=self.SECOND_MENU,
                parse_mode="HTML",
                reply_markup=self.SECOND_MENU_MARKUP
            )
            return

        if data == self.BACK_BUTTON:
            await query.answer()
            await query.edit_message_text(
                text=self.FIRST_MENU,
                parse_mode="HTML",
                reply_markup=self.FIRST_MENU_MARKUP
            )
            return

        # Close the query to end the client-side loading animation
        await query.answer()

    # 定义心跳任务
    async def heartbeat(self, context: ContextTypes.DEFAULT_TYPE):
        try:
            # 用 get_me() 验证网络连通性，不干扰 Polling 的 offset
            await self.application.bot.get_me()
            logger.debug("[HEARTBEAT] ❤️ 依然在线")

            # 检查 updater 是否意外停止
            if not self.application.updater.running:
                logger.warning("[WATCHDOG] ⚠️ Updater 停止了！尝试重启...")
                raise Exception("Updater 意外停止")

        except Exception as e:
            # 区分是网络问题还是 Updater 停止
            if str(e) == "Updater 意外停止":
                logger.warning("[WATCHDOG] ⚠️ 进入重启流程（Updater 停止触发）")
            else:
                logger.warning(f"[HEARTBEAT] 💔 心跳检测失败: {e}")
            try:
                logger.warning("[WATCHDOG] 正在强制重启 Polling...")

                # 第一步：停止 updater
                try:
                    await self.application.updater.stop()
                except Exception as stop_err:
                    logger.warning(f"[WATCHDOG] stop() 时报错（忽略）: {stop_err}")

                # 第二步：等待连接池完全释放
                await asyncio.sleep(3)

                # 第三步：强制关闭旧的 httpx 连接池，避免连接槽被占满
                # 第三步：强制关闭旧的 httpx 连接池，并重新初始化
                try:
                    await self.application.bot._request[0].shutdown()  # 替换 aclose()，走官方关闭流程
                    logger.info("[WATCHDOG] 旧连接池已关闭")
                except Exception as close_err:
                    logger.warning(f"[WATCHDOG] 关闭连接池时报错（忽略）: {close_err}")

                try:
                    await self.application.bot._request[0].initialize()  # ← 新增，重建 httpx client
                    logger.info("[WATCHDOG] 连接池已重新初始化")
                except Exception as init_err:
                    logger.warning(f"[WATCHDOG] 重新初始化连接池时报错（忽略）: {init_err}")

                await asyncio.sleep(1)

                # 第四步：等待网络恢复，最多等 3 分钟
                for i in range(18):  # 18 * 10秒 = 3分钟
                    await asyncio.sleep(10)
                    try:
                        async with httpx.AsyncClient(proxy=self.proxy_url) as client:
                            await client.get("https://api.telegram.org", timeout=5)
                        logger.info("[WATCHDOG] 网络已恢复，准备重启 Polling...")
                        break
                    except Exception:
                        logger.warning(f"[WATCHDOG] 网络未恢复，等待中... ({(i+1)*10}s)")

                # 第五步：重启 Polling
                await self.application.updater.start_polling(
                    allowed_updates=Update.ALL_TYPES,
                    timeout=5,
                    drop_pending_updates=False
                )
                logger.info("[WATCHDOG] Polling 重启成功！")

            except Exception as restart_error:
                logger.error(f"[WATCHDOG] 重启失败: {restart_error}")

    def main(self) -> None:
        """
        Start the bot.
        """
        # 准备 Request 对象
        request_params = {
            "connection_pool_size":32, # 链接窗口数量
            "pool_timeout":30,          # 链接排队时间
            "read_timeout":30,
            "write_timeout":30,
            "connect_timeout":10
        }
            
        if self.proxy_url:
            request_params["proxy"] = self.proxy_url
            logger.info(f"[TELEGRAM] 使用代理: {self.proxy_url}")

        
        request_instance = HTTPXRequest(**request_params)

        # Create the Application and pass it your bot's token.
        self.application = (
            Application.builder()
            .token(self.bot_token)
            .request(request_instance)
            .build()
        )

        # 每隔 1 分钟 (60秒) 执行一次
        # 这就像每隔一会儿戳一下服务器：“喂，由于什么原因断了吗？”
        # 如果断了，这个操作会强制抛出错误，进而唤醒僵尸连接
        if self.application.job_queue:
            self.application.job_queue.run_repeating(self.heartbeat, interval=60, first=10)
            logger.info("[TELEGRAM] 心跳保活任务已启动")

        # Register commands
        self.application.add_handler(CommandHandler("image_mode", self.image_mode))
        self.application.add_handler(CommandHandler("video_mode", self.video_mode))
        self.application.add_handler(CommandHandler("image_model", self.image_model))
        self.application.add_handler(CommandHandler("video_model", self.video_model))
        self.application.add_handler(CommandHandler("skills", self.skill))
        self.application.add_handler(CommandHandler("menu", self.menu))

        # Register handler for inline buttons
        self.application.add_handler(CallbackQueryHandler(self.button_tap))

        # Echo any message that is not a command
        filter_rules = (
            filters.TEXT | 
            filters.PHOTO | 
            filters.Document.ALL | 
            filters.VIDEO | 
            filters.VOICE |
            filters.StatusUpdate.ALL  # 如果您还需要处理进群/退群通知，加上这个
        ) & (~filters.COMMAND)
        self.application.add_handler(MessageHandler(filter_rules, self.echo))

        logger.info("[TELEGRAMBOT] Bot 正在启动...")

        # 在启动 Polling 之前，保存当前的事件循环
        # 注意：这里需要确保 loop 已经存在。
        # 如果是 script 运行，通常需要手动 get_event_loop
        try:
            self.main_loop = asyncio.get_event_loop()
        except RuntimeError:
            self.main_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.main_loop)
            
        logger.info(f"[TELEGRAMBOT] 主循环已捕获: {self.main_loop}")

        # Run the bot until you press Ctrl-C
        self.application.run_polling(
            allowed_updates=Update.ALL_TYPES,
            timeout=5,
            drop_pending_updates=True
        )

    def startup(self):
        self.main()

    def handler_single_msg(self, msg):
        try:
            cmsg = TelegramMessage(msg, False, self.main_loop)
        except NotImplementedError as e:
            logger.debug("[TELEGRAMBOT]single message {} skipped: {}".format(msg["MsgId"], e))
            error_reply = escape(e.args[0])
            self.application.bot.send_message(chat_id=msg.chat_id, text=error_reply)
            return None
        self.handle_single(cmsg)
        return None

    def handler_group_msg(self, msg):
        try:
            cmsg = TelegramMessage(msg, True, self.main_loop)
        except NotImplementedError as e:
            logger.debug("[TELEGRAMBOT]group message {} skipped: {}".format(msg["MsgId"], e))
            return None
        self.handle_group(cmsg)
        return None
    
    def handle_single(self, cmsg: ChatMessage):
        # filter system message
        if cmsg.other_user_id in ["weixin"]:
            return
        if cmsg.ctype == ContextType.VOICE:
            if conf().get("speech_recognition") != True:
                return
            logger.debug("[TELEGRAMBOT]receive voice msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.IMAGE:
            logger.debug("[TELEGRAMBOT]receive image msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.VIDEO:
            logger.debug("[TELEGRAMBOT]receive video msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.SHARING:
            logger.debug("[TELEGRAMBOT]receive url msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.PATPAT:
            logger.debug("[TELEGRAMBOT]receive patpat msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.FILE:
            logger.debug("[TELEGRAMBOT]receive file msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.TEXT:
            logger.debug("[TELEGRAMBOT]receive text msg: {}, cmsg={}".format(cmsg.content, cmsg))
        else:
            logger.debug("[TELEGRAMBOT]receive msg: {}, cmsg={}".format(cmsg.content, cmsg))
        context = self._compose_context(cmsg.ctype, cmsg.content, isgroup=False, msg=cmsg)
        if context and hasattr(cmsg, "get_quoted_image_path"):
            quoted_image_path = cmsg.get_quoted_image_path()
            if quoted_image_path:
                context["quoted_image_path"] = quoted_image_path
        if context and hasattr(cmsg, "get_quoted_video_path"):
            quoted_video_path = cmsg.get_quoted_video_path()
            if quoted_video_path:
                context["quoted_video_path"] = quoted_video_path
        if context and hasattr(cmsg, "get_quoted_file_path"):
            quoted_file_path = cmsg.get_quoted_file_path()
            if quoted_file_path:
                context["quoted_file_path"] = quoted_file_path
        if context:
            self.produce(context)

    def handle_group(self, cmsg: ChatMessage):
        if cmsg.ctype == ContextType.VOICE:
            if conf().get("group_speech_recognition") != True:
                return
            logger.debug("[TELEGRAMBOT]receive voice for group msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.IMAGE:
            logger.debug("[TELEGRAMBOT]receive image for group msg: {}".format(cmsg.content))
        elif cmsg.ctype in [ContextType.JOIN_GROUP, ContextType.PATPAT, ContextType.ACCEPT_FRIEND, ContextType.EXIT_GROUP]:
            logger.debug("[TELEGRAMBOT]receive note msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.TEXT:
            logger.debug("[TELEGRAMBOT]receive group msg: {}, cmsg={}".format(cmsg.content, cmsg))
            pass
        elif cmsg.ctype == ContextType.FILE:
            logger.debug(f"[TELEGRAMBOT]receive attachment msg, file_name={cmsg.content}")
        else:
            logger.debug("[TELEGRAMBOT]receive group msg: {}".format(cmsg.content))
        context = self._compose_context(cmsg.ctype, cmsg.content, isgroup=True, msg=cmsg)
        if context and hasattr(cmsg, "get_quoted_image_path"):
            quoted_image_path = cmsg.get_quoted_image_path()
            if quoted_image_path:
                context["quoted_image_path"] = quoted_image_path
        if context and hasattr(cmsg, "get_quoted_video_path"):
            quoted_video_path = cmsg.get_quoted_video_path()
            if quoted_video_path:
                context["quoted_video_path"] = quoted_video_path
        if context and hasattr(cmsg, "get_quoted_file_path"):
            quoted_file_path = cmsg.get_quoted_file_path()
            if quoted_file_path:
                context["quoted_file_path"] = quoted_file_path
        if context:
            self.produce(context)

    # 统一的发送函数，每个Channel自行实现，根据reply的type字段发送不同类型的消息
    def send(self, reply: Reply, context: Context):
        """
        兼容同步/异步/多线程的通用发送入口
        """
        # 1. 封装具体的发送任务（协程）
        coro = self._send_implementation(reply, context)
        
        # 2. 调度逻辑：自动判断当前环境
        try:
            # 尝试获取当前线程的 Loop
            loop = asyncio.get_running_loop()
            
            # 【情况A】主线程：直接创建任务
            loop.create_task(coro)
            logger.info("[TELEGRAMBOT] 检测到主线程环境，已使用 create_task 提交任务")
            
        except RuntimeError:
            # 【情况B】如果报错 "no running event loop"，说明我们在子线程里：跨线程提交给主 Loop
            logger.info("[TELEGRAMBOT] 检测到子线程环境，正在跨线程提交任务...")

            if self.main_loop and self.main_loop.is_running():
                asyncio.run_coroutine_threadsafe(coro, self.main_loop)
                logger.info("[TELEGRAMBOT] 已通过 run_coroutine_threadsafe 提交任务")
            else:
                # 最后的保底：如果实在拿不到 loop，说明程序状态不对
                logger.error("[TELEGRAMBOT] 无法获取主事件循环，消息发送失败！")

    async def _send_implementation(self, reply: Reply, context: Context):
        receiver = context["receiver"]
        reply_to_message_id = context.get("msg").parent_id if context.get("msg") else None

        def _reply_kwargs():
            if not reply_to_message_id:
                return {}
            return {"reply_to_message_id": reply_to_message_id}

        try:
            if reply.type == ReplyType.TEXT:
                await self.application.bot.send_message(chat_id=receiver, text=reply.content, **_reply_kwargs())
                logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(reply.content, receiver))
            elif reply.type == ReplyType.ERROR:
                error_text = reply.content if reply.content else const.ERROR_RESPONSE
                logger.error("[TELEGRAMBOT] sendMsg error, reply={}, receiver={}".format(reply, receiver))
                await self.application.bot.send_message(chat_id=receiver, text=error_text, **_reply_kwargs())
                logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(error_text, receiver))
            elif reply.type == ReplyType.INFO:
                await self.application.bot.send_message(chat_id=receiver, text=reply.content, **_reply_kwargs())
                logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(reply, receiver))
            elif reply.type == ReplyType.VOICE:
                await self.application.bot.send_voice(chat_id=receiver, voice=reply.content, **_reply_kwargs())
                logger.info("[TELEGRAMBOT] sendFile={}, receiver={}".format(reply.content, receiver))
            elif reply.type == ReplyType.IMAGE_URL:  # 获取网络资源
                response = reply.content
                if isinstance(response, list):
                    for img_url in response:
                        logger.debug(f"[TELEGRAMBOT] start download image, img_url={img_url}")
                        pic_res = requests.get(img_url, stream=True)
                        image_storage = io.BytesIO()
                        size = 0
                        for block in pic_res.iter_content(1024):
                            size += len(block)
                            image_storage.write(block)
                        logger.info(f"[TELEGRAMBOT] download image success, size={size}, img_url={img_url}")
                        image_storage.seek(0)
                        await self.application.bot.send_photo(chat_id=receiver, photo=image_storage, **_reply_kwargs())
                        logger.info("[TELEGRAMBOT] sendImage url={}, receiver={}".format(img_url, receiver))
                    return
                if hasattr(response, 'candidates'):
                    #获取网址
                    parts = response.candidates[0].content.parts
                    grouding_metadata = response.candidates[0].grounding_metadata
                    if parts is None:
                        finish_reason = response.candidates[0].finish_reason
                        logger.error("[TELEGRAMBOT] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, finish_reason))
                        await self.application.bot.send_message(chat_id=receiver, text=receiver, **_reply_kwargs())
                        logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(reply.content, receiver))
                    elif parts is not None:
                        reply_text = "\n".join(part.text for part in parts)
                        safe_reply_text = html.escape(reply_text)
                    if grouding_metadata is not None:
                        inline_url = self.get_search_sources(grouding_metadata)
                    reply_content = safe_reply_text + "\n\n" + inline_url
                    await self.application.bot.send_message(
                        chat_id=receiver, 
                        text=reply_content,
                        parse_mode='HTML',
                        disable_web_page_preview=True,
                        **_reply_kwargs(),
                    )
                    logger.info("[TELEGRAMBOT_{}] sendMsg={}, receiver={}".format(const.GEMINI_2_FLASH_IMAGE_GENERATION, reply_content, receiver))
                elif isinstance(response, str):
                    img_url = response
                    logger.debug(f"[TELEGRAMBOT] start download image, img_url={img_url}")
                    pic_res = requests.get(img_url, stream=True)
                    image_storage = io.BytesIO()
                    size = 0
                    for block in pic_res.iter_content(1024):
                        size += len(block)
                        image_storage.write(block)
                    logger.info(f"[TELEGRAMBOT] download image success, size={size}, img_url={img_url}")
                    image_storage.seek(0)
                    await self.application.bot.send_photo(chat_id=receiver, photo=image_storage, **_reply_kwargs())
                    logger.info("[TELEGRAMBOT] sendImage url={}, receiver={}".format(img_url, receiver))
                elif hasattr(response, 'content'):
                    # ——— Claude web_search response ———
                    import html as html_module

                    # 提取文本
                    reply_text = ""
                    for block in reply.content.content:
                        if hasattr(block, "text"):
                            reply_text += block.text

                    # markdown 转 HTML：** → <b>，* → <i>
                    import re
                    reply_text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', reply_text)
                    reply_text = re.sub(r'\*(.+?)\*', r'<i>\1</i>', reply_text)
                    safe_reply_text = html_module.escape(reply_text)
                    # escape 之后 bold/italic 标签也被转义了，还原回来
                    safe_reply_text = safe_reply_text.replace('&lt;b&gt;', '<b>').replace('&lt;/b&gt;', '</b>')
                    safe_reply_text = safe_reply_text.replace('&lt;i&gt;', '<i>').replace('&lt;/i&gt;', '</i>')

                    # 提取引用来源
                    inline_url = self._get_claude_search_sources(reply.content)

                    reply_content = safe_reply_text
                    if inline_url:
                        reply_content += f"\n\n{inline_url}"

                    await self.application.bot.send_message(
                        chat_id=receiver,
                        text=reply_content,
                        parse_mode='HTML',
                        disable_web_page_preview=True,
                        **_reply_kwargs(),
                    )
                    logger.info(f"[TELEGRAMBOT_CLAUDE] sendMsg={reply_content}, receiver={receiver}")
            elif reply.type == ReplyType.IMAGE:  # 从文件读取图片
                response = reply.content
                if hasattr(response, "candidates"):
                    parts = response.candidates[0].content.parts
                    if parts is None:
                        finish_reason = response.candidates[0].finish_reason
                        logger.error("[TELEGRAMBOT] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, finish_reason))
                        await self.application.bot.send_message(chat_id=receiver, text=const.ERROR_RESPONSE, **_reply_kwargs())
                        logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(reply.content, receiver))
                    else:
                        for part in parts:
                            if part.text:
                                reply_text = part.text
                                await self.application.bot.send_message(chat_id=receiver, text=reply_text, **_reply_kwargs())
                                logger.info("[TELEGRAMBOT_{}] sendMsg={}, receiver={}".format(self._get_current_image_model_id(receiver), reply_text, receiver))
                            elif part.inline_data:
                                image_bytes = part.inline_data.data
                                image = BytesIO(image_bytes)
                                logger.info(f"[TELEGRAMBOT_{self._get_current_image_model_id(receiver)}] reply={image}")
                                image.seek(0)
                                await self.application.bot.send_photo(chat_id=receiver, photo=image, **_reply_kwargs())
                                logger.info("[TELEGRAMBOT_{}] sendMsg={}, receiver={}".format(self._get_current_image_model_id(receiver), image, receiver))
                else:
                    response.seek(0)
                    await self.application.bot.send_photo(chat_id=receiver, photo=response, **_reply_kwargs())
                    logger.info("[TELEGRAMBOT_{}] sendImage binary, receiver={}".format(self._get_current_image_model_id(receiver), receiver))
            elif reply.type == ReplyType.FILE:  # 新增文件回复类型
                file_pathes = reply.content['function_response']['file_pathes']
                reply_text = escape(reply.content['reply_text'])
                for file_path in file_pathes:
                    with open(file_path, "rb") as f:
                        await self.application.bot.send_document(chat_id=receiver, document=f, **_reply_kwargs())
                    logger.info("[TELEGRAMBOT] sendFile={}, receiver={}".format(file_path, receiver))
                await self.application.bot.send_message(chat_id=receiver, text=reply_text, **_reply_kwargs())
                logger.info("[TELEGRAMBOT] sendMsg={}, receiver={}".format(reply_text, receiver))
            elif reply.type == ReplyType.VIDEO:  # 新增视频回复类型
                video_model_id = self._get_current_video_model_id(receiver)
                response = reply.content
                video_storage = io.BytesIO(response.video_bytes)
                video_storage.seek(0)
                await self.application.bot.send_video(
                    chat_id=receiver,
                    video=video_storage,
                    read_timeout=120,
                    write_timeout=120,
                    connect_timeout=60,
                    **_reply_kwargs(),
                )
                logger.info("[TELEGRAMBOT_{}] sendVideo binary, receiver={}".format(video_model_id, receiver))
            elif reply.type == ReplyType.VIDEO_URL:  # 新增视频URL回复类型
                video_duration = reply.content[0]
                video_url = reply.content[1]
                video_model_id = self._get_current_video_model_id(receiver)
                await self.application.bot.send_document(
                    chat_id=receiver, 
                    document=video_url, 
                    read_timeout=120,
                    write_timeout=120,
                    connect_timeout=60,
                    **_reply_kwargs(),
                )
                logger.info("[TELEGRAMBOT_{}] sendVideo url={}, duration={}, receiver={}".format(video_model_id, video_url, video_duration, receiver))
            elif reply.type == ReplyType.STREAM:
                generator = reply.content
                draft_id = abs(hash(str(receiver))) % (10**9) + 1
                full_text = ""
                for chunk in generator:
                    if not isinstance(chunk, str):
                        # 判断是 Gemini response 还是 Claude final_message
                        if hasattr(chunk, 'candidates'):
                            # Gemini grounding_metadata
                            grounding_metadata = getattr(
                                chunk.candidates[0], "grounding_metadata", None
                            ) if chunk.candidates else None
                            if grounding_metadata and grounding_metadata.grounding_chunks:
                                inline_url = self.get_search_sources(grounding_metadata)
                                if inline_url:
                                    full_text += f"\n\n{inline_url}"
                        else:
                            # Claude final_message citations
                            inline_url = self._get_claude_search_sources(chunk)
                            if inline_url:
                                full_text += f"\n\n{inline_url}"
                    else:
                        full_text += chunk

                    try:
                        await self.application.bot.send_message_draft(
                            chat_id=receiver,
                            draft_id=draft_id,
                            text=full_text + " ▍",
                            parse_mode='HTML'
                        )

                    except Exception as draft_err:
                        logger.warning(f"[TELEGRAMBOT_STREAM] send_message_draft 失败（忽略）: {draft_err}")

                # 流结束，finalize：发最终完整消息，draft 自动消失
                await self.application.bot.send_message(
                    chat_id=receiver,
                    text=full_text,
                    parse_mode='HTML',
                    disable_web_page_preview=True,
                    **_reply_kwargs(),
                )
                logger.info(f"[TELEGRAMBOT_STREAM] stream 发送完成, sendMsg={full_text}, receiver={receiver}")
        except Exception as e:
            logger.error("[TELEGRAMBOT] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, e))
            # 发送失败时，尝试给用户回个错误提示
            await self.application.bot.send_message(chat_id=receiver, text=const.ERROR_RESPONSE, **_reply_kwargs())

    def get_search_sources(self, grounding_metadata):
        """
        Get search sources from the response Grounded with Google Search
        """
        sources = []
        ground_chunks = grounding_metadata.grounding_chunks
        for i, ground_chunk in enumerate(ground_chunks):
            title = ground_chunk.web.title
            uri = ground_chunk.web.uri

            #【HTML 转义】
            #title = escape(title)
            #safe_title = escape_markdown(title, version=2)
            #source = f'{i+1}\\. [{safe_title}]({uri})'

            # 【HTML 转义】非常重要！防止 title 里包含 < 或 > 导致 HTML 解析报错
            # 在 python 3 中可以使用 html.escape
            safe_title = html.escape(title)
            
            # 组装 HTML 格式: <a href="url">标题</a>
            source = f'{i+1}. <a href="{uri}">{safe_title}</a>'
            
            sources.append(source)
        inline_url = '\n'.join(sources)
        return inline_url
    
    def _get_claude_search_sources(self, response) -> str:
        """从 Claude web_search response 的 TextBlock.citations 提取来源，返回 HTML 格式"""
        import html as html_module
        seen_urls = set()
        sources = []
        i = 1
        for block in response.content:
            if block.type != "text":
                continue
            citations = getattr(block, "citations", None)
            if not citations:
                continue
            for citation in citations:
                url = getattr(citation, "url", None)
                title = getattr(citation, "title", None) or url
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    safe_title = html_module.escape(title)
                    sources.append(f'{i}. <a href="{url}">{safe_title}</a>')
                    i += 1
        return '\n'.join(sources)
