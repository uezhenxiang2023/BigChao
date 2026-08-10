"""
飞书通道接入

@author uezhenxiang2023
@Date 2025/05/11
"""

# -*- coding=utf-8 -*-
import cv2
import io, json, os, uuid, requests, threading, re, time
from io import BytesIO
from flask import Flask, send_file, abort
from urllib.parse import urlparse, quote

from channel.feishu.feishu_message import FeishuMessage
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from common.log import logger
from common.media_store import build_public_media_url
from common.singleton import singleton
from common import const
from common.expired_dict import ExpiredDict
from common.tool_button import tool_state
from common.model_status import model_state
from common.tmp_dir import TmpDir, get_response_dir
from config import conf
from bridge.context import ContextType
from channel.chat_channel import ChatChannel
from channel.chat_message import ChatMessage
import lark_oapi as lark
from lark_oapi.api.im.v1 import *
from lark_oapi.adapter.flask import *


@singleton
class FeiShuChanel(ChatChannel):
    def __init__(self, session_id=None):
        super().__init__()
        self.app_id = conf().get('feishu_app_id')
        self.app_secret = conf().get('feishu_app_secret')
        self.encrypt_key = conf().get('feishu_encrypt_key')
        self.verification_token = conf().get('feishu_verify_token')
        self.websocket = conf().get('feishu_websocket')
        self.CLIENT_ENCRYPT_KEY = "" if self.websocket is True else self.encrypt_key
        self.CLIENT_VERIFICATION_TOKEN = "" if self.websocket is True else self.verification_token
        # 初始化 Flask app
        self.app = Flask(__name__)
        # 注册路由
        self.app.route("/", methods=["POST"])(self.handle_webhook_event)
        self.app.route("/tmp_media/<path:relative_path>", methods=["GET"])(self.serve_tmp_media)
        self.webhook_port = conf().get('feishu_webhook_port')
        # Register event handler.
        self.event_handler = (
            lark.EventDispatcherHandler.builder(self.CLIENT_ENCRYPT_KEY, self.CLIENT_VERIFICATION_TOKEN)
            .register_p2_im_message_receive_v1(self.do_p2_im_message_receive_v1)
            .register_p2_application_bot_menu_v6(self.do_p2_application_bot_menu_v6)
            .build()
        )
        # Create LarkClient object for requesting OpenAPI, and create LarkWSClient object for receiving events using long connection.
        self.client = (
            lark.Client.builder()
            .app_id(self.app_id)
            .app_secret(self.app_secret)
            .build()
            )
        self.wsClient = lark.ws.Client(
            self.app_id,
            self.app_secret,
            event_handler=self.event_handler,
            log_level=lark.LogLevel.INFO,
        )
        # Feishu can retry webhook events after a long-running image/video request.
        # Keep message ids long enough to cover delayed retries from async generation.
        self._recent_message_events = ExpiredDict(conf().get("feishu_event_dedupe_seconds", 60 * 60 * 24))
        self._recent_message_events_lock = threading.Lock()

    def _get_current_image_model_id(self, user_id):
        return model_state.get_image_model(user_id).upper()

    def _get_current_video_model_id(self, user_id):
        return model_state.get_video_state(user_id).upper()

    # Register event handler to handle received messages.
    # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/events/receive
    def do_p2_im_message_receive_v1(self, data: P2ImMessageReceiveV1) -> None:
        # Print tool_button stasus to console
        event = data.event
        message = event.message
        sender = event.sender.sender_id
        toUserName = sender.open_id
        event_id = getattr(data.header, "event_id", None)
        message_create_time = getattr(message, "create_time", None)
        max_event_age_seconds = conf().get("feishu_event_max_age_seconds", 60 * 10)
        if message_create_time and max_event_age_seconds:
            try:
                event_age_seconds = (time.time() * 1000 - int(message_create_time)) / 1000
                if event_age_seconds > max_event_age_seconds:
                    logger.warning(
                        "[Lark-event] stale event ignored, event_id=%s, message_id=%s, create_time=%s, age=%.1fs, max_age=%ss, open_id=%s",
                        event_id,
                        message.message_id,
                        message_create_time,
                        event_age_seconds,
                        max_event_age_seconds,
                        toUserName,
                    )
                    return
            except (TypeError, ValueError):
                logger.warning(
                    "[Lark-event] invalid message create_time, event_id=%s, message_id=%s, create_time=%s, open_id=%s",
                    event_id,
                    message.message_id,
                    message_create_time,
                    toUserName,
                )
        dedupe_key = event_id or getattr(message, "message_id", None)
        if dedupe_key:
            with self._recent_message_events_lock:
                if dedupe_key in self._recent_message_events:
                    logger.warning(
                        "[Lark-event] duplicate event ignored, event_id=%s, message_id=%s, create_time=%s, open_id=%s",
                        event_id,
                        message.message_id,
                        message.create_time,
                        toUserName,
                    )
                    return
                self._recent_message_events[dedupe_key] = {
                    "event_id": event_id,
                    "message_id": message.message_id,
                    "create_time": message.create_time,
                    "open_id": toUserName,
                }
        logger.info(
            "[Lark-event] event_id=%s, message_id=%s, parent_id=%s, create_time=%s, chat_type=%s, chat_id=%s, open_id=%s, message_type=%s",
            event_id,
            message.message_id,
            getattr(message, "parent_id", None),
            message.create_time,
            message.chat_type,
            message.chat_id,
            toUserName,
            message.message_type,
        )
        logger.info(
            f'[Lark-search] is {tool_state.get_search_state(toUserName)},\
            [Lark-image_mode] is {model_state.get_image_mode(toUserName)},\
            [Lark-image_model] is {model_state.get_image_model(toUserName)},\
            [Lark-video_mode] is {model_state.get_video_mode(toUserName)},\
            [Lark-print] is {tool_state.get_print_state(toUserName)},\
            [Lark-breakdown] is {tool_state.get_breakdown_state(toUserName)},\
            requester={toUserName}'
        )
        if data.event.message.chat_type == "p2p":
            self.handler_single_msg(data.event)

        elif data.event.message.chat_type == "group":
            self.handler_group_msg(data.event)
            """request: ReplyMessageRequest = (
                ReplyMessageRequest.builder()
                .message_id(data.event.message.message_id)
                .request_body(
                    ReplyMessageRequestBody.builder()
                    .content(content)
                    .msg_type("text")
                    .build()
                )
                .build()
            )
            # Reply to messages using reply OpenAPI
            # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/reply
            response: ReplyMessageResponse = self.client.im.v1.message.reply(request)
            if not response.success():
                raise Exception(
                    f"client.im.v1.message.reply failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
                )"""
    # Register event handler to handle bot menu.
    # https://open.feishu.cn/document/client-docs/bot-v3/events/menu
    def do_p2_application_bot_menu_v6(self, data: lark.application.v6.P2ApplicationBotMenuV6) -> None:
        logger.info(f'[ do_p2_application_bot_menu_v6 access ], data: {lark.JSON.marshal(data, indent=4)}')
        event_key = data.event.event_key
        open_id = data.event.operator.operator_id.open_id
        if event_key == 'Generation':
            model_state.toggle_image_mode(open_id, event_key)
            self.send_text(f"[INFO]\n图片模式已切换为：生成新图模式", open_id)
            logger.info(f'[Lark] switch image mode to {event_key}, requester={open_id}')
        elif event_key == 'Editing':
            model_state.toggle_image_mode(open_id, event_key)
            self.send_text(f"[INFO]\n图片模式已切换为：编辑现有图模式", open_id)
            logger.info(f'[Lark] switch image mode to {event_key}, requester={open_id}')
        elif event_key == 'FirstLast':
            model_state.toggle_video_mode(open_id, event_key)
            self.send_text(f"[INFO]\n视频模式已切换为：首尾帧生成模式", open_id)
            logger.info(f'[Lark] switch video mode to {event_key}, requester={open_id}')
        elif event_key == 'Reference':
            model_state.toggle_video_mode(open_id, event_key)
            self.send_text(f"[INFO]\n视频模式已切换为：参考图生成模式", open_id)
            logger.info(f'[Lark] switch video mode to {event_key}, requester={open_id}')
        elif event_key == 'VideoEdit':
            model_state.toggle_video_mode(open_id, event_key)
            self.send_text(f"[INFO]\n视频模式已切换为：视频编辑模式", open_id)
            logger.info(f'[Lark] switch video mode to {event_key}, requester={open_id}')
        elif event_key == 'VideoExtend':
            model_state.toggle_video_mode(open_id, event_key)
            self.send_text(f"[INFO]\n视频模式已切换为：视频延长模式", open_id)
            logger.info(f'[Lark] switch video mode to {event_key}, requester={open_id}')
        elif event_key == 'searching':
            self.search(open_id)
        elif event_key == 'printing':
            self.print(open_id)
        elif event_key == 'breakdowning':
            self.breakdown(open_id)
        elif event_key == 'SeedDream':
            model_state.toggle_image_model(open_id, const.DOUBAO_SEEDREAM_5)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.DOUBAO_SEEDREAM_5}", open_id)
            logger.info(f'[Lark] switch image model to {const.DOUBAO_SEEDREAM_5.upper()}, requester={open_id}')
        elif event_key == 'KlingImage':
            model_state.toggle_image_model(open_id, const.KLING_V3_OMNI)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.KLING_V3_OMNI}", open_id)
            logger.info(f'[Lark] switch image model to {const.KLING_V3_OMNI.upper().upper()}, requester={open_id}')
        elif event_key == 'GPTImage':
            model_state.toggle_image_model(open_id, const.GPT_IMAGE_2)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.GPT_IMAGE_2}", open_id)
            logger.info(f'[Lark] switch image model to {const.GPT_IMAGE_2.upper()}, requester={open_id}')
        elif event_key == 'NanoBanana':
            model_state.toggle_image_model(open_id, const.GEMINI_31_FLASH_IMAGE_PREVIEW)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.GEMINI_31_FLASH_IMAGE_PREVIEW}", open_id)
            logger.info(f'[Lark] switch image model to {const.GEMINI_31_FLASH_IMAGE_PREVIEW.upper()}, requester={open_id}')
        elif event_key == 'GrokImage':
            model_state.toggle_image_model(open_id, const.GROK_IMAGINE_IMAGE_QUALITY)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.GROK_IMAGINE_IMAGE_QUALITY}", open_id)
            logger.info(f'[Lark] switch image model to {const.GROK_IMAGINE_IMAGE_QUALITY.upper()}, requester={open_id}')
        elif event_key == 'LumaImage':
            model_state.toggle_image_model(open_id, const.UNI_1)
            self.send_text(f"[INFO]\n图片模型已切换为：{const.UNI_1}", open_id)
            logger.info(f'[Lark] switch image model to {const.UNI_1.upper()}, requester={open_id}')
        elif event_key == 'SeedDance':
            model_state.toggle_video_model(open_id, const.DOUBAO_SEEDANCE_20)
            self.send_text(f"[INFO]\n视频模型已切换为：{const.DOUBAO_SEEDANCE_20}", open_id)
            logger.info(f'[Lark] switch video model to {const.DOUBAO_SEEDANCE_20.upper()}, requester={open_id}')
        elif event_key == 'KlingVideo':
            model_state.toggle_video_model(open_id, const.KLING_V3_OMNI)
            self.send_text(f"[INFO]\n视频模型已切换为：{const.KLING_V3_OMNI}", open_id)
            logger.info(f'[Lark] switch video model to {const.KLING_V3_OMNI.upper()}, requester={open_id}')
        elif event_key == 'Veo':
            model_state.toggle_video_model(open_id, const.VEO_31)
            self.send_text(f"[INFO]\n视频模型已切换为：{const.VEO_31}", open_id)
            logger.info(f'[Lark] switch video model to {const.VEO_31.upper()}, requester={open_id}')
        elif event_key == 'GrokVideo':
            model_state.toggle_video_model(open_id, const.GROK_IMAGINE_VIDEO)
            self.send_text(f"[INFO]\n视频模型已切换为：{const.GROK_IMAGINE_VIDEO}", open_id)
            logger.info(f'[Lark] switch video model to {const.GROK_IMAGINE_VIDEO.upper()}, requester={open_id}')
        elif event_key == 'LumaVideo':
            model_state.toggle_video_model(open_id, const.LUMA_RAY_32)
            self.send_text(f"[INFO]\n视频模型已切换为：{const.LUMA_RAY_32}", open_id)
            logger.info(f'[Lark] switch video model to {const.LUMA_RAY_32.upper()}, requester={open_id}')

    def handle_webhook_event(self):
        """Webhook event handler"""
        logger.debug(f"Current thread: {threading.current_thread().name}")
        try:
            # 获取请求数据
            event_data = parse_req()
            resp = self.event_handler.do(event_data)
            return parse_resp(resp)
        except Exception as e:
            logger.error(f"[Lark]Error handling webhook event: {str(e)}")
            return {"[Lark]error": str(e)}, 500
    
    def search(self, toUserName) -> None:
        """
        This function handles the search menu
        """
        if tool_state.get_search_state(toUserName):
            text = "[INFO]\n联网搜索功能已关闭，可以在消息框输入#search或点击输入框上方的‘其他工具’菜单随时开启。"
        else:
            text = "[INFO]\n联网搜索功能已开启。"
        tool_state.toggle_searching(toUserName)
        self.send_text(text, toUserName)
        logger.info(f'[Lark]{text} requester={toUserName}')
        
    
    def image(self, toUserName) -> None:
        """
        This function handles image menu
        """
        image_model = model_state.get_image_model(toUserName)
        image_model_name = image_model.upper()
        if tool_state.get_image_state(toUserName):
            text = "[INFO]\n图片编辑功能已关闭，可以在消息框输入#image或点击输入框上方的‘其他工具’菜单随时开启。"
        else:
            text = f"[INFO]\n图片编辑功能已开启，当前默认模型：{image_model_name}，切换请选择图片模型菜单。"
        tool_state.toggle_imaging(toUserName)
        self.send_text(text, toUserName)
        logger.info(f'[Lark_{image_model}]{text} requester={toUserName}')

    def edit(self, toUserName) -> None:
        """
        This function handles edit menu
        """
        video_model = model_state.get_video_state(toUserName)
        video_model_name = video_model.upper()
        if tool_state.get_edit_state(toUserName):
            text = "[INFO]\n视频编辑功能已关闭，可以在消息框输入#edit或点击输入框上方的‘其他工具’菜单随时开启。"
        else:
            text = f"[INFO]\n视频编辑功能已开启，当前默认模型：{video_model_name}，切换请选择视频模型菜单。"
        tool_state.toggle_editing(toUserName)
        self.send_text(text, toUserName)
        logger.info(f'[Lark_{video_model_name}]{text} requester={toUserName}')

    def print(self, toUserName) -> None:
        """
        This function handles the print menu
        """
        if tool_state.get_print_state(toUserName):
            text = "[INFO]\n剧本排版功能已关闭，可以在消息框输入#print或点击输入框上方的‘剧本排版’菜单随时开启。"
        else:
            text = "[INFO]\n剧本排版功能已开启,请先在输入框中点击“+”号上传pdf格式的剧本，然后在对话框中输入编剧姓名。\n我会按照好莱坞编剧工会的标准格式进行排版，让您的剧本看起来更专业、读起来更舒服，大大提升获得‘绿灯’的几率。"
        tool_state.toggle_printing(toUserName)
        status = tool_state.get_print_state(toUserName)
        self.send_text(text, toUserName)
        logger.info(f'[Lark]printing_stasus={status},{text} requester={toUserName}')

    def breakdown(self, toUserName) -> None:
        """
        This function handles breakdown menu
        """
        if tool_state.get_breakdown_state(toUserName):
            text = "[INFO]\n拆解顺分场表功能已关闭，可以在消息框输入#breakdown或点击输入框上方的‘顺分场表’菜单随时开启。"
        else:
            text = "[INFO]\n拆解顺分场表功能已开启。"
        tool_state.toggle_breakdowning(toUserName)
        self.send_text(text, toUserName)
        logger.info(f'[Lark]{text} requester={toUserName}')

    def main(self):
        if self.websocket is True:
            # 使用websocket长链接接收飞书事件.
            self.wsClient.start()
        else:
            # 使用webhook模式，通过本地服务器接收飞书事件
            logger.info(f"Starting webhook server on port {self.webhook_port}")
            self.app.run(
                host='0.0.0.0',  # 允许外部访问
                port=self.webhook_port,
                debug=False  # 生产环境建议设为False
            )
    def startup(self):
        self.main()

    def handler_single_msg(self, msg):
        try:
            cmsg = FeishuMessage(msg, False)
        except NotImplementedError as e:
            logger.debug("[Lark]single message {} skipped: {}".format(msg["MsgId"], e))
            error_reply = e
            self.send_text(error_reply, msg.chat_id)
            return None
        try:
            self.handle_single(cmsg)
        except Exception as e:
            logger.exception(
                "[Lark] handler_single_msg failed, message_id=%s, content=%s, ctype=%s, sender=%s",
                getattr(cmsg, "msg_id", None),
                getattr(cmsg, "content", None),
                getattr(cmsg, "ctype", None),
                getattr(cmsg, "from_user_id", None),
            )
            raise
        return None

    def handler_group_msg(self, msg):
        try:
            cmsg = FeishuMessage(msg, True)
        except NotImplementedError as e:
            logger.debug("[Lark]group message {} skipped: {}".format(msg["MsgId"], e))
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
            logger.debug("[Lark]receive voice msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.IMAGE:
            logger.debug("[Lark]receive image msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.VIDEO:
            logger.debug("[Lark]receive video msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.SHARING:
            logger.debug("[Lark]receive url msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.PATPAT:
            logger.debug("[Lark]receive patpat msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.FILE:
            logger.debug("[Lark]receive file msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.TEXT:
            logger.debug("[Lark]receive text msg: {}, cmsg={}".format(cmsg.content, cmsg))
        else:
            logger.debug("[Lark]receive msg: {}, cmsg={}".format(cmsg.content, cmsg))
        try:
            context = self._compose_context(cmsg.ctype, cmsg.content, isgroup=False, msg=cmsg)
        except Exception as e:
            logger.exception(
                "[Lark] _compose_context failed, message_id=%s, content=%s, ctype=%s, sender=%s",
                getattr(cmsg, "msg_id", None),
                getattr(cmsg, "content", None),
                getattr(cmsg, "ctype", None),
                getattr(cmsg, "from_user_id", None),
            )
            raise
        if context and cmsg.ctype == ContextType.VIDEO:
            public_url = self.build_public_media_url(cmsg.content)
            if public_url:
                context["video_public_url"] = public_url
        if context and hasattr(cmsg, "get_quoted_image_path"):
            quoted_image_path = cmsg.get_quoted_image_path()
            if quoted_image_path:
                context["quoted_image_path"] = quoted_image_path
        if context and hasattr(cmsg, "get_quoted_video_path"):
            quoted_video_path = cmsg.get_quoted_video_path()
            if quoted_video_path:
                context["quoted_video_path"] = quoted_video_path
                quoted_video_public_url = self.build_public_media_url(quoted_video_path)
                if quoted_video_public_url:
                    context["quoted_video_public_url"] = quoted_video_public_url
        if context and hasattr(cmsg, "get_quoted_file_path"):
            quoted_file_path = cmsg.get_quoted_file_path()
            if quoted_file_path:
                context["quoted_file_path"] = quoted_file_path
        if context and cmsg.ctype == ContextType.FILE:
            suffix = os.path.splitext(cmsg.content)[1].lstrip(".").lower()
            if suffix in const.VIDEO:
                public_url = self.build_public_media_url(cmsg.content)
                if public_url:
                    context["video_public_url"] = public_url
        if context:
            try:
                self.produce(context)
            except Exception as e:
                logger.exception(
                    "[Lark] produce failed, session_id=%s, receiver=%s, content=%s, ctype=%s",
                    context.get("session_id"),
                    context.get("receiver"),
                    context.content,
                    context.type,
                )
                raise

    def handle_group(self, cmsg: ChatMessage):
        if cmsg.ctype == ContextType.VOICE:
            if conf().get("group_speech_recognition") != True:
                return
            logger.debug("[Lark]receive voice for group msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.IMAGE:
            logger.debug("[Lark]receive image for group msg: {}".format(cmsg.content))
        elif cmsg.ctype in [ContextType.JOIN_GROUP, ContextType.PATPAT, ContextType.ACCEPT_FRIEND, ContextType.EXIT_GROUP]:
            logger.debug("[Lark]receive note msg: {}".format(cmsg.content))
        elif cmsg.ctype == ContextType.TEXT:
            logger.debug("[Lark]receive group msg: {}, cmsg={}".format(cmsg.content, cmsg))
            pass
        elif cmsg.ctype == ContextType.FILE:
            logger.debug(f"[Lark]receive attachment msg, file_name={cmsg.content}")
        else:
            logger.debug("[Lark]receive group msg: {}".format(cmsg.content))
        context = self._compose_context(cmsg.ctype, cmsg.content, isgroup=True, msg=cmsg)
        if context and cmsg.ctype == ContextType.VIDEO:
            public_url = self.build_public_media_url(cmsg.content)
            if public_url:
                context["video_public_url"] = public_url
        if context and hasattr(cmsg, "get_quoted_image_path"):
            quoted_image_path = cmsg.get_quoted_image_path()
            if quoted_image_path:
                context["quoted_image_path"] = quoted_image_path
        if context and hasattr(cmsg, "get_quoted_video_path"):
            quoted_video_path = cmsg.get_quoted_video_path()
            if quoted_video_path:
                context["quoted_video_path"] = quoted_video_path
                quoted_video_public_url = self.build_public_media_url(quoted_video_path)
                if quoted_video_public_url:
                    context["quoted_video_public_url"] = quoted_video_public_url
        if context and hasattr(cmsg, "get_quoted_file_path"):
            quoted_file_path = cmsg.get_quoted_file_path()
            if quoted_file_path:
                context["quoted_file_path"] = quoted_file_path
        if context and cmsg.ctype == ContextType.FILE:
            suffix = os.path.splitext(cmsg.content)[1].lstrip(".").lower()
            if suffix in const.VIDEO:
                public_url = self.build_public_media_url(cmsg.content)
                if public_url:
                    context["video_public_url"] = public_url
        if context:
            self.produce(context)

    # 统一的发送函数，每个Channel自行实现，根据reply的type字段发送不同类型的消息
    def send(self, reply: Reply, context: Context):
        receiver = context["receiver"]
        error_response = "网络有点小繁忙，请过几秒再试一试，给您带来不便，大超子深表歉意"
        if reply.type == ReplyType.TEXT:
            try:
                self.send_text(reply.content, toUserName=receiver)
                logger.info("[Lark] sendMsg={}, receiver={}".format(reply, receiver))
            except Exception as e:
                logger.error("[Lark] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, e))
                self.send_text(error_response, toUserName=receiver)
                logger.info("[Lark] sendMsg={}, receiver={}".format(error_response, receiver))
        elif reply.type == ReplyType.ERROR:
            error_text = reply.content if reply.content else error_response
            self.send_text(error_text, toUserName=receiver)
            logger.info("[Lark] sendMsg={}, receiver={}".format(error_text, receiver))
        elif reply.type == ReplyType.INFO:
            self.send_text(reply.content, toUserName=receiver)
            logger.info("[Lark] sendMsg={}, receiver={}".format(reply, receiver))
        elif reply.type == ReplyType.VOICE:
            self.send_file(reply.content, toUserName=receiver)
            logger.info("[Lark] sendFile={}, receiver={}".format(reply.content, receiver))
        elif reply.type == ReplyType.IMAGE_URL:  # 获取网络资源
            response = reply.content
            if not isinstance(response, str):
                if isinstance(response, list):
                    for image_url in response:
                        image_reply = Reply(ReplyType.IMAGE_URL, image_url)
                        self.send(image_reply, context)
                    return
                if hasattr(response, 'candidates'):
                    # gemini 获取引用的网址链接
                    parts = response.candidates[0].content.parts
                    grouding_metadata = response.candidates[0].grounding_metadata
                    if parts is None:
                        finish_reason = response.candidates[0].finish_reason
                        logger.error("[Lark] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, finish_reason))
                        self.send_text(const.ERROR_RESPONSE, toUserName=receiver)
                        logger.info("[Lark] sendMsg={}, receiver={}".format(reply.content, receiver))
                    elif parts is not None:
                        reply_text = "\n".join(part.text for part in parts)
                    if grouding_metadata is not None:
                        inline_url = self.get_search_sources(grouding_metadata)
                    reply_content = reply_text + "\n\n" + inline_url

                    self.send_text(reply_content, receiver)
                    logger.info("[Lark] sendMsg={}, receiver={}".format(reply_content, receiver))

                elif hasattr(response, 'content') and hasattr(response, 'model'):
                    # claude 获取引用的网址链接
                    reply_text = ""
                    for block in response.content:
                        if hasattr(block, "text"):
                            reply_text += block.text

                    inline_url = self._get_claude_search_sources(response)
                    reply_content = reply_text
                    if inline_url:
                        reply_content += f"\n\n{inline_url}"

                    self.send_text(reply_content, receiver)
                    logger.info("[Lark_CLAUDE] sendMsg={}, receiver={}".format(reply_content, receiver))
            else:
                # 下载图片
                file_name = self.extract_image_filename(response)
                logger.debug(f"[Lark] start download image, img_url={response}")
                pic_res = requests.get(response, stream=True)
                image_storage = io.BytesIO()
                size = 0
                for block in pic_res.iter_content(1024):
                    size += len(block)
                    image_storage.write(block)
                logger.info(f"[Lark] download image success, size={size}, img_url={response}")
                image_storage.seek(0)
                user_dir = get_response_dir(conf().get("channel_type", "feishu"), receiver)
                image_path = user_dir + file_name
                with open(image_path, 'wb') as f:
                    f.write(image_storage.read())
                self.send_image(image_path, toUserName=receiver, source_url=response)
                logger.info("[Lark] sendImage url={}, receiver={}".format(response, receiver))
        elif reply.type == ReplyType.IMAGE:  # 从文件读取图片
            image_model_id = self._get_current_image_model_id(receiver)
            response = reply.content
            if hasattr(response, "candidates"):
                parts = response.candidates[0].content.parts
                if parts is None:
                    finish_reason = response.candidates[0].finish_reason
                    logger.error("[Lark] sendMsg error, reply={}, receiver={}, error={}".format(reply, receiver, finish_reason))
                    self.send_text(const.ERROR_RESPONSE, toUserName=receiver)
                    logger.info("[Lark] sendMsg={}, receiver={}".format(reply.content, receiver))
                else:
                    for part in parts:
                        if part.text:
                            reply_text = part.text
                            self.send_text(reply_text, receiver)
                            logger.info("[Lark_{}] sendMsg={}, receiver={}".format(image_model_id, part.text, receiver))
                        elif part.inline_data:
                            image_type = part.inline_data.mime_type.split('/')[-1]
                            image = BytesIO(part.inline_data.data)
                            logger.info(f"[Lark_{image_model_id}] reply={image}")
                            image.seek(0)
                            user_dir = get_response_dir(conf().get("channel_type", "feishu"), receiver)
                            response_uuid = str(uuid.uuid4())
                            image_path = user_dir + response_uuid + '.' + image_type
                            with open(image_path, 'wb') as f:
                                f.write(image.read())
                            self.send_image(image_path, receiver)
                            logger.info("[Lark_{}] sendMsg={}, receiver={}".format(image_model_id, image, receiver))
            else:
                image = response
                image.seek(0)
                image_bytes = image.read()
                image.seek(0)
                try:
                    from PIL import Image

                    with Image.open(BytesIO(image_bytes)) as pil_image:
                        image_type = (pil_image.format or "PNG").lower()
                except Exception:
                    image_type = "png"
                user_dir = get_response_dir(conf().get("channel_type", "feishu"), receiver)
                response_uuid = str(uuid.uuid4())
                image_path = user_dir + response_uuid + '.' + image_type
                with open(image_path, 'wb') as f:
                    f.write(image_bytes)
                self.send_image(image_path, receiver)
                logger.info("[Lark_{}] sendImage binary receiver={}".format(image_model_id, receiver))
        elif reply.type == ReplyType.FILE:  # 新增文件回复类型
            file_pathes = reply.content['function_response']['file_pathes']
            reply_text = reply.content['reply_text']
            for file_path in file_pathes:
                self.send_file(file_path, toUserName=receiver)
                logger.info("[Lark] sendFile={}, receiver={}".format(file_path, receiver))
            self.send_text(reply_text, receiver)
            logger.info("[Lark] sendMsg={}, receiver={}".format(reply_text, receiver))
        elif reply.type == ReplyType.VIDEO:  # 新增视频回复类型
            video_model_id = self._get_current_video_model_id(receiver)
            video_duration = conf().get("duration_seconds")
            response = reply.content
            file_uri = response.uri
            file_name = re.search(r'/files/([^/:]+)', file_uri).group(1) + '.mp4'
            video_storage = io.BytesIO(response.video_bytes)
            video_path = self.save_media_file(receiver, video_storage, file_name)
            self.send_video(video_duration, video_path, receiver)
            logger.info("[Lark_{}] sendFile video_name={}, receiver={}".format(video_model_id, file_name, receiver))
        elif reply.type == ReplyType.VIDEO_URL:  # 新增视频URL回复类型
            video_model_id = self._get_current_video_model_id(receiver)
            video_duration = reply.content[0]
            video_url = reply.content[1]
            file_name = self.extract_image_filename(video_url)
            logger.debug(f"[Lark_{video_model_id}] start download video, video_url={video_url}")
            video_res = requests.get(video_url, stream=True)
            video_storage = io.BytesIO()
            size = 0
            for block in video_res.iter_content(1024):
                size += len(block)
                video_storage.write(block)
            logger.info(f"[Lark_{video_model_id}] download video success, size={size}, video_url={video_url}")
            video_storage.seek(0)
            video_path = self.save_media_file(receiver, video_storage, file_name)
            self.send_video(video_duration, video_path, receiver)
            logger.info("[Lark_{}] sendVideo url={}, receiver={}".format(video_model_id, video_path, receiver))
    
    def save_media_file(self, receiver, media_storage, file_name):
        """将网页链接中的图片或视频文件存储到本地硬盘"""
        user_dir = get_response_dir(conf().get("channel_type", "feishu"), receiver)
        media_path = user_dir + file_name
        with open(media_path, 'wb') as f:
            f.write(media_storage.read())
        logger.info(f"[Lark] {media_path} downloads to local tmp dir successfully")
        return media_path

    def get_search_sources(self, grounding_metadata):
        """
        Get search sources from the response Grounded with Google Search
        """
        sources = []
        ground_chunks = grounding_metadata.grounding_chunks
        for i, ground_chunk in enumerate(ground_chunks):
            title = ground_chunk.web.title
            uri = ground_chunk.web.uri
            source = f'{i+1}.[{title}]({uri})'
            sources.append(source)
        inline_url = '\n'.join(sources)
        return inline_url
    
    def _get_claude_search_sources(self, response) -> str:
        """从 Claude web_search response 的 TextBlock.citations 提取来源
        飞书用 Markdown 格式链接，不用 HTML"""
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
                    sources.append(f'{i}. [{title}]({url})')
                    i += 1
        return '\n'.join(sources)

    def send_text(self, reply_content, toUserName):
        """
        This function sends a response text message back to the user.
        """
        content = json.dumps(
            {
                "text": reply_content
            }
        )
        request = (
                CreateMessageRequest.builder()
                .receive_id_type("open_id")
                .request_body(
                    CreateMessageRequestBody.builder()
                    .receive_id(toUserName)
                    .msg_type("text")
                    .content(content)
                    .build()
                )
                .build()
            )
        # Use send OpenAPI to send messages
        # https://open.feishu.cn/document/uAjLw4CM/ukTMukTMukTM/reference/im-v1/message/create
        response = self.client.im.v1.chat.create(request)

        if not response.success():
            raise Exception(
                f"client.im.v1.chat.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}"
            )

    def send_image(self, reply_content, toUserName, source_url=None):
        """
        This function sends a response image back to the user.
        """
        # 创建client
        max_image_size = 10 * 1024 * 1024
        image_size = os.path.getsize(reply_content)
        if image_size > max_image_size and source_url:
            reply_text = f"图片过大，请点击链接在网页中下载。\n{source_url}"
            logger.info("[Lark] image too large, send url as text, size={}, receiver={}".format(image_size, toUserName))
            self.send_text(reply_text, toUserName)
            return

        client = self.client
        image_key = self.create_image(reply_content)
        if not image_key:
            logger.error("[Lark] create image failed, image_path={}, receiver={}".format(reply_content, toUserName))
            return
        content = json.dumps(
            {
                "image_key": image_key
            }
        )

        # 生成唯一的UUID
        request_uuid = str(uuid.uuid4())

        # 构造请求对象
        request: CreateMessageRequest = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(toUserName)
                .msg_type("image")
                .content(content)
                .uuid(request_uuid)
                .build()) \
            .build()

        # 发起请求
        response: CreateMessageResponse = client.im.v1.message.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))
    
    def create_image(self, image_path):
        """
        This function uploads image to Lark OpenAPI.
        """
        # 创建client
        client = self.client

        # 构造请求对象
        file = open(image_path, "rb")
        request: CreateImageRequest = CreateImageRequest.builder() \
            .request_body(CreateImageRequestBody.builder()
                .image_type("message")
                .image(file)
                .build()) \
            .build()

        # 发起请求
        response: CreateImageResponse = client.im.v1.image.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.image.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return
        # 处理业务结果
        else:
            lark.logger.info(lark.JSON.marshal(response.data, indent=4))
            return response.data.image_key

    def create_file(self, file_path):
        """
        This function uploads file to Lark OpenAPI.
        """
        # 创建client
        client = self.client
        # 构造请求对象
        filename = os.path.basename(file_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lstrip('.')
        file = open(file_path, "rb")
        request: CreateFileRequest = CreateFileRequest.builder() \
            .request_body(CreateFileRequestBody.builder()
                .file_type(ext)
                .file_name(filename)
                .file(file)
                .build()) \
            .build()

        # 发起请求
        response: CreateFileResponse = client.im.v1.file.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.file.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return

        else:
            # 处理业务结果
            lark.logger.info(lark.JSON.marshal(response.data, indent=4))
            return response.data.file_key
        
    def create_video(self, video_duration, video_path):
        """
        This function uploads video to Lark OpenAPI.
        """
        # 创建client
        client = self.client
        # 构造请求对象
        duration = int(video_duration * 1000)
        filename = os.path.basename(video_path)
        name, ext = os.path.splitext(filename)
        ext = ext.lstrip('.')
        file = open(video_path, "rb")
        request: CreateFileRequest = CreateFileRequest.builder() \
            .request_body(CreateFileRequestBody.builder()
                .file_type(ext)
                .file_name(filename)
                .duration(duration)
                .file(file)
                .build()) \
            .build()

        # 发起请求
        response: CreateFileResponse = client.im.v1.file.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.file.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return

        else:
            # 处理业务结果
            lark.logger.info(lark.JSON.marshal(response.data, indent=4))
            return response.data.file_key

    def send_file(self, reply_content, toUserName):
        """
        This function sends a response file back to the user.
        """
        # 创建client
        client = self.client
        file_key = self.create_file(reply_content)
        content = json.dumps(
            {
                "file_key": file_key
            }
        )

        # 生成唯一的UUID
        request_uuid = str(uuid.uuid4())

        # 构造请求对象
        request: CreateMessageRequest = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(toUserName)
                .msg_type("file")
                .content(content)
                .uuid(request_uuid)
                .build()) \
            .build()

        # 发起请求
        response: CreateMessageResponse = client.im.v1.message.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    def send_video(self, video_duration, video_path, toUserName):
        """
        This function sends a response video back to the user.
        """
        # 创建client
        client = self.client
        file_key = self.create_video(video_duration, video_path)

        # 截取第一帧作为封面
        image_key = self._extract_video_cover(video_path)

        content_dict = {"file_key": file_key}
        if image_key:
            content_dict["image_key"] = image_key
        content = json.dumps(content_dict)

        # 生成唯一的UUID
        request_uuid = str(uuid.uuid4())

        # 构造请求对象
        request: CreateMessageRequest = CreateMessageRequest.builder() \
            .receive_id_type("open_id") \
            .request_body(CreateMessageRequestBody.builder()
                .receive_id(toUserName)
                .msg_type("media")
                .content(content)
                .uuid(request_uuid)
                .build()) \
            .build()

        # 发起请求
        response: CreateMessageResponse = client.im.v1.message.create(request)

        # 处理失败返回
        if not response.success():
            lark.logger.error(
                f"client.im.v1.message.create failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
            return

        # 处理业务结果
        lark.logger.info(lark.JSON.marshal(response.data, indent=4))

    def extract_image_filename(self, url):
        # 解析URL
        parsed_url = urlparse(url)
        # 获取路径部分
        path = parsed_url.path
        # 提取文件名
        filename = os.path.basename(path)
        # 如果文件名包含查询参数，只取问号前的部分
        filename = filename.split('?')[0]
        return filename

    def serve_tmp_media(self, relative_path):
        tmp_root = os.path.abspath(TmpDir().path())
        target_path = os.path.abspath(os.path.join(tmp_root, relative_path))
        if not target_path.startswith(tmp_root + os.sep) and target_path != tmp_root:
            abort(403)
        if not os.path.exists(target_path):
            abort(404)
        return send_file(target_path, conditional=True)

    def build_public_media_url(self, file_path):
        return build_public_media_url(file_path)
    
    def _extract_video_cover(self, video_path: str) -> str | None:
        """截取视频第一帧，上传飞书返回 image_key"""
        try:
            cap = cv2.VideoCapture(video_path)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return None
            cover_path = video_path + '_cover.jpg'
            cv2.imwrite(cover_path, frame)
            image_key = self.create_image(cover_path)
            os.remove(cover_path)
            return image_key
        except Exception as e:
            logger.warning(f"[Lark] extract video cover failed: {e}")
            return None
