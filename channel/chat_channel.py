import os
import re
import base64
import threading
import time
import docx
from PIL import Image
from asyncio import CancelledError
from concurrent.futures import Future, ThreadPoolExecutor

from bridge.context import *
from bridge.reply import *
from channel.channel import Channel
from common.dequeue import Dequeue
from common import memory, const
from common.tmp_dir import create_user_dir
from common.tool_button import tool_state
from common.model_status import model_state
from plugins.bigchao.script_breakdown import cache_media
from plugins import *
from config import conf

try:
    from voice.audio_convert import any_to_wav
except Exception as e:
    pass


# 抽象类, 它包含了与消息通道无关的通用处理逻辑
class ChatChannel(Channel):
    name = None  # 登录的用户名
    user_id = None  # 登录的用户id
    futures = {}  # 记录每个session_id提交到线程池的future对象, 用于重置会话时把没执行的future取消掉，正在执行的不会被取消
    sessions = {}  # 用于控制并发，每个session_id同时只能有一个context在处理
    lock = threading.Lock()  # 用于控制对sessions的访问
    handler_pool = ThreadPoolExecutor(max_workers=16)  # 处理消息的线程池
    cache_locks = {}  # 为每个session添加缓存锁
    _consumer_thread = None
    _consumer_pid = None

    def __init__(self):
        super().__init__()
        self.ensure_consumer_thread()

    def ensure_consumer_thread(self):
        current_pid = os.getpid()
        thread = self.__class__._consumer_thread
        if self.__class__._consumer_pid != current_pid or thread is None or not thread.is_alive():
            thread = threading.Thread(target=self.consume, name=f"{self.__class__.__name__}-consumer", daemon=True)
            thread.start()
            self.__class__._consumer_thread = thread
            self.__class__._consumer_pid = current_pid
            logger.info("%s consumer thread started in pid=%s", self._get_channel(), current_pid)

    # 根据消息构造context，消息内容相关的触发项写在这里
    def _compose_context(self, ctype: ContextType, content, **kwargs):
        context = Context(ctype, content)
        context.kwargs = kwargs
        # context首次传入时，origin_ctype是None,
        # 引入的起因是：当输入语音时，会嵌套生成两个context，第一步语音转文本，第二步通过文本生成文字回复。
        # origin_ctype用于第二步文本回复时，判断是否需要匹配前缀，如果是私聊的语音，就不需要匹配前缀
        if "origin_ctype" not in context:
            context["origin_ctype"] = ctype
        # context首次传入时，receiver是None，根据类型设置receiver
        first_in = "receiver" not in context
        # 群名匹配过程，设置session_id和receiver
        if first_in:  # context首次传入时，receiver是None，根据类型设置receiver
            config = conf()
            cmsg = context["msg"]
            user_data = conf().get_user_data(cmsg.from_user_id)
            context["openai_api_key"] = user_data.get("openai_api_key")
            context["gpt_model"] = user_data.get("gpt_model")
            if context.get("isgroup", False):
                group_name = cmsg.other_user_nickname
                group_id = cmsg.other_user_id

                group_name_white_list = config.get("group_name_white_list", [])
                group_name_keyword_white_list = config.get("group_name_keyword_white_list", [])
                if any(
                    [
                        group_name in group_name_white_list,
                        "ALL_GROUP" in group_name_white_list,
                        check_contain(group_name, group_name_keyword_white_list),
                    ]
                ):
                    group_chat_in_one_session = conf().get("group_chat_in_one_session", [])
                    session_id = cmsg.actual_user_id
                    if any(
                        [
                            group_name in group_chat_in_one_session,
                            "ALL_GROUP" in group_chat_in_one_session,
                        ]
                    ):
                        session_id = group_id
                else:
                    return None
                context["session_id"] = session_id
                context["receiver"] = group_id
            else:
                context["session_id"] = cmsg.other_user_id
                context["receiver"] = cmsg.other_user_id
            e_context = PluginManager().emit_event(EventContext(Event.ON_RECEIVE_MESSAGE, {"channel": self, "context": context}))
            context = e_context["context"]
            if e_context.is_pass() or context is None:
                return context
            if cmsg.from_user_id == self.user_id and not config.get("trigger_by_self", True):
                logger.debug("%s self message skipped", self._get_channel(context))
                return None

        # 消息内容匹配过程，并处理content
        if ctype == ContextType.TEXT:
            if first_in and "」\n- - - - - - -" in content:  # 初次匹配 过滤引用消息
                logger.debug(content)
                logger.debug("%s reference query skipped", self._get_channel(context))
                return None

            nick_name_black_list = conf().get("nick_name_black_list", [])
            if context.get("isgroup", False):  # 群聊
                # 校验关键字
                match_prefix = check_prefix(content, conf().get("group_chat_prefix"))
                match_contain = check_contain(content, conf().get("group_chat_keyword"))
                flag = False
                if context["msg"].to_user_id != context["msg"].actual_user_id:
                    if match_prefix is not None or match_contain is not None:
                        flag = True
                        if match_prefix:
                            content = content.replace(match_prefix, "", 1).strip()
                    if context["msg"].is_at:
                        nick_name = context["msg"].actual_user_nickname
                        if nick_name and nick_name in nick_name_black_list:
                            # 黑名单过滤
                            logger.warning("%s Nickname %s in In BlackList, ignore", self._get_channel(context), nick_name)
                            return None

                        logger.info("%s receive group at", self._get_channel(context))
                        if not conf().get("group_at_off", False):
                            flag = True
                        pattern = f"@{re.escape(self.name)}(\u2005|\u0020)"
                        subtract_res = re.sub(pattern, r"", content)
                        if isinstance(context["msg"].at_list, list):
                            for at in context["msg"].at_list:
                                pattern = f"@{re.escape(at)}(\u2005|\u0020)"
                                subtract_res = re.sub(pattern, r"", subtract_res)
                        if subtract_res == content and context["msg"].self_display_name:
                            # 前缀移除后没有变化，使用群昵称再次移除
                            pattern = f"@{re.escape(context['msg'].self_display_name)}(\u2005|\u0020)"
                            subtract_res = re.sub(pattern, r"", content)
                        content = subtract_res
                if not flag:
                    if context["origin_ctype"] == ContextType.VOICE:
                        logger.info("%s receive group voice, but checkprefix didn't match", self._get_channel(context))
                    return None
            else:  # 单聊
                nick_name = context["msg"].from_user_nickname
                if nick_name and nick_name in nick_name_black_list:
                    # 黑名单过滤
                    logger.warning("%s Nickname '%s' in In BlackList, ignore", self._get_channel(context), nick_name)
                    return None

                match_prefix = check_prefix(content, conf().get("single_chat_prefix", [""]))
                if match_prefix is not None:  # 判断如果匹配到自定义前缀，则返回过滤掉前缀+空格后的内容
                    content = content.replace(match_prefix, "", 1).strip()
                elif context["origin_ctype"] == ContextType.VOICE:  # 如果源消息是私聊的语音消息，允许不匹配前缀，放宽条件
                    pass
                else:
                    return None
            content = content.strip()
            video_match_prefix = check_prefix(content, conf().get("video_create_prefix", ["//"])) 
            img_match_prefix = check_prefix(content, conf().get("image_create_prefix"))
            if video_match_prefix:
                content = content.replace(video_match_prefix, "", 1)
                context.type = ContextType.VIDEO_CREATE
            elif img_match_prefix:
                content = content.replace(img_match_prefix, "", 1)
                context.type = ContextType.IMAGE_CREATE
            else:
                context.type = ContextType.TEXT
            context.content = content.strip()
            if "desire_rtype" not in context and conf().get("always_reply_voice") and ReplyType.VOICE not in self.NOT_SUPPORT_REPLYTYPE:
                context["desire_rtype"] = ReplyType.VOICE
        elif context.type == ContextType.VOICE:
            if "desire_rtype" not in context and conf().get("voice_reply_voice") and ReplyType.VOICE not in self.NOT_SUPPORT_REPLYTYPE:
                context["desire_rtype"] = ReplyType.VOICE

        return context

    def _handle(self, context: Context):
        if context is None or not context.content:
            return
        logger.debug("%s ready to handle context: %s", self._get_channel(context), context)
        # reply的构建步骤
        reply = self._generate_reply(context)

        logger.debug("%s ready to decorate reply: %s", self._get_channel(context), reply)
        # reply的包装步骤
        reply = self._decorate_reply(context, reply)

        # reply的发送步骤
        self._send_reply(context, reply)

    def _get_channel(self, context: Context = None):
        channel = context.get("channel") if context else None
        channel_type = getattr(channel, "channel_type", None) or getattr(self, "channel_type", None) or conf().get("channel_type", "wx")
        return f"[{str(channel_type).upper()}]"

    def _cache_quoted_image(self, context: Context):
        quoted_image_path = context.get("quoted_image_path")
        session_id = context.get("session_id")
        if not quoted_image_path or not session_id:
            return
        channel = self._get_channel(context)
        try:
            img = Image.open(quoted_image_path)
            memory.USER_QUOTED_IMAGE_CACHE[session_id] = {
                "path": [quoted_image_path],
                "files": [img]
            }
            logger.info(
                f"{channel} quoted image cached, session_id={session_id}, "
                f"path={quoted_image_path}, size={getattr(img, 'size', None)}, mode={getattr(img, 'mode', None)}"
            )
        except Exception as e:
            logger.warning(f"{channel} failed to cache quoted image: {e}")

    def _cache_quoted_video(self, context: Context):
        quoted_video_path = context.get("quoted_video_path")
        session_id = context.get("session_id")
        if not quoted_video_path or not session_id:
            return
        channel = self._get_channel(context)
        try:
            video_cache_item = {
                "path": quoted_video_path,
                "public_url": context.get("quoted_video_public_url"),
                "mime_type": f"video/{os.path.splitext(quoted_video_path)[1].lstrip('.').lower() or 'mp4'}",
            }
            memory.USER_QUOTED_VIDEO_CACHE[session_id] = {"files": [video_cache_item]}
            logger.info(f"{channel} quoted video cached, session_id={session_id}, path={quoted_video_path}")
        except Exception as e:
            logger.warning(f"{channel} failed to cache quoted video: {e}")

    def _cache_quoted_file(self, context: Context):
        quoted_file_path = context.get("quoted_file_path")
        session_id = context.get("session_id")
        if not quoted_file_path or not session_id:
            return
        channel = self._get_channel(context)
        try:
            suffix = os.path.splitext(quoted_file_path)[1].lstrip(".").lower()
            normalized_suffix = "plain" if suffix in {"txt", "plain"} else suffix
            if normalized_suffix not in const.DOCUMENT:
                logger.warning(f"{channel} unsupported quoted file type: {suffix}")
                return

            if normalized_suffix == "pdf":
                with open(quoted_file_path, 'rb') as file:
                    encoded_data = base64.b64encode(file.read()).decode('utf-8')
            elif normalized_suffix == "docx":
                doc = docx.Document(quoted_file_path)
                encoded_data = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
            else:
                with open(quoted_file_path, 'r', encoding='utf-8', errors='ignore') as file:
                    encoded_data = file.read()

            memory.USER_QUOTED_FILE_CACHE[session_id] = {
                "path": quoted_file_path,
                "msg": context.get("msg"),
                "files": [{
                    "path": quoted_file_path,
                    "msg": context.get("msg"),
                    "mime_type": f"application/{normalized_suffix}",
                    "data": encoded_data,
                }],
            }
            file_size = os.path.getsize(quoted_file_path) if os.path.exists(quoted_file_path) else 0
            logger.info(
                f"{channel} quoted file cached, session_id={session_id}, "
                f"path={quoted_file_path}, mime_type=application/{normalized_suffix}, "
                f"size={file_size}, content_length={len(encoded_data)}"
            )
        except Exception as e:
            logger.warning(f"{channel} failed to cache quoted file: {e}")

    def _generate_reply(self, context: Context, reply: Reply = Reply()) -> Reply:
        session_id = context["session_id"]
                
        # 确保session有对应的锁
        if session_id not in self.cache_locks:
            self.cache_locks[session_id] = threading.Lock()
            
        # 获取该session的缓存锁
        cache_lock = self.cache_locks[session_id]

        e_context = PluginManager().emit_event(
            EventContext(
                Event.ON_HANDLE_CONTEXT,
                {"channel": self, "context": context, "reply": reply},
            )
        )
        reply = e_context["reply"]
        model = model_state.get_basic_state(session_id)
        if not e_context.is_pass():
            logger.debug("%s ready to handle context: type=%s, content=%s", self._get_channel(context), context.type, context.content)
            if context.type in (ContextType.IMAGE_CREATE, ContextType.VIDEO_CREATE):
                self._cache_quoted_image(context)
            if context.type == ContextType.VIDEO_CREATE:
                self._cache_quoted_video(context)
            if context.type == ContextType.VOICE:  # 语音消息
                cmsg = context["msg"]
                cmsg.prepare()
                file_path = context.content
                wav_path = os.path.splitext(file_path)[0] + ".wav"
                try:
                    any_to_wav(file_path, wav_path)
                except Exception as e:  # 转换失败，直接使用mp3，对于某些api，mp3也可以识别
                    logger.warning("%s any to wav error, use raw path. %s", self._get_channel(context), e)
                    wav_path = file_path
                # 语音识别
                reply = super().build_voice_to_text(wav_path)
                # 删除临时文件
                try:
                    os.remove(file_path)
                    if wav_path != file_path:
                        os.remove(wav_path)
                except Exception as e:
                    pass
                    # logger.warning("[WX]delete temp file error: " + str(e))

                if reply.type == ReplyType.TEXT:
                    new_context = self._compose_context(ContextType.TEXT, reply.content, **context.kwargs)
                    if new_context:
                        reply = self._generate_reply(new_context)
                    else:
                        return
            elif context.type == ContextType.TEXT:
                # 文字消息
                with cache_lock: # 等待图片缓存完成后再处理文本
                    if getattr(context.get("msg"), "_prepare_fn", None):
                        context["msg"].prepare()
                    self._cache_quoted_image(context)
                    self._cache_quoted_file(context)
                    context["channel"] = e_context["channel"]
                    reply = super().build_reply_content(context.content, context)
            elif context.type == ContextType.IMAGE:
                image_path = context.content
                dir_path = os.path.dirname(image_path)
                dir_exists = os.path.exists(dir_path)
                if not dir_exists:
                    create_user_dir(dir_path)
                    # 使用用户特定的imaging状态
                session_id = context["session_id"]
                is_imaging = tool_state.get_image_state(session_id)
                is_video = tool_state.get_edit_state(session_id)
                model = (
                    model_state.get_video_state(session_id) if is_video
                    else model_state.get_image_model(session_id) if is_imaging
                    else model_state.get_basic_state(session_id)
                )
                logger.info(f'[{model.upper()}] query with image, path={image_path}')
                mime_type = image_path[(image_path.rfind('.') + 1):]
                type_id = 'image'
                channel_type = conf().get("channel_type")
                if mime_type in const.IMAGE or channel_type == 'feishu':
                    with cache_lock: # 使用锁确保缓存完成
                        context['msg'].prepare()
                        img = Image.open(image_path)
                        image_file = img
                        # check if the image has an alpha channel
                        if img.mode in ('RGBA','LA') or (img.mode == 'P' and 'transparency' in img.info):
                            # Convert the image to RGB mode,whick removes the alpha channel
                            img = img.convert('RGB')
                            # Save the converted image
                            img_path_no_alpha = image_path + '.jpg' if channel_type == 'feishu' else image_path[:len(image_path)-3] + 'jpg'
                            img.save(img_path_no_alpha)
                            # Update img_path with the path to the converted image
                            image_path = img_path_no_alpha
                            img.format = "JPEG"
                            image_file = img
                        cache_media(image_path, image_file, context)
                else:
                    logger.warning(f'[{model.upper()}] query with unsupported image type:{mime_type}') 
            elif context.type == ContextType.SHARING and model in const.GEMINI_GENAI_SDK:  
                # 分享信息
                context["channel"] = e_context["channel"]
                reply = super().build_reply_content(context.content, context)
            elif context.type == ContextType.FILE:
                file_path = context.content
                dir_path = os.path.dirname(file_path)
                dir_exists = os.path.exists(dir_path)
                if not dir_exists:
                    create_user_dir(dir_path)
                logger.info(f'[{model.upper()}] query with file, path={file_path}')
                mime_type = file_path[(file_path.rfind('.') + 1):]
                type_id = 'application'
                if mime_type in const.VIDEO:
                    with cache_lock:
                        context["msg"].prepare()
                        video_cache_item = {
                            "path": file_path,
                            "public_url": context.get("video_public_url"),
                            "mime_type": f"video/{mime_type}",
                        }
                        existing_cache = memory.USER_VIDEO_CACHE.get(session_id)
                        if existing_cache and isinstance(existing_cache.get("files"), list):
                            existing_cache["files"].append(video_cache_item)
                        else:
                            memory.USER_VIDEO_CACHE[session_id] = {"files": [video_cache_item]}
                        logger.info(f'[{model.upper()}] {file_path} cached to video memory from file branch')
                elif mime_type in const.DOCUMENT:
                    with cache_lock:
                        context["msg"].prepare()
                        if mime_type == 'pdf':
                            with open(file_path, 'rb') as file:
                                file_data = file.read()
                                encoded_data = base64.b64encode(file_data).decode('utf-8')
                        elif mime_type == 'docx':
                            doc = docx.Document(file_path)
                            full_text = []
                            for paragraph in doc.paragraphs:
                                full_text.append(paragraph.text)
                            encoded_data = '\n'.join(full_text)
                        else:
                            with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
                                encoded_data = file.read()
                        file_cache_item = {
                            "path": file_path,
                            "msg": context.get("msg"),
                            "mime_type": f'{type_id}/{mime_type}',
                            "data": encoded_data,
                        }
                        existing_cache = memory.USER_FILE_CACHE.get(session_id)
                        if existing_cache and isinstance(existing_cache.get("files"), list):
                            existing_cache["files"].append(file_cache_item)
                            existing_cache["path"] = file_path
                            existing_cache["msg"] = context.get("msg")
                        else:
                            memory.USER_FILE_CACHE[session_id] = {
                                "path": file_path,
                                "msg": context.get("msg"),
                                "files": [file_cache_item],
                            }
                        logger.info(f'[{model.upper()}] {file_path} cached to file memory from file branch')
                else:
                    logger.warning(f'[{model.upper()}] query with unsupported file type:{mime_type}')
            elif context.type == ContextType.IMAGE_CREATE:
                reply = super().build_image_content(context.content, context)
            elif context.type == ContextType.VIDEO_CREATE:
                reply = super().build_video_content(context.content, context)
            elif context.type == ContextType.VIDEO:   
                file_path = context.content
                dir_path = os.path.dirname(file_path)
                if not os.path.exists(dir_path):
                    create_user_dir(dir_path)
                session_id = context["session_id"]
                model = model_state.get_video_state(session_id)
                logger.info(f'[{model.upper()}] query with video, path={file_path}')
                with cache_lock:
                    context["msg"].prepare()
                    video_cache_item = {
                        "path": file_path,
                        "public_url": context.get("video_public_url"),
                        "mime_type": "video/mp4",
                    }
                    existing_cache = memory.USER_VIDEO_CACHE.get(session_id)
                    if existing_cache and isinstance(existing_cache.get("files"), list):
                        existing_cache["files"].append(video_cache_item)
                    else:
                        memory.USER_VIDEO_CACHE[session_id] = {"files": [video_cache_item]}
                    logger.info(f'[{model.upper()}] {file_path} cached to video memory')
            else:
                logger.warning("%s unknown context type: %s", self._get_channel(context), context.type)
                return
        return reply

    def _decorate_reply(self, context: Context, reply: Reply) -> Reply:
        if reply and reply.type:
            e_context = PluginManager().emit_event(
                EventContext(
                    Event.ON_DECORATE_REPLY,
                    {"channel": self, "context": context, "reply": reply},
                )
                )
            reply = e_context["reply"]
            desire_rtype = context.get("desire_rtype")
            if not e_context.is_pass() and reply and reply.type:
                if reply.type in self.NOT_SUPPORT_REPLYTYPE:
                    logger.error("%s reply type not support: %s", self._get_channel(context), reply.type)
                    reply.type = ReplyType.ERROR
                    reply.content = "不支持发送的消息类型: " + str(reply.type)

                if reply.type == ReplyType.TEXT:
                    reply_text = reply.content
                    if desire_rtype == ReplyType.VOICE and ReplyType.VOICE not in self.NOT_SUPPORT_REPLYTYPE:
                        reply = super().build_text_to_voice(reply.content)
                        return self._decorate_reply(context, reply)
                    if context.get("isgroup", False):
                        if not context.get("no_need_at", False):
                            reply_text = "@" + context["msg"].actual_user_nickname + "\n" + reply_text.strip()
                        reply_text = conf().get("group_chat_reply_prefix", "") + reply_text + conf().get("group_chat_reply_suffix", "")
                    else:
                        reply_text = conf().get("single_chat_reply_prefix", "") + reply_text + conf().get("single_chat_reply_suffix", "")
                    reply.content = reply_text
                elif reply.type == ReplyType.ERROR or reply.type == ReplyType.INFO:
                    reply.content = "[" + str(reply.type) + "]\n" + reply.content
                elif reply.type in (
                    ReplyType.IMAGE_URL,
                    ReplyType.VOICE,
                    ReplyType.IMAGE,
                    ReplyType.FILE,
                    ReplyType.VIDEO,
                    ReplyType.VIDEO_URL,
                    ReplyType.STREAM
                ):
                    pass
                else:
                    logger.error("%s unknown reply type: %s", self._get_channel(context), reply.type)
                    return
            if desire_rtype and desire_rtype != reply.type and reply.type not in [ReplyType.ERROR, ReplyType.INFO]:
                logger.warning(
                    "%s desire_rtype: %s, but reply type: %s",
                    self._get_channel(context),
                    context.get("desire_rtype"),
                    reply.type,
                )
            return reply

    def _send_reply(self, context: Context, reply: Reply):
        if reply and reply.type:
            e_context = PluginManager().emit_event(
                EventContext(
                    Event.ON_SEND_REPLY,
                    {"channel": self, "context": context, "reply": reply},
                )
            )
            reply = e_context["reply"]
            if not e_context.is_pass() and reply and reply.type:
                logger.debug("%s ready to send reply: %s, context: %s", self._get_channel(context), reply, context)
                self._send(reply, context)

    def _send(self, reply: Reply, context: Context, retry_cnt=0):
        try:
            self.send(reply, context)
        except Exception as e:
            logger.error("%s sendMsg error: %s", self._get_channel(context), str(e))
            if isinstance(e, NotImplementedError):
                return
            logger.exception(e)
            if retry_cnt < 2:
                time.sleep(3 + 3 * retry_cnt)
                self._send(reply, context, retry_cnt + 1)

    def _success_callback(self, session_id, **kwargs):  # 线程正常结束时的回调函数
        logger.debug("Worker return success, session_id = {}".format(session_id))

    def _fail_callback(self, session_id, exception, **kwargs):  # 线程异常结束时的回调函数
        logger.exception("Worker return exception: {}".format(exception))

    def _thread_pool_callback(self, session_id, **kwargs):
        def func(worker: Future):
            try:
                worker_exception = worker.exception()
                if worker_exception:
                    self._fail_callback(session_id, exception=worker_exception, **kwargs)
                else:
                    self._success_callback(session_id, **kwargs)
            except CancelledError as e:
                logger.info("Worker cancelled, session_id = {}".format(session_id))
            except Exception as e:
                logger.exception("Worker raise exception: {}".format(e))
            with self.lock:
                self.sessions[session_id][1].release()

        return func

    def produce(self, context: Context):
        self.ensure_consumer_thread()
        session_id = context["session_id"]
        with self.lock:
            if session_id not in self.sessions:
                self.sessions[session_id] = [
                    Dequeue(),
                    threading.BoundedSemaphore(conf().get("concurrency_in_session", 4)),
                ]
            if context.type == ContextType.TEXT and context.content.startswith("#"):
                self.sessions[session_id][0].putleft(context)  # 优先处理管理命令
            else:
                self.sessions[session_id][0].put(context)

    # 消费者函数，单独线程，用于从消息队列中取出消息并处理
    def consume(self):
        while True:
            try:
                with self.lock:
                    session_ids = list(self.sessions.keys())
                    for session_id in session_ids:
                        context_queue, semaphore = self.sessions[session_id]
                        if semaphore.acquire(blocking=False):  # 等线程处理完毕才能删除
                            if not context_queue.empty():
                                context = context_queue.get()
                                logger.debug("%s consume context: %s", self._get_channel(context), context)
                                future: Future = self.handler_pool.submit(self._handle, context)
                                future.add_done_callback(self._thread_pool_callback(session_id, context=context))
                                self.futures.setdefault(session_id, []).append(future)
                            elif semaphore._initial_value == semaphore._value + 1:  # 除了当前，没有任务再申请到信号量，说明所有任务都处理完毕
                                session_futures = [t for t in self.futures.get(session_id, []) if not t.done()]
                                self.futures[session_id] = session_futures
                                assert len(session_futures) == 0, "thread pool error"
                                del self.sessions[session_id]
                                self.futures.pop(session_id, None)
                            else:
                                semaphore.release()
            except Exception as e:
                logger.exception("%s consume loop error: %s", self._get_channel(), e)
            time.sleep(0.1)

    # 取消session_id对应的所有任务，只能取消排队的消息和已提交线程池但未执行的任务
    def cancel_session(self, session_id):
        with self.lock:
            if session_id in self.sessions:
                for future in self.futures[session_id]:
                    future.cancel()
                cnt = self.sessions[session_id][0].qsize()
                if cnt > 0:
                    logger.info("Cancel {} messages in session {}".format(cnt, session_id))
                self.sessions[session_id][0] = Dequeue()

    def cancel_all_session(self):
        with self.lock:
            for session_id in self.sessions:
                for future in self.futures[session_id]:
                    future.cancel()
                cnt = self.sessions[session_id][0].qsize()
                if cnt > 0:
                    logger.info("Cancel {} messages in session {}".format(cnt, session_id))
                self.sessions[session_id][0] = Dequeue()


def check_prefix(content, prefix_list):
    if not prefix_list:
        return None
    for prefix in prefix_list:
        if content.startswith(prefix):
            return prefix
    return None


def check_contain(content, keyword_list):
    if not keyword_list:
        return None
    for ky in keyword_list:
        if content.find(ky) != -1:
            return True
    return None
