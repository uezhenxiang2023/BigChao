"""
Weixin channel implementation.

Uses HTTP long-poll (getUpdates) to receive messages and sendMessage to reply.
Login via QR code scan through the ilink bot API.
"""

import json
import io
import os
import threading
import time
import uuid

import requests

from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from channel.chat_channel import ChatChannel, check_prefix
from channel.weixin.weixin_api import (
    WeixinApi, upload_media_to_cdn,
    DEFAULT_BASE_URL, CDN_BASE_URL,
)
from channel.weixin.weixin_message import WeixinMessage
from common import const
from common.expired_dict import ExpiredDict
from common.log import logger
from common.media_store import build_public_media_url
from common.singleton import singleton
from common.tmp_dir import get_response_dir
from config import conf

MAX_CONSECUTIVE_FAILURES = 3
BACKOFF_DELAY = 30
RETRY_DELAY = 2
SESSION_EXPIRED_ERRCODE = -14
TEXT_CHUNK_LIMIT = 4000
QR_LOGIN_TIMEOUT_S = 480
QR_MAX_REFRESHES = 10


def _load_credentials(cred_path: str) -> dict:
    """Load saved credentials from JSON file."""
    try:
        if os.path.exists(cred_path):
            with open(cred_path, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"[Weixin] Failed to load credentials: {e}")
    return {}


def _save_credentials(cred_path: str, data: dict):
    """Atomically save credentials to JSON file (tmp + rename)."""
    os.makedirs(os.path.dirname(cred_path), exist_ok=True)
    tmp_path = f"{cred_path}.tmp"
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    try:
        os.chmod(tmp_path, 0o600)
    except Exception:
        pass
    os.replace(tmp_path, cred_path)


@singleton
class WeixinChannel(ChatChannel):

    # ilink bot protocol has no outbound voice item; deliver TTS as a file.
    NOT_SUPPORT_REPLYTYPE = []

    LOGIN_STATUS_IDLE = "idle"
    LOGIN_STATUS_WAITING = "waiting_scan"
    LOGIN_STATUS_SCANNED = "scanned"
    LOGIN_STATUS_OK = "logged_in"

    def __init__(self, session_id=None):
        super().__init__()
        self.api = None
        self._stop_event = threading.Event()
        self._poll_thread = None
        # user_id -> context_token. Guarded by _context_tokens_lock for any
        # mutation that races with disk persistence.
        self._context_tokens = {}
        self._context_tokens_lock = threading.Lock()
        self._received_msgs = ExpiredDict(60 * 60 * 7.1)
        self._get_updates_buf = ""
        self._credentials_path = ""
        self.login_status = self.LOGIN_STATUS_IDLE
        self._current_qr_url = ""

        conf()["single_chat_prefix"] = [""]

    # ── Lifecycle ──────────────────────────────────────────────────────

    def startup(self):
        self._stop_event.clear()

        base_url = conf().get("weixin_base_url", DEFAULT_BASE_URL)
        cdn_base_url = conf().get("weixin_cdn_base_url", CDN_BASE_URL)
        token = conf().get("weixin_token", "")

        self._credentials_path = os.path.expanduser(
            conf().get("weixin_credentials_path", "~/.weixin_cow_credentials.json")
        )

        # Always load credentials so we can restore context_tokens even when
        # the bot token itself comes from config.
        creds = _load_credentials(self._credentials_path)
        if not token:
            token = creds.get("token", "")
            if creds.get("base_url"):
                base_url = creds["base_url"]

        # Restore persisted context_tokens so scheduler can deliver pushes
        # immediately after restart, without waiting for the user to ping
        # the bot first.
        self._restore_context_tokens_from_creds(creds)

        if not token:
            token, base_url = self._login_with_retry(base_url)
            if not token:
                return

        self.api = WeixinApi(base_url=base_url, token=token, cdn_base_url=cdn_base_url)
        self.login_status = self.LOGIN_STATUS_OK

        logger.info(f"[Weixin] 微信通道已启动，凭证保存在 {self._credentials_path}，"
                     f"如需重新扫码登录请删除该文件后重启")
        self.report_startup_success()

        self._poll_loop()

    def _login_with_retry(self, base_url: str) -> tuple:
        """Attempt QR login, then wait for stop if failed.
        Returns (token, base_url) on success, or ("", "") if stopped."""
        logger.info("[Weixin] No token found, starting QR login...")
        self.login_status = self.LOGIN_STATUS_WAITING
        login_result = self._qr_login(base_url)
        if login_result:
            return login_result["token"], login_result.get("base_url", base_url)

        self.login_status = self.LOGIN_STATUS_IDLE
        if not self._stop_event.is_set():
            logger.info("[Weixin] QR login timed out, waiting for stop or reconnect...")
            print("  二维码登录超时，请通过控制台重新接入\n")
            self._stop_event.wait()

        logger.info("[Weixin] Login cancelled by stop event")
        return "", ""

    def stop(self):
        logger.info("[Weixin] stop() called")
        self._stop_event.set()

    def _relogin(self) -> bool:
        """Re-login after session expiry. Returns True on success."""
        base_url = self.api.base_url if self.api else DEFAULT_BASE_URL
        # Clearing the whole credentials file is intentional: the new login
        # will issue a fresh `token` and persisted context_tokens belong to
        # the previous bot identity, so they must not survive.
        with self._context_tokens_lock:
            self._context_tokens.clear()
            if os.path.exists(self._credentials_path):
                try:
                    os.remove(self._credentials_path)
                except Exception:
                    pass
        self.login_status = self.LOGIN_STATUS_WAITING
        result = self._qr_login(base_url)
        if not result:
            self.login_status = self.LOGIN_STATUS_IDLE
            return False
        self.api = WeixinApi(
            base_url=result.get("base_url", base_url),
            token=result["token"],
            cdn_base_url=self.api.cdn_base_url if self.api else CDN_BASE_URL,
        )
        self.login_status = self.LOGIN_STATUS_OK
        return True

    # ── Context token persistence ──────────────────────────────────────
    # ilink requires every outbound send to echo the context_token from the
    # user's latest inbound message. We mirror the in-memory map into the
    # credentials JSON so scheduled pushes survive process restarts.
    # All mutation + disk IO is serialized via _context_tokens_lock so that
    # concurrent updates can never lose each other's writes.

    def _restore_context_tokens_from_creds(self, creds: dict) -> None:
        if not isinstance(creds, dict):
            return
        tokens = creds.get("context_tokens")
        if not isinstance(tokens, dict):
            return
        restored = 0
        with self._context_tokens_lock:
            for user_id, token in tokens.items():
                if isinstance(user_id, str) and isinstance(token, str) and token:
                    self._context_tokens[user_id] = token
                    restored += 1
        if restored:
            logger.info(f"[Weixin] Restored {restored} context_tokens from credentials")

    def _persist_context_tokens_locked(self) -> None:
        """Flush the token map to disk. Caller must hold _context_tokens_lock."""
        if not self._credentials_path:
            return
        try:
            creds = _load_credentials(self._credentials_path) or {}
            creds["context_tokens"] = dict(self._context_tokens)
            _save_credentials(self._credentials_path, creds)
        except Exception as e:
            logger.warning(f"[Weixin] Failed to persist context_tokens: {e}")

    def _update_context_token(self, user_id: str, token: str) -> None:
        """Update the in-memory token for a user; flush to disk only on change."""
        if not user_id or not token:
            return
        with self._context_tokens_lock:
            if self._context_tokens.get(user_id) == token:
                return
            self._context_tokens[user_id] = token
            self._persist_context_tokens_locked()

    def _invalidate_context_token(self, user_id: str) -> None:
        """Drop the cached token for a user (used after -14 / send rejection)."""
        if not user_id:
            return
        with self._context_tokens_lock:
            if user_id not in self._context_tokens:
                return
            del self._context_tokens[user_id]
            logger.info(f"[Weixin] Invalidated stale context_token for {user_id}")
            self._persist_context_tokens_locked()

    # ── QR Login ───────────────────────────────────────────────────────

    @staticmethod
    def _print_qr(qrcode_url: str):
        """Print QR code to terminal for scanning."""
        print("\n" + "=" * 60)
        print("  请使用微信扫描二维码登录 (二维码约2分钟后过期)")
        print("=" * 60)
        try:
            import qrcode as qr_lib
            import io
            qr = qr_lib.QRCode(error_correction=qr_lib.constants.ERROR_CORRECT_L, box_size=1, border=1)
            qr.add_data(qrcode_url)
            qr.make(fit=True)
            buf = io.StringIO()
            qr.print_ascii(out=buf, invert=True)
            try:
                print(buf.getvalue())
            except UnicodeEncodeError:
                # Windows GBK terminals cannot render Unicode block characters
                print(f"\n  (终端不支持显示二维码，请使用链接扫码)")
                print(f"  二维码链接: {qrcode_url}\n")
        except ImportError:
            print(f"\n  二维码链接: {qrcode_url}")
            print("  (安装 'qrcode' 包可在终端显示二维码)\n")

    def _notify_cloud_qrcode(self, qrcode_url: str):
        """Send QR code URL to cloud console when running in cloud mode."""
        if not self.cloud_mode:
            return
        try:
            from common import cloud_client
            client = getattr(cloud_client, "chat_client", None)
            if client and getattr(client, "client_id", None):
                client.send_channel_qrcode("weixin", qrcode_url)
        except Exception as e:
            logger.warning(f"[Weixin] Failed to notify cloud QR code: {e}")

    def _notify_cloud_connected(self):
        """Send connected status to cloud console when login succeeds."""
        if not self.cloud_mode:
            return
        try:
            from common import cloud_client
            client = getattr(cloud_client, "chat_client", None)
            if client and getattr(client, "client_id", None):
                client.send_channel_status("weixin", "connected")
        except Exception as e:
            logger.warning(f"[Weixin] Failed to notify cloud connected: {e}")

    def _qr_login(self, base_url: str) -> dict:
        """Perform interactive QR code login. Returns dict with token/base_url or empty dict."""
        api = WeixinApi(base_url=base_url)
        try:
            qr_resp = api.fetch_qr_code()
        except Exception as e:
            logger.error(f"[Weixin] Failed to fetch QR code: {e}")
            return {}

        qrcode = qr_resp.get("qrcode", "")
        qrcode_url = qr_resp.get("qrcode_img_content", "")

        if not qrcode:
            logger.error("[Weixin] No QR code returned from server")
            return {}

        self._current_qr_url = qrcode_url
        logger.info(f"[Weixin] 微信二维码链接: {qrcode_url}")
        self._print_qr(qrcode_url)
        self._notify_cloud_qrcode(qrcode_url)
        print("  等待扫码...\n")

        scanned_printed = False
        refresh_count = 0
        deadline = time.time() + QR_LOGIN_TIMEOUT_S

        while not self._stop_event.is_set():
            if time.time() >= deadline:
                logger.warning(f"[Weixin] QR login timed out after {QR_LOGIN_TIMEOUT_S}s")
                print(f"\n  二维码登录超时（{QR_LOGIN_TIMEOUT_S}s），请重启后重试")
                break

            try:
                status_resp = api.poll_qr_status(qrcode)
            except Exception as e:
                logger.error(f"[Weixin] QR status poll error: {e}")
                return {}

            status = status_resp.get("status", "wait")

            if status == "wait":
                pass
            elif status == "scaned":
                self.login_status = self.LOGIN_STATUS_SCANNED
                if not scanned_printed:
                    print("  已扫码，请在手机上确认...")
                    scanned_printed = True
            elif status == "expired":
                refresh_count += 1
                if refresh_count >= QR_MAX_REFRESHES:
                    logger.warning(f"[Weixin] QR code refreshed {QR_MAX_REFRESHES} times, giving up")
                    print(f"\n  二维码已刷新 {QR_MAX_REFRESHES} 次仍未扫码，请重启后重试")
                    break
                print(f"  二维码已过期，正在刷新（{refresh_count}/{QR_MAX_REFRESHES}）...")
                try:
                    qr_resp = api.fetch_qr_code()
                    qrcode = qr_resp.get("qrcode", "")
                    qrcode_url = qr_resp.get("qrcode_img_content", "")
                    scanned_printed = False
                    self._current_qr_url = qrcode_url
                    logger.info(f"[Weixin] 微信二维码链接 ({refresh_count}/{QR_MAX_REFRESHES}): {qrcode_url}")
                    self._print_qr(qrcode_url)
                    self._notify_cloud_qrcode(qrcode_url)
                except Exception as e:
                    logger.error(f"[Weixin] QR refresh failed: {e}")
                    return {}
            elif status == "confirmed":
                bot_token = status_resp.get("bot_token", "")
                bot_id = status_resp.get("ilink_bot_id", "")
                result_base_url = status_resp.get("baseurl", base_url)
                user_id = status_resp.get("ilink_user_id", "")

                if not bot_token or not bot_id:
                    logger.error("[Weixin] Login confirmed but missing token/bot_id")
                    return {}

                self._current_qr_url = ""
                print(f"\n  ✅ 微信登录成功！bot_id={bot_id}")
                logger.info(f"[Weixin] Login confirmed: bot_id={bot_id}")
                self._notify_cloud_connected()

                creds = {
                    "token": bot_token,
                    "base_url": result_base_url,
                    "bot_id": bot_id,
                    "user_id": user_id,
                }
                _save_credentials(self._credentials_path, creds)
                logger.info(f"[Weixin] Credentials saved to {self._credentials_path}")

                return {"token": bot_token, "base_url": result_base_url}

            self._stop_event.wait(1)

        self._current_qr_url = ""
        if self._stop_event.is_set():
            logger.info("[Weixin] QR login cancelled by stop event")
        return {}

    # ── Long-poll loop ─────────────────────────────────────────────────

    def _poll_loop(self):
        """Main long-poll loop: getUpdates -> parse -> produce."""
        logger.info("[Weixin] Starting long-poll loop")
        consecutive_failures = 0

        while not self._stop_event.is_set():
            try:
                resp = self.api.get_updates(self._get_updates_buf)

                ret = resp.get("ret", 0)
                errcode = resp.get("errcode", 0)

                is_error = (ret != 0) or (errcode != 0)
                if is_error:
                    if errcode == SESSION_EXPIRED_ERRCODE or ret == SESSION_EXPIRED_ERRCODE:
                        logger.error("[Weixin] Session expired (errcode -14), starting re-login...")
                        if self._relogin():
                            logger.info("[Weixin] Re-login successful, resuming long-poll")
                            self._get_updates_buf = ""
                            consecutive_failures = 0
                            continue
                        else:
                            logger.error("[Weixin] Re-login failed, will retry in 5 minutes")
                            self._stop_event.wait(300)
                            continue

                    consecutive_failures += 1
                    errmsg = resp.get("errmsg", "")
                    logger.error(f"[Weixin] getUpdates error: ret={ret} errcode={errcode} "
                                 f"errmsg={errmsg} ({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                    if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                        consecutive_failures = 0
                        self._stop_event.wait(BACKOFF_DELAY)
                    else:
                        self._stop_event.wait(RETRY_DELAY)
                    continue

                consecutive_failures = 0

                # Update sync cursor
                new_buf = resp.get("get_updates_buf", "")
                if new_buf:
                    self._get_updates_buf = new_buf

                # Process messages
                msgs = resp.get("msgs", [])
                for raw_msg in msgs:
                    try:
                        self.handler_single_msg(raw_msg)
                    except Exception as e:
                        logger.error(f"[Weixin] Failed to process message: {e}", exc_info=True)

            except Exception as e:
                if self._stop_event.is_set():
                    break
                consecutive_failures += 1
                logger.error(f"[Weixin] getUpdates exception: {e} "
                             f"({consecutive_failures}/{MAX_CONSECUTIVE_FAILURES})")
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    consecutive_failures = 0
                    self._stop_event.wait(BACKOFF_DELAY)
                else:
                    self._stop_event.wait(RETRY_DELAY)

        logger.info("[Weixin] Long-poll loop ended")
    
    def handler_single_msg(self, raw_msg: dict):
        """从 long-poll 原始消息体组装 WeixinMessage，再交给 handle_single 处理。"""
        msg_type = raw_msg.get("message_type", 0)
        if msg_type != 1:  # Only process USER messages (type=1)
            return
 
        msg_id = str(raw_msg.get("message_id", raw_msg.get("seq", "")))
        if self._received_msgs.get(msg_id):
            return
        self._received_msgs[msg_id] = True
 
        from_user = raw_msg.get("from_user_id", "")
        context_token = raw_msg.get("context_token", "")
        if context_token and from_user:
            self._update_context_token(from_user, context_token)
 
        cdn_base_url = self.api.cdn_base_url if self.api else CDN_BASE_URL
        try:
            wx_msg = WeixinMessage(raw_msg, cdn_base_url=cdn_base_url)
        except Exception as e:
            logger.error(f"[Weixin] Failed to parse WeixinMessage: {e}", exc_info=True)
            return
 
        self.handle_single(wx_msg)

    def handle_single(self, cmsg: WeixinMessage):
        """处理单条已解析的 WeixinMessage：打日志、注入 quoted media、produce。"""
        # 按 ctype 分类打日志
        if cmsg.ctype == ContextType.VOICE:
            logger.debug(f"[Weixin] receive voice msg: {cmsg.content}")
        elif cmsg.ctype == ContextType.IMAGE:
            logger.debug(f"[Weixin] receive image msg: {cmsg.content}")
        elif cmsg.ctype == ContextType.VIDEO:
            logger.debug(f"[Weixin] receive video msg: {cmsg.content}")
        elif cmsg.ctype == ContextType.FILE:
            logger.debug(f"[Weixin] receive file msg: {cmsg.content}")
        elif cmsg.ctype == ContextType.TEXT:
            logger.debug(f"[Weixin] receive text msg: {cmsg.content}, cmsg={cmsg}")
        else:
            logger.debug(f"[Weixin] receive msg: {cmsg.content}, cmsg={cmsg}")
 
        logger.info(f"[Weixin] Received: from={cmsg.from_user_id} ctype={cmsg.ctype} "
                    f"content={str(cmsg.content)[:50]}")
 
        context = self._compose_context(
            cmsg.ctype,
            cmsg.content,
            isgroup=False,
            msg=cmsg,
            no_need_at=True,
        )
        if context is None:
            return
 
        # 注入 quoted media
        if hasattr(cmsg, "get_quoted_image_path"):
            quoted_image_path = cmsg.get_quoted_image_path()
            if quoted_image_path:
                context["quoted_image_path"] = quoted_image_path
 
        if hasattr(cmsg, "get_quoted_video_path"):
            quoted_video_path = cmsg.get_quoted_video_path()
            if quoted_video_path:
                context["quoted_video_path"] = quoted_video_path
                quoted_video_public_url = build_public_media_url(quoted_video_path)
                if quoted_video_public_url:
                    context["quoted_video_public_url"] = quoted_video_public_url
 
        if hasattr(cmsg, "get_quoted_file_path"):
            quoted_file_path = cmsg.get_quoted_file_path()
            if quoted_file_path:
                context["quoted_file_path"] = quoted_file_path
 
        # VIDEO 消息本体公网 URL
        if cmsg.ctype == ContextType.VIDEO:
            public_url = build_public_media_url(cmsg.content)
            if public_url:
                context["video_public_url"] = public_url
 
        # FILE 消息如果是视频文件，也补充公网 URL
        if cmsg.ctype == ContextType.FILE:
            suffix = os.path.splitext(cmsg.content)[1].lstrip(".").lower()
            if suffix in const.VIDEO:
                public_url = build_public_media_url(cmsg.content)
                if public_url:
                    context["video_public_url"] = public_url
 
        self.produce(context)

    # ── _compose_context ───────────────────────────────────────────────

    def _compose_context(self, ctype: ContextType, content, **kwargs):
        context = Context(ctype, content)
        context.kwargs = kwargs
        if "channel_type" not in context:
            context["channel_type"] = self.channel_type
        if "origin_ctype" not in context:
            context["origin_ctype"] = ctype

        cmsg = context["msg"]
        context["session_id"] = cmsg.from_user_id
        context["receiver"] = cmsg.other_user_id

        if ctype == ContextType.TEXT:
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
            if "desire_rtype" not in context and conf().get("always_reply_voice"):
                context["desire_rtype"] = ReplyType.VOICE

        elif ctype == ContextType.VOICE:
            if "desire_rtype" not in context and (
                conf().get("voice_reply_voice") or conf().get("always_reply_voice")
            ):
                context["desire_rtype"] = ReplyType.VOICE

        return context

    # ── Send reply ─────────────────────────────────────────────────────

    def send(self, reply: Reply, context: Context):
        receiver = context.get("receiver", "")
        msg = context.get("msg")
        context_token = self._get_context_token(receiver, msg)
        reply_type = reply.type

        if not context_token:
            logger.error(
                "[Weixin] No context_token for receiver=%s, reply_type=%s, cannot send",
                receiver,
                reply_type,
            )
            return

        if reply_type == ReplyType.TEXT:
            self._send_text(reply.content, receiver, context_token)
            logger.info("[Weixin] sendMsg=%s, receiver=%s", reply, receiver)
        elif reply_type == ReplyType.ERROR:
            error_text = reply.content if reply.content else "网络有点小繁忙，请过几秒再试一试"
            self._send_text(error_text, receiver, context_token)
            logger.info("[Weixin] sendError=%s, receiver=%s", error_text, receiver)
        elif reply_type == ReplyType.INFO:
            self._send_text(reply.content, receiver, context_token)
            logger.info("[Weixin] sendInfo=%s, receiver=%s", reply, receiver)
        elif reply_type in (ReplyType.IMAGE_URL, ReplyType.IMAGE):
            self._send_image(reply.content, receiver, context_token)
            logger.info("[Weixin] sendImage=%s, receiver=%s", self._summarize_reply_content(reply.content), receiver)
        elif reply_type == ReplyType.FILE:
            self._send_file(reply.content, receiver, context_token)
            logger.info("[Weixin] sendFile=%s, receiver=%s", self._summarize_reply_content(reply.content), receiver)
        elif reply_type in (ReplyType.VIDEO, ReplyType.VIDEO_URL):
            self._send_video(reply.content, receiver, context_token)
            logger.info("[Weixin] sendVideo=%s, receiver=%s", self._summarize_reply_content(reply.content), receiver)
        elif reply_type == ReplyType.VOICE:
            # ilink has no outbound voice item; deliver TTS as a file attachment.
            self._send_file(reply.content, receiver, context_token)
            logger.info("[Weixin] sendVoiceAsFile=%s, receiver=%s", self._summarize_reply_content(reply.content), receiver)
        else:
            logger.warning("[Weixin] Unsupported reply type=%s, fallback to text", reply_type)
            self._send_text(str(reply.content), receiver, context_token)
            logger.info("[Weixin] sendFallbackText=%s, receiver=%s", self._summarize_reply_content(reply.content), receiver)

    @staticmethod
    def _summarize_reply_content(content, limit: int = 160) -> str:
        """Return a compact, log-safe preview of reply content."""
        if content is None:
            return ""
        preview = str(content).replace("\n", "\\n")
        if len(preview) > limit:
            preview = preview[:limit] + "..."
        return preview

    def _get_context_token(self, receiver: str, msg=None) -> str:
        """Get the context_token for a receiver, required for all sends."""
        if msg and hasattr(msg, "context_token") and msg.context_token:
            return msg.context_token
        return self._context_tokens.get(receiver, "")

    def _check_send_response(self, resp, receiver: str) -> None:
        """Inspect a send-API response; drop stale context_token on -14.

        ilink uses ret/errcode = -14 to signal that the session (and any
        cached context_token) is no longer valid. The plugin keeps running
        because the bot itself can re-login; we just need to forget the
        per-user token so the next push won't retry forever.
        """
        if not isinstance(resp, dict):
            return
        ret = resp.get("ret")
        errcode = resp.get("errcode")
        if ret == -14 or errcode == -14:
            logger.warning(
                f"[Weixin] Send returned -14 (session expired) for "
                f"receiver={receiver}; dropping cached context_token"
            )
            self._invalidate_context_token(receiver)

    def _send_text(self, text: str, receiver: str, context_token: str):
        if len(text) <= TEXT_CHUNK_LIMIT:
            try:
                resp = self.api.send_text(receiver, text, context_token)
                self._check_send_response(resp, receiver)
                logger.debug(f"[Weixin] Text sent to {receiver}, len={len(text)}")
            except Exception as e:
                logger.error(f"[Weixin] Failed to send text: {e}")
            return

        chunks = self._split_text(text, TEXT_CHUNK_LIMIT)
        for i, chunk in enumerate(chunks):
            try:
                resp = self.api.send_text(receiver, chunk, context_token)
                self._check_send_response(resp, receiver)
                logger.debug(f"[Weixin] Text chunk {i+1}/{len(chunks)} sent to {receiver}, len={len(chunk)}")
            except Exception as e:
                logger.error(f"[Weixin] Failed to send text chunk {i+1}/{len(chunks)}: {e}")
                break
            if i < len(chunks) - 1:
                time.sleep(0.5)

    @staticmethod
    def _split_text(text: str, limit: int) -> list:
        """Split text into chunks, preferring to break at paragraph or line boundaries."""
        if len(text) <= limit:
            return [text]
        chunks = []
        while text:
            if len(text) <= limit:
                chunks.append(text)
                break
            cut = text.rfind("\n\n", 0, limit)
            if cut <= 0:
                cut = text.rfind("\n", 0, limit)
            if cut <= 0:
                cut = limit
            chunks.append(text[:cut])
            text = text[cut:].lstrip("\n")
        return chunks

    def _send_image(self, img_path_or_url, receiver: str, context_token: str):
        if isinstance(img_path_or_url, (list, tuple)):
            for image_item in img_path_or_url:
                self._send_image(image_item, receiver, context_token)
            return

        local_path = self._resolve_media_path(img_path_or_url, receiver)
        if not local_path:
            self._send_text("[Image send failed: file not found]", receiver, context_token)
            return
        try:
            result = upload_media_to_cdn(self.api, local_path, receiver, media_type=1)
            resp = self.api.send_image_item(
                to=receiver,
                context_token=context_token,
                encrypt_query_param=result["encrypt_query_param"],
                aes_key_b64=result["aes_key_b64"],
                ciphertext_size=result["ciphertext_size"],
            )
            self._check_send_response(resp, receiver)
            logger.info(f"[Weixin] Image sent to {receiver}")
        except Exception as e:
            logger.error(f"[Weixin] Image send failed: {e}")
            self._send_text("[Image send failed]", receiver, context_token)

    def _send_file(self, file_path_or_url: str, receiver: str, context_token: str):
        local_path = self._resolve_media_path(file_path_or_url, receiver)
        if not local_path:
            self._send_text("[File send failed: file not found]", receiver, context_token)
            return
        try:
            result = upload_media_to_cdn(self.api, local_path, receiver, media_type=3)
            resp = self.api.send_file_item(
                to=receiver,
                context_token=context_token,
                encrypt_query_param=result["encrypt_query_param"],
                aes_key_b64=result["aes_key_b64"],
                file_name=os.path.basename(local_path),
                file_size=result["raw_size"],
            )
            self._check_send_response(resp, receiver)
            logger.info(f"[Weixin] File sent to {receiver}")
        except Exception as e:
            logger.error(f"[Weixin] File send failed: {e}")
            self._send_text("[File send failed]", receiver, context_token)

    def _send_video(self, video_content, receiver: str, context_token: str):
        local_path = self._resolve_video_path(video_content, receiver, context_token)
        if not local_path:
            self._send_text("[Video send failed: file not found]", receiver, context_token)
            return
        try:
            result = upload_media_to_cdn(self.api, local_path, receiver, media_type=2)
            resp = self.api.send_video_item(
                to=receiver,
                context_token=context_token,
                encrypt_query_param=result["encrypt_query_param"],
                aes_key_b64=result["aes_key_b64"],
                ciphertext_size=result["ciphertext_size"],
            )
            self._check_send_response(resp, receiver)
            logger.info(f"[Weixin] Video sent to {receiver}")
        except Exception as e:
            logger.error(f"[Weixin] Video send failed: {e}")
            self._send_text("[Video send failed]", receiver, context_token)

    def _resolve_video_path(self, video_content, receiver: str, context_token: str) -> str:
        """Normalize VIDEO/VIDEO_URL reply content to a local video path."""
        if isinstance(video_content, (tuple, list)) and len(video_content) >= 2:
            # VIDEO_URL replies are shaped like (duration, url), matching Feishu.
            video_content = video_content[1]

        if hasattr(video_content, "video_bytes"):
            tmp_path = f"/tmp/wx_video_{uuid.uuid4().hex[:8]}.mp4"
            try:
                with open(tmp_path, "wb") as f:
                    f.write(video_content.video_bytes)
                return tmp_path
            except Exception as e:
                logger.error(f"[Weixin] Failed to save generated video bytes: {e}")
                return ""

        if not isinstance(video_content, str):
            logger.error(
                "[Weixin] Unsupported video content type=%s, content=%s",
                type(video_content).__name__,
                self._summarize_reply_content(video_content),
            )
            return ""

        return self._resolve_media_path(video_content, receiver)

    def _resolve_media_path(self, path_or_url, receiver: str = "") -> str:
        """Resolve a file path or URL to a local file path. Downloads if needed."""
        if not path_or_url:
            return ""

        if isinstance(path_or_url, (bytes, bytearray)):
            return self._save_media_bytes(bytes(path_or_url), receiver)

        if isinstance(path_or_url, io.BytesIO):
            try:
                pos = path_or_url.tell()
            except Exception:
                pos = None
            try:
                path_or_url.seek(0)
                media_bytes = path_or_url.read()
            except Exception as e:
                logger.error(f"[Weixin] Failed to read in-memory media: {e}")
                return ""
            finally:
                if pos is not None:
                    try:
                        path_or_url.seek(pos)
                    except Exception:
                        pass
            return self._save_media_bytes(media_bytes, receiver)

        if hasattr(path_or_url, "read"):
            try:
                media_bytes = path_or_url.read()
            except Exception as e:
                logger.error(f"[Weixin] Failed to read file-like media: {e}")
                return ""
            return self._save_media_bytes(media_bytes, receiver)

        if not isinstance(path_or_url, str):
            logger.error(
                "[Weixin] Unsupported media content type=%s, content=%s",
                type(path_or_url).__name__,
                self._summarize_reply_content(path_or_url),
            )
            return ""

        local_path = path_or_url
        if local_path.startswith("file://"):
            local_path = local_path[7:]

        if local_path.startswith(("http://", "https://")):
            try:
                resp = requests.get(local_path, stream=True, timeout=180)
                resp.raise_for_status()
                ct = resp.headers.get("Content-Type", "")
                ext = ".bin"
                if "jpeg" in ct or "jpg" in ct:
                    ext = ".jpg"
                elif "png" in ct:
                    ext = ".png"
                elif "gif" in ct:
                    ext = ".gif"
                elif "webp" in ct:
                    ext = ".webp"
                elif "mp4" in ct:
                    ext = ".mp4"
                elif "pdf" in ct:
                    ext = ".pdf"

                tmp_path = self._build_response_media_path(receiver, ext)
                with open(tmp_path, "wb") as f:
                    for block in resp.iter_content(1024 * 1024):
                        if block:
                            f.write(block)
                return tmp_path
            except Exception as e:
                logger.error(f"[Weixin] Failed to download media: {e}")
                return ""

        if os.path.exists(local_path):
            return local_path

        logger.warning(f"[Weixin] Media file not found: {local_path}")
        return ""

    def _save_media_bytes(self, media_bytes: bytes, receiver: str = "") -> str:
        if not media_bytes:
            return ""
        ext = self._guess_media_extension(media_bytes)
        tmp_path = self._build_response_media_path(receiver, ext)
        try:
            with open(tmp_path, "wb") as f:
                f.write(media_bytes)
            return tmp_path
        except Exception as e:
            logger.error(f"[Weixin] Failed to save in-memory media: {e}")
            return ""

    def _guess_media_extension(self, media_bytes: bytes) -> str:
        if media_bytes.startswith(b"\xff\xd8\xff"):
            return ".jpg"
        if media_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
            return ".png"
        if media_bytes.startswith(b"GIF87a") or media_bytes.startswith(b"GIF89a"):
            return ".gif"
        if media_bytes.startswith(b"RIFF") and media_bytes[8:12] == b"WEBP":
            return ".webp"
        if media_bytes.startswith(b"%PDF"):
            return ".pdf"
        if len(media_bytes) > 12 and media_bytes[4:8] == b"ftyp":
            return ".mp4"
        return ".bin"

    def _build_response_media_path(self, receiver: str, ext: str) -> str:
        channel_type = getattr(self, "channel_type", None) or conf().get("channel_type", "wx")
        response_dir = get_response_dir(channel_type, receiver or "unknown")
        return os.path.join(response_dir, f"{uuid.uuid4()}{ext}")
