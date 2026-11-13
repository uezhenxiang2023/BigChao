import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path

import httpx
from anthropic import Anthropic, APITimeoutError
from bot.bot import Bot
from bot.claude.claude_ai_session import ClaudeAiSession
from bot.session_manager import SessionManager
from bridge.context import Context, ContextType
from bridge.reply import Reply, ReplyType
from common.log import logger
from common import memory
from common.media_store import build_public_media_url
from config import conf

from common.tool_button import tool_state
from common.model_status import model_state


class ClaudeAIBot(Bot):
    _CLAUDE_FILES_API_BETA = "files-api-2025-04-14"
    _CLAUDE_BASE64_PDF_LIMIT_BYTES = 32 * 1024 * 1024
    _CLAUDE_OFFICIAL_BASE_URL = "https://api.anthropic.com"

    def __init__(self):
        super().__init__()
        connect_timeout = float(conf().get("anthropic_connect_timeout") or 15.0)
        read_timeout = float(conf().get("anthropic_timeout") or 300.0)
        client_timeout = httpx.Timeout(timeout=read_timeout, connect=connect_timeout)
        self.client = Anthropic(
            api_key=conf().get("anthropic_api_key"),
            base_url=conf().get("anthropic_base_url"),
            timeout=client_timeout,
            max_retries=2,
        )
        self.sessions = SessionManager(ClaudeAiSession, model=conf().get("model") or "gpt-3.5-turbo")
        self.system_prompt = conf().get("character_desc")
        self.claude_api_cookie = conf().get("anthropic_api_cookie")
        self.stream = conf().get("stream")
        self.proxy = conf().get("proxy")

    def reply(self, query, context: Context = None) -> Reply:
        if context.type == ContextType.IMAGE:
            return self.claude_vision(query, context)
        elif context.type == ContextType.FILE:
            return self._file_cache(query, context)
        elif context.type == ContextType.TEXT:
            return self._chat(query, context)
        elif context.type == ContextType.IMAGE_CREATE:
            ok, res = self.create_img(query, 0)
            if ok:
                reply = Reply(ReplyType.IMAGE_URL, res)
            else:
                reply = Reply(ReplyType.ERROR, res)
            return reply
        else:
            reply = Reply(ReplyType.ERROR, "Bot不支持处理{}类型的消息".format(context.type))
            return reply
        
    def _chat(self, query, context, retry_count=0) -> Reply:
        """
        发起对话请求
        :param query: 请求提示词
        :param context: 对话上下文
        :param retry_count: 当前递归重试次数
        :return: 回复
        """
        session_id = context["session_id"]
        self.model = model_state.get_basic_state(session_id)
        self.Model_ID = self.model.upper()
        if retry_count >= 2:
            # exit from retry 2 times
            logger.warning(f"[{self.Model_ID}] failed after maximum number of retry times")
            return Reply(ReplyType.ERROR, "请再问我一次吧")

        try:
            # 先构建多模态 content 块（含媒体+文本）
            current_content = self._build_current_content(query, session_id)
            logger.info(f"[{self.Model_ID}] query={query}")

            # 将多模态 content 块写入 session，而不是纯字符串 query
            #
            self.sessions.session_query(current_content, session_id)  # ← 传入列表，保留媒体块
            
            # 从 session 获取完整消息历史
            session = self.sessions.build_session(session_id)
            claude_message = self._sanitize_claude_session_messages(session.messages)

            # 判断联网搜索状态
            is_searching = tool_state.get_search_state(session_id)

            has_mcp = (
                conf().get("blender_mcp_enabled", False)
                or conf().get("kling_mcp_enabled", False)
                or conf().get("unreal_mcp_enabled", False)
            )

            create_kwargs = dict(
                model=self.model,
                max_tokens=4096 if has_mcp else 1000,
                temperature=0.0,
                system=self.system_prompt or "",
                messages=claude_message
            )

            if has_mcp:
                return self._chat_with_mcp(create_kwargs, session_id, current_content)
            
            if is_searching:
                create_kwargs["tools"] = [
                    {
                        "type": "web_search_20250305", 
                        "name": "web_search",
                        "max_uses": 5
                    }
                ]
            if is_searching and not self.stream:
                # 搜索开启 + stream 关闭 → IMAGE_URL 模式
                response = self._create_claude_message(create_kwargs, current_content)
                # 提取文本回复
                reply_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        reply_content += block.text

                logger.info(f"[{self.Model_ID}] reply={reply_content}, total_tokens=invisible")
                self.sessions.session_reply(reply_content, session_id, 100)
                return Reply(ReplyType.IMAGE_URL, response)
            
            elif self.stream:
                # stream 开启 → STREAM 模式
                logger.info(f"[{self.Model_ID}] stream 模式已开启")
                
                def stream_generator():
                    full_text = ""
                    with self._stream_claude_message(create_kwargs, current_content) as s:
                        for text_chunk in s.text_stream:
                            full_text += text_chunk
                            yield text_chunk
                        # 流结束，写回 session
                        final_message = s.get_final_message()
                        total_tokens = final_message.usage.output_tokens
                        self.sessions.session_reply(full_text, session_id, total_tokens)
                        logger.info(f"[{self.Model_ID}] stream 完成, session_id={session_id}, tokens={total_tokens}")
                        # 搜索开启时，从 final_message 提取 citations yield 出去
                        if is_searching:
                            yield final_message

                return Reply(ReplyType.STREAM, stream_generator())

            else:
                # 普通模式
                response = self._create_claude_message(create_kwargs, current_content)
                reply_content = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        reply_content += block.text
                logger.info(f"[{self.Model_ID}] reply={reply_content}")
                self.sessions.session_reply(reply_content, session_id, 100)
                return Reply(ReplyType.TEXT, reply_content)

        except Exception as e:
            if self._is_timeout_error(e):
                logger.warning(f"[{self.Model_ID}] fetch reply timeout: {e}")
                return Reply(ReplyType.TEXT, "网络忙，请稍后重试")
            real_err = self._unwrap_exception(e)
            logger.error("[{}] fetch reply error, {}".format(self.Model_ID, real_err))
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] {real_err}")

    @staticmethod
    def _is_timeout_error(e: BaseException) -> bool:
        """递归检查异常或 ExceptionGroup 是否包含超时错误（连接超时、请求超时等）。"""
        if isinstance(e, (APITimeoutError, httpx.TimeoutException, TimeoutError)):
            return True
        if hasattr(e, "exceptions"):
            for sub in getattr(e, "exceptions", []):
                if ClaudeAIBot._is_timeout_error(sub):
                    return True
        if getattr(e, "__cause__", None) and ClaudeAIBot._is_timeout_error(e.__cause__):
            return True
        if getattr(e, "__context__", None) and ClaudeAIBot._is_timeout_error(e.__context__):
            return True
        msg = str(e).lower()
        return "timed out" in msg or "timeout" in msg

    @staticmethod
    def _unwrap_exception(e: BaseException) -> BaseException:
        """从 ExceptionGroup 中解包出核心子异常。"""
        curr = e
        while hasattr(curr, "exceptions") and curr.exceptions:
            curr = curr.exceptions[0]
        return curr

    def _chat_with_mcp(self, create_kwargs, session_id, current_content=None):
        """通过 MCPHub 在一条 Claude 消息内执行完整的 MCP 工具调用循环。"""
        try:
            reply_content, output_tokens, video_info = asyncio.run(
                self._run_mcp_chat(create_kwargs, session_id, current_content)
            )
            self.sessions.session_reply(reply_content, session_id, output_tokens)
            logger.info(
                f"[{self.Model_ID}] MCP reply={reply_content}, "
                f"session_id={session_id}, tokens={output_tokens}"
            )

            if video_info and video_info.get("video_url"):
                video_url = video_info["video_url"]
                duration = video_info.get("duration", 5)
                from bot.mcp.hub import MCPHub
                clean_reply_content = MCPHub.strip_media_links(reply_content)
                return Reply(ReplyType.VIDEO_URL, (duration, video_url, clean_reply_content))

            return Reply(ReplyType.TEXT, reply_content)
        except Exception as e:
            if self._is_timeout_error(e):
                logger.warning(f"[{self.Model_ID}] MCP request timeout: {e}")
                return Reply(ReplyType.TEXT, "网络忙，请稍后重试")
            real_err = self._unwrap_exception(e)
            logger.exception(f"[{self.Model_ID}] MCP request failed: {real_err}")
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] MCP 调用失败：{real_err}")

    async def _run_mcp_chat(self, create_kwargs, session_id, current_content=None):
        """通过 MCPHub 聚合所有启用的 MCP 工具（Blender, Kling, Unreal），由 Claude 调度执行。"""
        from bot.mcp.hub import mcp_hub

        messages = list(create_kwargs["messages"])
        context = {"session_id": session_id, "messages": messages}

        async with mcp_hub.session(session_id=session_id, context=context) as mcp:
            if not mcp.has_tools:
                logger.info(f"[{self.Model_ID}] 无可用 MCP 工具，回退为普通会话")
                response = self._create_claude_message(create_kwargs, current_content)
                reply_content = "".join([b.text for b in response.content if hasattr(b, "text")])
                return reply_content, getattr(getattr(response, "usage", None), "output_tokens", 100), None

            # 追加 Hub 聚合 instructions 到系统提示词
            if mcp.instructions:
                current_system = create_kwargs.get("system") or ""
                create_kwargs["system"] = (current_system + "\n\n" + mcp.instructions).strip()

            tool_definitions = mcp.tool_definitions
            max_rounds = mcp.max_tool_rounds

            captured_video = None
            for round_idx in range(max_rounds):
                response = self.client.messages.create(
                    **dict(create_kwargs, messages=messages, tools=tool_definitions)
                )
                response_blocks = self._serialize_anthropic_blocks(response.content)
                tool_uses = [
                    block for block in response_blocks
                    if block.get("type") == "tool_use"
                ]

                if not tool_uses:
                    captured_video = await mcp.finalize()
                    final_text = self._extract_claude_text(response_blocks)
                    output_tokens = response.usage.output_tokens
                    return final_text, output_tokens, captured_video

                messages.append({"role": "assistant", "content": response_blocks})
                tool_results = []
                for tool_use in tool_uses:
                    tool_name = tool_use["name"]
                    tool_input = tool_use.get("input", {})
                    serialized_res, is_error = await mcp.call_tool(tool_name, tool_input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "is_error": is_error,
                        "content": serialized_res,
                    })

                messages.append({"role": "user", "content": tool_results})

            raise RuntimeError(f"MCP 工具调用超过最大轮次（{max_rounds}）")

    @staticmethod
    def _serialize_anthropic_blocks(blocks):
        return [
            block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            for block in blocks
        ]

    @staticmethod
    def _sanitize_claude_session_messages(messages):
        """将历史媒体转换为 Claude Messages API 支持的格式，避免后续请求被 400 拒绝。"""
        converted_image_count = 0
        converted_video_count = 0
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            if not isinstance(content, list):
                continue
            sanitized_content = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    data_url = block.get("image_url", {}).get("url", "")
                    data_url_match = re.fullmatch(
                        r"data:(image/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)",
                        data_url,
                    )
                    if data_url_match:
                        sanitized_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": data_url_match.group(1),
                                "data": data_url_match.group(2),
                            },
                        })
                        converted_image_count += 1
                    else:
                        sanitized_content.append({
                            "type": "text",
                            "text": "[历史图片已生成并发送给用户]",
                        })
                        converted_image_count += 1
                elif isinstance(block, dict) and block.get("type") == "video_url":
                    # Claude Messages API 目前没有 video_url 输入块。保留视频已生成的
                    # 上下文提示，避免把 OpenAI 风格的块直接传给 Claude 导致 400。
                    sanitized_content.append({
                        "type": "text",
                        "text": "[历史视频已生成并发送给用户]",
                    })
                    converted_video_count += 1
                else:
                    sanitized_content.append(block)
            message["content"] = sanitized_content
        if converted_image_count or converted_video_count:
            logger.info(
                "[Claude] normalized historical media: images=%s, videos_as_text=%s",
                converted_image_count,
                converted_video_count,
            )
        return messages

    @staticmethod
    def _serialize_mcp_result(result):
        content = []
        for block in getattr(result, "content", []) or []:
            if hasattr(block, "model_dump"):
                content.append(block.model_dump(exclude_none=True))
            elif isinstance(block, dict):
                content.append(block)
            else:
                content.append({"type": "text", "text": str(block)})
        payload = {"content": content}
        structured_content = getattr(result, "structured_content", getattr(result, "structuredContent", None))
        if structured_content is not None:
            payload["structured_content"] = structured_content
        return json.dumps(payload, ensure_ascii=False, default=str)

    @staticmethod
    def _extract_claude_text(blocks):
        text = "".join(
            block.get("text", "") for block in blocks
            if block.get("type") == "text"
        )
        return text or "操作已完成。"
        
    def _build_current_content(self, query: str, session_id: str) -> list:
        """
        构建当前轮次的多模态 content 块列表。
        媒体块在前，文本在后。消费并清除缓存。
        """
        current_content = []

        image_cache = memory.USER_IMAGE_CACHE.get(session_id)
        if image_cache:
            image_paths = image_cache.get("path", [])
            for image_index, cached_file in enumerate(image_cache.get("files", []), start=1):
                data_type = type(cached_file).__name__

                if data_type in ("JpegImageFile", "PngImageFile", "Image"):
                    try:
                        img = cached_file
                        if img.mode in ("RGBA", "LA", "P"):
                            img = img.convert("RGB")
                            fmt, media_type = "JPEG", "image/jpeg"
                        elif data_type == "PngImageFile":
                            fmt, media_type = "PNG", "image/png"
                        else:
                            fmt, media_type = "JPEG", "image/jpeg"

                        buf = io.BytesIO()
                        img.save(buf, format=fmt)
                        img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

                        current_content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": media_type,
                                "data": img_b64
                            }
                        })
                        logger.debug(f"[{self.Model_ID}] image block added, format={fmt}")

                        # 可灵 MCP 的图生视频工具需要由服务端可访问的 http(s) URL，
                        # 不能使用发给 Claude 识图的 base64 数据。项目已有的本地媒体
                        # 服务会将 tmp 文件通过 media_public_base_url（例如 ngrok）公开。
                        image_path = image_paths[image_index - 1] if image_index <= len(image_paths) else None
                        public_url = build_public_media_url(image_path) if image_path else None
                        if public_url:
                            current_content.append({
                                "type": "text",
                                "text": (
                                    f"[参考图 {image_index} 的可灵公网图片地址：{public_url}。"
                                    "若调用 Kling 的 image_to_video、omni_ref_video 或其他需要图片 URL 的工具，"
                                    "请直接使用此地址，勿要求用户再次提供链接。]"
                                ),
                            })
                            logger.info(
                                "[%s] Kling reference image %s public_url=%s",
                                self.Model_ID,
                                image_index,
                                public_url,
                            )
                        else:
                            logger.warning(
                                "[%s] could not build public URL for Kling reference image: %s",
                                self.Model_ID,
                                image_path,
                            )
                    except Exception as e:
                        logger.error(f"[{self.Model_ID}] failed to encode image: {e}")
                else:
                    logger.warning(f"[{self.Model_ID}] unsupported cached image type: {data_type}")

            memory.USER_IMAGE_CACHE.pop(session_id)

        # Claude 的消息 API 不直接接收本地视频文件；可灵工具则需要服务端能访问的
        # http(s) 地址。飞书通道已将上传/引用视频缓存为 path + public_url，在此统一
        # 注入公网地址，供模型直接传给 Kling 的视频输入参数。
        video_cache = memory.USER_VIDEO_CACHE.get(session_id)
        self._append_kling_video_references(current_content, video_cache, "参考视频")
        if video_cache:
            memory.USER_VIDEO_CACHE.pop(session_id, None)

        quoted_video_cache = memory.USER_QUOTED_VIDEO_CACHE.get(session_id)
        self._append_kling_video_references(current_content, quoted_video_cache, "引用视频")
        if quoted_video_cache:
            memory.USER_QUOTED_VIDEO_CACHE.pop(session_id, None)

        file_cache = memory.USER_FILE_CACHE.get(session_id)
        if file_cache:
            for cached_file in file_cache.get("files", []):
                if not isinstance(cached_file, dict):
                    logger.warning(f"[{self.Model_ID}] unsupported cached file type: {type(cached_file).__name__}")
                    continue

                mime_type = cached_file.get("mime_type", "")
                raw_data = cached_file.get("data", "")
                file_path = cached_file.get("path", "")

                if mime_type == "application/pdf":
                    pdf_block = self._build_pdf_document_block(file_path, raw_data)
                    if pdf_block:
                        current_content.append(pdf_block)

                elif mime_type in ("application/docx", "application/doc", "application/plain"):
                    current_content.append({
                        "type": "text",
                        "text": f"<document>\n{raw_data}\n</document>"
                    })
                    logger.debug(f"[{self.Model_ID}] document text block added, mime_type={mime_type}")

                else:
                    logger.warning(f"[{self.Model_ID}] unsupported mime_type: {mime_type}")

            memory.USER_FILE_CACHE.pop(session_id)

        # 文本块追加在末尾
        current_content.append({
            "type": "text",
            "text": query
        })

        return current_content

    def _append_kling_video_references(self, current_content: list, video_cache, label: str):
        """将飞书上传或引用的视频公网地址作为文本块提供给 Claude/Kling。"""
        if not video_cache:
            return

        for video_index, cached_file in enumerate(video_cache.get("files", []), start=1):
            if not isinstance(cached_file, dict):
                logger.warning(
                    "[%s] unsupported cached video type: %s",
                    getattr(self, "Model_ID", "CLAUDE"),
                    type(cached_file).__name__,
                )
                continue

            video_path = cached_file.get("path")
            public_url = cached_file.get("public_url") or build_public_media_url(video_path)
            if not public_url:
                logger.warning(
                    "[%s] could not build public URL for Kling %s: %s",
                    getattr(self, "Model_ID", "CLAUDE"),
                    label,
                    video_path,
                )
                continue

            current_content.append({
                "type": "text",
                "text": (
                    f"[{label} {video_index} 的可灵公网视频地址：{public_url}。"
                    "若调用 Kling 的 motion_control、omni_ref_video 或其他需要视频 URL 的工具，"
                    "请直接使用此地址，勿要求用户再次提供链接。]"
                ),
            })
            logger.info(
                "[%s] Kling %s %s public_url=%s",
                getattr(self, "Model_ID", "CLAUDE"),
                label,
                video_index,
                public_url,
            )

    def _build_pdf_document_block(self, file_path: str, raw_data: str):
        file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
        if file_size > self._CLAUDE_BASE64_PDF_LIMIT_BYTES:
            if not self._supports_claude_files_api():
                raise ValueError("当前 Claude 代理不支持 Files API，大于 32MB 的 PDF 请切换到官方 Claude 接口后再试。")
            file_upload = self._upload_pdf_to_claude(file_path)
            if not file_upload:
                return None
            logger.info(f"[{self.Model_ID}] PDF exceeds 32MB, use Claude Files API, file_id={file_upload.id}")
            return {
                "type": "document",
                "source": {
                    "type": "file",
                    "file_id": file_upload.id,
                }
            }

        logger.debug(f"[{self.Model_ID}] PDF document block added")
        return {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": raw_data
            }
        }

    def _upload_pdf_to_claude(self, file_path: str):
        with open(file_path, "rb") as pdf_file:
            return self.client.beta.files.upload(
                file=(os.path.basename(file_path), pdf_file, "application/pdf"),
                betas=[self._CLAUDE_FILES_API_BETA]
            )

    def _supports_claude_files_api(self):
        base_url = str(conf().get("anthropic_base_url") or self._CLAUDE_OFFICIAL_BASE_URL).rstrip("/")
        return base_url == self._CLAUDE_OFFICIAL_BASE_URL

    def _contains_file_document(self, content_blocks):
        if not isinstance(content_blocks, list):
            return False
        for block in content_blocks:
            if block.get("type") != "document":
                continue
            source = block.get("source", {})
            if source.get("type") == "file":
                return True
        return False

    def _requires_files_api(self, current_content, session_messages):
        if self._contains_file_document(current_content):
            return True
        for message in session_messages:
            if self._contains_file_document(message.get("content")):
                return True
        return False

    def _create_claude_message(self, create_kwargs, current_content):
        session_messages = create_kwargs.get("messages", [])
        if self._requires_files_api(current_content, session_messages):
            return self.client.beta.messages.create(
                **create_kwargs,
                betas=[self._CLAUDE_FILES_API_BETA]
            )
        return self.client.messages.create(**create_kwargs)

    def _stream_claude_message(self, create_kwargs, current_content):
        session_messages = create_kwargs.get("messages", [])
        if self._requires_files_api(current_content, session_messages):
            return self.client.beta.messages.stream(
                **create_kwargs,
                betas=[self._CLAUDE_FILES_API_BETA]
            )
        return self.client.messages.stream(**create_kwargs)
