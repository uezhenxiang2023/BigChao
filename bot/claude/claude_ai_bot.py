import asyncio
import base64
import io
import json
import os
import re
from pathlib import Path

from anthropic import Anthropic
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
        self.client = Anthropic(
            api_key=conf().get("anthropic_api_key"),
            base_url=conf().get("anthropic_base_url")
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

            blender_mcp_enabled = conf().get("blender_mcp_enabled", False)
            kling_mcp_enabled = conf().get("kling_mcp_enabled", False)
            kling_ready = False
            if kling_mcp_enabled:
                from bot.kling.kling_mcp_auth import get_kling_oauth_manager
                kling_ready = get_kling_oauth_manager().has_valid_tokens()

            system_prompt = self.system_prompt or ""
            if kling_mcp_enabled and kling_ready:
                system_prompt += (
                    "\n\n[MCP 工具调用重要规范]\n"
                    "你已成功连接可灵（Kling）官方 MCP 服务，挂载了 text_to_video（文生视频）、image_to_video（图生视频）、"
                    "text_to_image（文生图）等工具能力。\n"
                    "1. 当用户明确要求制作、生成或想要查看视频/图片时，你可以根据对话内容自动润色分镜提示词并【直接调用对应的可灵工具】完成生成。\n"
                    "2. 绝对不要引导或提示用户使用 '//' 或 '/' 等前缀（前缀为旧系统机制，当前已升级为全自然语言 Agent 调用）。\n"
                    "3. 如果用户只是咨询视效技术、探讨剧本或闲聊，未明确要求生成，则只提供文字讨论，不触发工具。\n"
                    "4. 调用 text_to_video 时，请根据用户意图合理填充 prompt、aspect_ratio、duration 等参数。\n"
                    "5. 【关键——异步任务轮询】text_to_video、image_to_video、text_to_image 等生成类工具是异步的，"
                    "调用后只会返回 generationId（任务ID），不会直接返回视频/图片文件。"
                    "你**必须**紧接着调用 query_tasks 工具（传入该 generationId）来轮询任务状态，"
                    "直到任务完成（status 为 succeed/finished）并获取到最终的视频/图片 URL。"
                    "**不要在拿到 generationId 后就停下来给用户回复**，"
                    "必须持续轮询 query_tasks 直到获取到最终文件 URL。"
                    "如果 query_tasks 返回任务仍在处理中（status 为 submitted/processing/running），"
                    "你应继续再次调用 query_tasks 进行轮询。\n"
                    "6. 拿到最终视频/图片 URL 后，将 URL 包含在你的回复文本中，以便系统自动提取并下发给用户。\n"
                    "7. 用户上传或连续发送的参考图会在消息中附带“可灵公网图片地址”。图生视频或多图参考时，"
                    "必须将这些地址直接填入相应 Kling 工具的图片 URL 参数，绝不能要求用户再次提供 http/https 链接。\n"
                    "8. 用户上传或引用的视频会在消息中附带“可灵公网视频地址”。需要基于该视频进行动作控制、"
                    "视频参考或视频编辑时，必须将此地址直接填入 motion_control、omni_ref_video 或其他对应 Kling 工具的"
                    "视频 URL 参数；绝不能忽略该地址或要求用户再次提供链接。"
                )

            create_kwargs = dict(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                system=system_prompt,
                messages=claude_message
            )

            if blender_mcp_enabled or (kling_mcp_enabled and kling_ready):
                # MCP 模式下增大 max_tokens，因为多轮工具调用需要更多输出空间
                create_kwargs["max_tokens"] = 4096
                return self._chat_with_mcp(create_kwargs, session_id)
            
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
            logger.error("[{}] fetch reply error, {}".format(self.Model_ID, e))
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] {e}")

    def _chat_with_blender_mcp(self, create_kwargs, session_id):
        return self._chat_with_mcp(create_kwargs, session_id)

    def _chat_with_mcp(self, create_kwargs, session_id):
        """在一条 Claude 消息内执行完整的 MCP 工具调用循环（支持 Blender MCP 与 Kling MCP）。"""
        try:
            reply_content, output_tokens, video_info = asyncio.run(
                self._run_mcp_chat(create_kwargs, session_id)
            )
            self.sessions.session_reply(reply_content, session_id, output_tokens)
            logger.info(
                f"[{self.Model_ID}] MCP reply={reply_content}, "
                f"session_id={session_id}, tokens={output_tokens}"
            )

            # 如果检测到生成了可灵视频，作为视频回复下发。
            # session_reply 已保存本轮文本；不要调用通用 session_inject_media：其
            # video_url/image_url 块属于其他模型的格式，Claude Messages API 不接受。
            if video_info and video_info.get("video_url"):
                video_url = video_info["video_url"]
                duration = video_info.get("duration", 5)

                # VIDEO_URL 是单一 Reply 类型，不能同时再返回一个 TEXT Reply。
                # 把 Claude 的最终文本随视频一并带到 channel，由支持的通道先发送文本、
                # 再发送视频；保留前两个元素以兼容既有 VIDEO_URL 消费者。
                from bot.kling.kling_mcp_client import KlingMCPClient

                clean_reply_content = KlingMCPClient.strip_media_links(reply_content)
                return Reply(ReplyType.VIDEO_URL, (duration, video_url, clean_reply_content))

            return Reply(ReplyType.TEXT, reply_content)
        except Exception as e:
            logger.exception(f"[{self.Model_ID}] MCP request failed: {e}")
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] MCP 调用失败：{e}")

    async def _run_mcp_chat(self, create_kwargs, session_id):
        """聚合已启用的 MCP Server（Blender stdio、Kling Streamable HTTP 等）供 Claude 在本轮请求中按需调用。"""
        import contextlib
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise RuntimeError("缺少 MCP 客户端依赖，请执行 pip install -r requirements.txt") from e

        max_rounds = int(conf().get("blender_mcp_max_tool_rounds", 20))
        messages = list(create_kwargs["messages"])
        kling_reference_urls = []

        async with contextlib.AsyncExitStack() as stack:
            # mcp_tools: tool_name -> (session, is_kling)
            mcp_tools = {}
            tool_definitions = []

            # 1. 初始化 Blender MCP (若启用)
            if conf().get("blender_mcp_enabled"):
                try:
                    from bot.blender.blender_mcp_client import BlenderMCPClient
                    blender_client = BlenderMCPClient()
                    blender_connection = await stack.enter_async_context(blender_client.connect())
                    blender_tools = blender_connection.tool_definitions()
                    for tool in blender_tools:
                        mcp_tools[tool["name"]] = (blender_connection, False)
                    tool_definitions.extend(blender_tools)
                    logger.info(
                        f"[{self.Model_ID}] Blender MCP 已连接并载入工具: "
                        f"{[tool['name'] for tool in blender_tools]}"
                    )
                except Exception as e:
                    logger.exception(f"[{self.Model_ID}] 初始化 Blender MCP 失败: {e}")
                    raise RuntimeError(f"Blender MCP 服务连接失败：{e}")

            # 2. 初始化 Kling MCP (若启用且已完成授权)
            from bot.kling.kling_mcp_auth import get_kling_oauth_manager
            if conf().get("kling_mcp_enabled") and get_kling_oauth_manager().has_valid_tokens():
                try:
                    from bot.kling.kling_mcp_client import KlingMCPClient

                    # 用户可能先发参考图、再确认生成，因此从最近的历史消息提取图片地址。
                    kling_reference_urls = KlingMCPClient.extract_reference_urls(messages)
                    kling_client = KlingMCPClient()
                    kling_connection = await stack.enter_async_context(kling_client.connect())
                    kling_tools = kling_connection.tool_definitions()
                    for tool in kling_tools:
                        mcp_tools[tool["name"]] = (kling_connection, True)
                    tool_definitions.extend(kling_tools)
                    logger.info(
                        f"[{self.Model_ID}] Kling MCP 已连接并载入工具: "
                        f"{[tool['name'] for tool in kling_tools]}"
                    )
                except Exception as e:
                    logger.exception(f"[{self.Model_ID}] 初始化 Kling MCP 失败: {e}")
                    raise RuntimeError(f"可灵 MCP 服务连接失败：{e}")

            if not mcp_tools:
                raise RuntimeError("未启用或未获取到任何有效的 MCP 工具")

            captured_video = None
            # 跟踪可灵异步生成任务：记录 generationId 和最后一次 query_tasks 的参数
            _kling_pending_generation_id = None
            _kling_last_query_args = None
            _kling_generation_duration = 5  # 默认时长

            for round_idx in range(max_rounds):
                response = self.client.messages.create(
                    **dict(create_kwargs, messages=messages, tools=tool_definitions)
                )
                response_blocks = self._serialize_anthropic_blocks(response.content)
                tool_uses = [
                    block for block in response_blocks
                    if block.get("type") == "tool_use"
                ]

                # ── 关键：Claude 没有发起 tool_use，但我们知道有未完成的 Kling 生成任务 ──
                if not tool_uses:
                    if _kling_pending_generation_id and not captured_video and "query_tasks" in mcp_tools:
                        # Claude 放弃轮询了，但视频还没拿到 → 代码层面强制接管轮询
                        logger.info(
                            f"[{self.Model_ID}] Claude 停止了 tool_use，但 Kling 任务 "
                            f"{_kling_pending_generation_id} 尚未完成，启动代码层强制轮询..."
                        )
                        final_text = self._extract_claude_text(response_blocks)
                        output_tokens = response.usage.output_tokens

                        target_connection, _ = mcp_tools["query_tasks"]
                        captured_video = await target_connection.poll_task(
                            _kling_pending_generation_id,
                            duration=_kling_generation_duration,
                        )

                        return final_text, output_tokens, captured_video
                    else:
                        return self._extract_claude_text(response_blocks), response.usage.output_tokens, captured_video

                messages.append({"role": "assistant", "content": response_blocks})
                tool_results = []
                for tool_use in tool_uses:
                    tool_name = tool_use["name"]
                    if tool_name not in mcp_tools:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use["id"],
                            "is_error": True,
                            "content": f"未知的 MCP 工具：{tool_name}",
                        })
                        continue

                    target_session, is_kling = mcp_tools[tool_name]

                    # omni_ref_video 是多图全能参考入口。Claude 偶尔会只把部分
                    # 参考图填入工具参数；此时不要让可灵提交一个错误的任务，而是把
                    # 缺失信息作为 tool_result 返给 Claude，促使它补齐后重试。
                    if is_kling and tool_name == "omni_ref_video" and len(kling_reference_urls) > 1:
                        tool_input = tool_use.get("input", {})
                        submitted_urls = KlingMCPClient.extract_http_urls(tool_input)
                        missing_urls = KlingMCPClient.missing_reference_urls(
                            tool_input,
                            kling_reference_urls,
                        )
                        logger.info(
                            "[%s] Kling omni_ref_video reference URLs: expected=%s, submitted=%s",
                            self.Model_ID,
                            len(kling_reference_urls),
                            len(submitted_urls),
                        )
                        if missing_urls:
                            logger.warning(
                                "[%s] reject incomplete omni_ref_video references: missing=%s",
                                self.Model_ID,
                                len(missing_urls),
                            )
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use["id"],
                                "is_error": True,
                                "content": (
                                    "本轮收到 %s 张用户参考图，但 omni_ref_video 工具参数只包含 %s 张。"
                                    "请重新调用 omni_ref_video，并将每一张“可灵公网图片地址”全部原样填入"
                                    "多图参考参数后再提交。"
                                ) % (len(kling_reference_urls), len(kling_reference_urls) - len(missing_urls)),
                            })
                            continue

                    # 对可灵 query_tasks 轮询请求，加入等待间隔（视频生成需要 1-3+ 分钟）
                    if is_kling and tool_name == "query_tasks":
                        poll_interval = 15  # 秒
                        logger.info(f"[{self.Model_ID}] 可灵任务轮询中，等待 {poll_interval} 秒后查询...")
                        await asyncio.sleep(poll_interval)

                    result = await target_session.call_tool(tool_name, tool_use.get("input", {}))
                    from bot.blender.blender_mcp_client import BlenderMCPClient
                    serialized_res = (
                        KlingMCPClient.serialize_result(result)
                        if is_kling
                        else BlenderMCPClient.serialize_result(result)
                    )

                    if is_kling:
                        video_url, cover_url = KlingMCPClient.extract_video_info(serialized_res)
                        if video_url:
                            captured_video = {
                                "video_url": video_url,
                                "cover_url": cover_url,
                                "duration": tool_use.get("input", {}).get("duration", 5),
                            }
                            logger.info(f"[{self.Model_ID}] 捕获到可灵生成视频: {video_url}")

                        # 追踪全部可灵异步生成工具（全能参考 omni_ref_video 也会异步返回任务 ID）。
                        if tool_name in (
                            "text_to_video", "image_to_video", "omni_ref_video",
                            "motion_control", "text_to_image", "image_to_image",
                        ) and not video_url:
                            generation_id = KlingMCPClient.extract_generation_id(serialized_res)
                            if generation_id:
                                _kling_pending_generation_id = generation_id
                                _kling_generation_duration = tool_use.get("input", {}).get("duration", 5)
                                logger.info(f"[{self.Model_ID}] 追踪到可灵异步任务 generationId={_kling_pending_generation_id}")

                        # 记录 query_tasks 的参数，以便代码层强制轮询时复用
                        if tool_name == "query_tasks":
                            _kling_last_query_args = tool_use.get("input", {})
                            # 即使生成工具响应格式变化无法解析，也能从 Claude 发出的
                            # query_tasks 入参恢复任务 ID，保证 Claude 提前结束时可接管轮询。
                            if not video_url and not _kling_pending_generation_id:
                                _kling_pending_generation_id = KlingMCPClient.extract_generation_id(
                                    _kling_last_query_args
                                )
                                if _kling_pending_generation_id:
                                    logger.info(
                                        f"[{self.Model_ID}] 从 query_tasks 入参恢复可灵任务 "
                                        f"generationId={_kling_pending_generation_id}"
                                    )
                            # 如果 query_tasks 返回了视频，清除 pending 标记
                            if video_url:
                                _kling_pending_generation_id = None

                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": tool_use["id"],
                        "is_error": bool(getattr(result, "is_error", getattr(result, "isError", False))),
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
        return text or "Blender 操作已完成。"
        
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
