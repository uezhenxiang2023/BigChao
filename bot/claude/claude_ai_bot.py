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
            claude_message = session.messages

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
                    "6. 拿到最终视频/图片 URL 后，将 URL 包含在你的回复文本中，以便系统自动提取并下发给用户。"
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

            # 如果检测到生成了可灵视频，优先注入当前 session 上下文，并作为视频回复下发
            if video_info and video_info.get("video_url"):
                video_url = video_info["video_url"]
                duration = video_info.get("duration", 5)
                try:
                    from common.utils import url_to_base64
                    base64_data = url_to_base64(video_url)
                    self.sessions.session_inject_media(
                        session_id=session_id,
                        media_type="video",
                        data=base64_data,
                        source_model="kling-mcp",
                        remote_url=video_url
                    )
                    logger.info(f"[{self.Model_ID}] Kling MCP 视频已注入 session={session_id}")
                except Exception as e:
                    logger.warning(f"[{self.Model_ID}] 注入视频至 session 异常: {e}")

                # VIDEO_URL 是单一 Reply 类型，不能同时再返回一个 TEXT Reply。
                # 把 Claude 的最终文本随视频一并带到 channel，由支持的通道先发送文本、
                # 再发送视频；保留前两个元素以兼容既有 VIDEO_URL 消费者。
                clean_reply_content = self._strip_kling_media_links(reply_content)
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

        async with contextlib.AsyncExitStack() as stack:
            # mcp_tools: tool_name -> (session, is_kling)
            mcp_tools = {}
            tool_definitions = []

            # 1. 初始化 Blender MCP (若启用)
            if conf().get("blender_mcp_enabled"):
                command = conf().get("blender_mcp_command")
                if not command:
                    raise ValueError("未配置 blender_mcp_command")

                mcp_cwd = conf().get("blender_mcp_cwd") or None
                configured_args = conf().get("blender_mcp_args") or ["-m", "blmcp"]

                if mcp_cwd and configured_args == ["-m", "blmcp"]:
                    runner_script = os.path.join(mcp_cwd, "run_mcp_entrypoint.py")
                    try:
                        with open(runner_script, "w", encoding="utf-8") as f:
                            f.write(
                                "import os, sys\n"
                                "from pathlib import Path\n"
                                "here = Path(__file__).resolve().parent\n"
                                "venv_site = here / '.venv' / 'Lib' / 'site-packages'\n"
                                "if venv_site.exists():\n"
                                "    sys.path.insert(0, str(venv_site))\n"
                                "sys.path.insert(0, str(here))\n"
                                "os.environ['PYDEVD_DISABLE_FILE_VALIDATION'] = '1'\n"
                                "os.environ['PYDEVD_DONT_TRACE_CHILD_PROCESSES'] = '1'\n"
                                "from blmcp import main\n"
                                "if __name__ == '__main__':\n"
                                "    sys.exit(main())\n"
                            )
                        server_args = [runner_script]
                    except Exception as e:
                        logger.warning(f"[{self.Model_ID}] 写入 MCP entrypoint 失败，回退到默认参数: {e}")
                        server_args = configured_args
                else:
                    server_args = configured_args

                if mcp_cwd:
                    _venv_site = str(Path(conf().get("blender_mcp_command", "")).parent.parent / "Lib" / "site-packages")
                    server_env = {
                        "PYTHONPATH": _venv_site + os.pathsep + mcp_cwd,
                        "PYDEVD_DISABLE_FILE_VALIDATION": "1",
                        "PYDEVD_DONT_TRACE_CHILD_PROCESSES": "1",
                    }
                else:
                    server_env = None

                server_params = StdioServerParameters(
                    command=command,
                    args=server_args,
                    cwd=mcp_cwd,
                    env=server_env,
                )
                stderr_log_path = os.path.join(os.getcwd(), "logs", "blender_mcp_stderr.log")
                os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

                stderr_log = stack.enter_context(open(stderr_log_path, "a", encoding="utf-8"))
                stderr_log.write("\n--- Blender MCP session started ---\n")
                stderr_log.flush()

                read_stream, write_stream = await stack.enter_async_context(
                    stdio_client(server_params, errlog=stderr_log)
                )
                blender_session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
                await blender_session.initialize()
                tool_list = await blender_session.list_tools()
                for tool in tool_list.tools:
                    mcp_tools[tool.name] = (blender_session, False)
                    tool_definitions.append({
                        "name": tool.name,
                        "description": tool.description or "",
                        "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
                    })

            # 2. 初始化 Kling MCP (若启用且已完成授权)
            from bot.kling.kling_mcp_auth import get_kling_oauth_manager
            if conf().get("kling_mcp_enabled") and get_kling_oauth_manager().has_valid_tokens():
                try:
                    import httpx2
                    from mcp.client.streamable_http import streamable_http_client
                    from bot.kling.kling_mcp_auth import get_kling_oauth_manager

                    kling_oauth = get_kling_oauth_manager()
                    kling_provider = kling_oauth.get_auth_provider()
                    server_url = conf().get("kling_mcp_server_url") or "https://klingai.com/mcp"

                    timeout = httpx2.Timeout(connect=20.0, read=360.0, write=60.0, pool=360.0)
                    kling_http = await stack.enter_async_context(
                        httpx2.AsyncClient(auth=kling_provider, timeout=timeout)
                    )
                    k_read, k_write = await stack.enter_async_context(
                        streamable_http_client(server_url, http_client=kling_http)
                    )
                    kling_session = await stack.enter_async_context(ClientSession(k_read, k_write))
                    await kling_session.initialize()
                    kling_tool_list = await kling_session.list_tools()
                    for tool in kling_tool_list.tools:
                        mcp_tools[tool.name] = (kling_session, True)
                        tool_definitions.append({
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
                        })
                    logger.info(f"[{self.Model_ID}] Kling MCP 已连接并载入工具: {[t.name for t in kling_tool_list.tools]}")
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

                        target_session, _ = mcp_tools["query_tasks"]
                        poll_args = _kling_last_query_args or {"generationId": _kling_pending_generation_id}
                        max_code_polls = 24  # 最多轮询 24 次 × 15 秒 = 6 分钟
                        for poll_i in range(max_code_polls):
                            await asyncio.sleep(15)
                            logger.info(
                                f"[{self.Model_ID}] 代码层强制轮询 query_tasks "
                                f"({poll_i + 1}/{max_code_polls}), args={poll_args}"
                            )
                            try:
                                poll_result = await target_session.call_tool("query_tasks", poll_args)
                                poll_text = self._serialize_mcp_result(poll_result)
                                logger.debug(f"[{self.Model_ID}] query_tasks 返回: {poll_text[:500]}")

                                from bot.kling.kling_mcp_client import KlingMCPClient
                                video_url, cover_url = KlingMCPClient.extract_video_info(poll_text)
                                if video_url:
                                    captured_video = {
                                        "video_url": video_url,
                                        "cover_url": cover_url,
                                        "duration": _kling_generation_duration,
                                    }
                                    logger.info(f"[{self.Model_ID}] 代码层轮询捕获到可灵视频: {video_url}")
                                    break

                                # 检查是否明确失败
                                poll_lower = poll_text.lower()
                                if any(s in poll_lower for s in ['"failed"', '"error"', '"cancelled"', '"timeout"']):
                                    logger.warning(f"[{self.Model_ID}] 可灵任务失败: {poll_text[:300]}")
                                    break
                            except Exception as poll_e:
                                logger.warning(f"[{self.Model_ID}] 代码层轮询异常: {poll_e}")
                                break

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

                    # 对可灵 query_tasks 轮询请求，加入等待间隔（视频生成需要 1-3+ 分钟）
                    if is_kling and tool_name == "query_tasks":
                        poll_interval = 15  # 秒
                        logger.info(f"[{self.Model_ID}] 可灵任务轮询中，等待 {poll_interval} 秒后查询...")
                        await asyncio.sleep(poll_interval)

                    result = await target_session.call_tool(tool_name, tool_use.get("input", {}))
                    serialized_res = self._serialize_mcp_result(result)

                    if is_kling:
                        from bot.kling.kling_mcp_client import KlingMCPClient
                        video_url, cover_url = KlingMCPClient.extract_video_info(serialized_res)
                        if video_url:
                            captured_video = {
                                "video_url": video_url,
                                "cover_url": cover_url,
                                "duration": tool_use.get("input", {}).get("duration", 5),
                            }
                            logger.info(f"[{self.Model_ID}] 捕获到可灵生成视频: {video_url}")

                        # 追踪 text_to_video 等生成类工具返回的 generationId
                        if tool_name in ("text_to_video", "image_to_video", "text_to_image") and not video_url:
                            import re as _re
                            gen_match = _re.search(r'generationId["\s:]+["\']?([A-Za-z0-9_-]{10,})', serialized_res)
                            if gen_match:
                                _kling_pending_generation_id = gen_match.group(1)
                                _kling_generation_duration = tool_use.get("input", {}).get("duration", 5)
                                logger.info(f"[{self.Model_ID}] 追踪到可灵异步任务 generationId={_kling_pending_generation_id}")

                        # 记录 query_tasks 的参数，以便代码层强制轮询时复用
                        if tool_name == "query_tasks":
                            _kling_last_query_args = tool_use.get("input", {})
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
    def _strip_kling_media_links(reply_content):
        """移除可灵结果文本中的原始视频/封面链接，视频仍由通道单独下发。"""
        if not isinstance(reply_content, str):
            return reply_content

        # 可灵通常以“视频地址（无水印版）”或“封面图”单独成行，链接紧随其后。
        # 逐行处理可避免超长签名 URL 或括号格式影响后续正常文案。
        cleaned_lines = []
        skip_next_link = False
        for line in reply_content.splitlines():
            if re.match(
                r"^\s*(?:视频地址\s*[（(].*[)）]|封面图)\s*[:：]\s*"
                r"(?:https?://\S+|\[[^\]]*\]\(https?://[^)]+\))\s*$",
                line,
            ):
                continue
            if re.match(r"^\s*(?:视频地址\s*[（(].*[)）]|封面图)\s*[:：]\s*$", line):
                skip_next_link = True
                continue
            if skip_next_link:
                if not line.strip():
                    continue
                if re.match(r"^\s*(?:https?://|\[[^\]]*\]\(https?://)", line):
                    skip_next_link = False
                    continue
                # 兼容标签和链接同行的非标准输出：仅跳过标签，保留该行正文。
                skip_next_link = False
            cleaned_lines.append(line)

        cleaned_content = "\n".join(cleaned_lines)
        return re.sub(r"\n{3,}", "\n\n", cleaned_content).strip()

    @staticmethod
    def _serialize_anthropic_blocks(blocks):
        return [
            block.model_dump(exclude_none=True) if hasattr(block, "model_dump") else block
            for block in blocks
        ]

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
            for cached_file in image_cache.get("files", []):
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
                    except Exception as e:
                        logger.error(f"[{self.Model_ID}] failed to encode image: {e}")
                else:
                    logger.warning(f"[{self.Model_ID}] unsupported cached image type: {data_type}")

            memory.USER_IMAGE_CACHE.pop(session_id)

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
