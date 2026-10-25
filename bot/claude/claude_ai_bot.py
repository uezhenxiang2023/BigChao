import asyncio
import base64
import io
import json
import os
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

            create_kwargs = dict(
                model=self.model,
                max_tokens=1000,
                temperature=0.0,
                system=self.system_prompt,
                messages=claude_message
            )

            if conf().get("blender_mcp_enabled"):
                return self._chat_with_blender_mcp(create_kwargs, session_id)
            
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
        """在一条 Claude 消息内执行完整的 MCP 工具调用循环。"""
        try:
            reply_content, output_tokens = asyncio.run(
                self._run_blender_mcp_chat(create_kwargs)
            )
            self.sessions.session_reply(reply_content, session_id, output_tokens)
            logger.info(
                f"[{self.Model_ID}] Blender MCP reply={reply_content}, "
                f"session_id={session_id}, tokens={output_tokens}"
            )
            return Reply(ReplyType.TEXT, reply_content)
        except Exception as e:
            logger.exception(f"[{self.Model_ID}] Blender MCP request failed: {e}")
            return Reply(ReplyType.ERROR, f"[{self.Model_ID}] Blender MCP 调用失败：{e}")

    async def _run_blender_mcp_chat(self, create_kwargs):
        """连接 stdio MCP Server，供 Claude 在本轮请求中按需调用 Blender 工具。"""
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
        except ImportError as e:
            raise RuntimeError("缺少 MCP 客户端依赖，请执行 pip install -r requirements.txt") from e

        command = conf().get("blender_mcp_command")
        if not command:
            raise ValueError("未配置 blender_mcp_command")

        mcp_cwd = conf().get("blender_mcp_cwd") or None
        configured_args = conf().get("blender_mcp_args") or ["-m", "blmcp"]

        # 为避免 VSCode/外部调试器拦截子进程导致 -m blmcp 模块丢失，
        # 在扩展根目录生成一个自包含 entrypoint 脚本，强制在 sys.path 首位注入 site-packages 与扩展根目录。
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
        max_rounds = int(conf().get("blender_mcp_max_tool_rounds", 8))
        messages = list(create_kwargs["messages"])
        stderr_log_path = os.path.join(os.getcwd(), "logs", "blender_mcp_stderr.log")
        os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

        logger.info(f"[{self.Model_ID}] Blender MCP stderr log: {stderr_log_path}")
        with open(stderr_log_path, "a", encoding="utf-8") as stderr_log:
            stderr_log.write("\n--- Blender MCP session started ---\n")
            stderr_log.flush()
            async with stdio_client(server_params, errlog=stderr_log) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as mcp_session:
                    await mcp_session.initialize()
                    tool_list = await mcp_session.list_tools()
                    mcp_tools = {tool.name: tool for tool in tool_list.tools}
                    if not mcp_tools:
                        raise RuntimeError("Blender MCP Server 未返回任何工具")

                    tool_definitions = [
                        {
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
                        }
                        for tool in mcp_tools.values()
                    ]

                    for _ in range(max_rounds):
                        response = self.client.messages.create(
                            **dict(create_kwargs, messages=messages, tools=tool_definitions)
                        )
                        response_blocks = self._serialize_anthropic_blocks(response.content)
                        tool_uses = [
                            block for block in response_blocks
                            if block.get("type") == "tool_use"
                        ]
                        if not tool_uses:
                            return self._extract_claude_text(response_blocks), response.usage.output_tokens

                        messages.append({"role": "assistant", "content": response_blocks})
                        tool_results = []
                        for tool_use in tool_uses:
                            tool_name = tool_use["name"]
                            if tool_name not in mcp_tools:
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use["id"],
                                    "is_error": True,
                                    "content": f"未知的 Blender MCP 工具：{tool_name}",
                                })
                                continue

                            result = await mcp_session.call_tool(tool_name, tool_use.get("input", {}))
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": tool_use["id"],
                                "is_error": bool(getattr(result, "is_error", getattr(result, "isError", False))),
                                "content": self._serialize_mcp_result(result),
                            })
                        messages.append({"role": "user", "content": tool_results})

        raise RuntimeError(f"Blender MCP 工具调用超过最大轮次（{max_rounds}）")

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
