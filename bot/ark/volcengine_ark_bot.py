"""
Bytedance volcengine_ark bot

@author fort
@Date 2025/10/19
"""

import asyncio
import json
import os
import time

from arkruntime import Ark

from bot.ark.ark_error import (
    format_ark_api_error,
    format_ark_connection_error,
    is_ark_connection_error,
    is_ark_status_error,
    is_ark_timeout_error,
)
from common.media_store import build_public_media_url
from config import conf
from bot.bot import Bot
from bot.ark.ark_media import process_image_files, process_video_files
from bot.session_manager import SessionManager
from bot.ark.volcengine_ark_session import VolcengineArkSession
from bridge.context import ContextType, Context
from bridge.reply import Reply, ReplyType
from common import const, memory
from common.log import logger
from common.model_status import model_state

_ark_sessions = SessionManager(VolcengineArkSession, model=const.DOUBAO_SEED_20)

class VolcengineArkBot(Bot):
    _ARK_INLINE_PDF_LIMIT_BYTES = 50 * 1024 * 1024
    _ARK_MAX_FILE_API_PDF_LIMIT_BYTES = 512 * 1024 * 1024
    _ARK_FILE_POLL_TIMEOUT_SECONDS = 300
    _ARK_FILE_POLL_INTERVAL_SECONDS = 2

    def __init__(self):
        super().__init__()
        self.api_key = conf().get("ark_api_key")
        self.system_prompt = conf().get("character_desc") 
        self.use_responses_api = conf().get("ark_use_responses_api", False)
        self.sessions = _ark_sessions

        self.client = Ark(
            api_key=self.api_key,
            base_url="https://ark.cn-beijing.volces.com/api/v3"
        )

    def reply(self, query, context: Context = None) -> Reply:
        try:
            session_id = context["session_id"]
            self.model = model_state.get_basic_state(session_id)
            self.Model_ID = self.model.upper()

            if context.type != ContextType.TEXT:
                return Reply(ReplyType.ERROR, "Only text context is supported")

            logger.info(f"[{self.Model_ID}] query={query}, requester={session_id}")

            # 检查缓存中是否媒体文件
            image_cache = memory.USER_IMAGE_CACHE.get(session_id)
            video_cache = memory.USER_VIDEO_CACHE.get(session_id)
            file_cache = memory.USER_FILE_CACHE.get(session_id)
            quoted_image_cache = memory.USER_QUOTED_IMAGE_CACHE.get(session_id)
            quoted_file_cache = memory.USER_QUOTED_FILE_CACHE.get(session_id)
            logger.info(
                f"[{self.Model_ID}] cache summary before request, "
                f"quoted_files={len((quoted_file_cache or {}).get('files', []))}, "
                f"files={len((file_cache or {}).get('files', []))}, "
                f"quoted_images={len((quoted_image_cache or {}).get('files', []))}, "
                f"images={len((image_cache or {}).get('files', []))}, "
                f"videos={len((video_cache or {}).get('files', []))}"
            )

            media_contents = []
            kling_mcp_enabled = conf().get("kling_mcp_enabled", False)
            if quoted_file_cache:
                quoted_file_contents = self._build_file_contents(quoted_file_cache)
                if quoted_file_contents:
                    media_contents.extend(quoted_file_contents)
                    logger.info(f"[{self.Model_ID}] 从引用回复文档缓存取内容, count={len(quoted_file_contents)}")
                memory.USER_QUOTED_FILE_CACHE.pop(session_id, None)
            if file_cache:
                file_contents = self._build_file_contents(file_cache)
                if file_contents:
                    media_contents.extend(file_contents)
                    logger.info(f"[{self.Model_ID}] 从内存文档缓存取内容, count={len(file_contents)}")
                memory.USER_FILE_CACHE.pop(session_id, None)
            if quoted_image_cache:
                quoted_image_contents = process_image_files(quoted_image_cache)
                media_contents.extend(quoted_image_contents)
                if kling_mcp_enabled:
                    media_contents.extend(
                        self._build_kling_image_reference_contents(
                            quoted_image_cache,
                            "引用参考图",
                        )
                    )
                logger.info(f"[{self.Model_ID}] 从引用回复图片缓存取内容, count={len(quoted_image_contents)}")
                memory.USER_QUOTED_IMAGE_CACHE.pop(session_id, None)
            if image_cache:
                image_contents = process_image_files(image_cache)
                media_contents.extend(image_contents)
                if kling_mcp_enabled:
                    media_contents.extend(
                        self._build_kling_image_reference_contents(
                            image_cache,
                            "参考图",
                        )
                    )
                logger.info(f"[{self.Model_ID}] 从内存参考图取内容, count={len(image_contents)}")
                memory.USER_IMAGE_CACHE.pop(session_id)
            if video_cache:
                video_contents = process_video_files(video_cache)
                media_contents.extend(video_contents)
                logger.info(f"[{self.Model_ID}] 从内存参考视频取内容, count={len(video_contents)}")
                memory.USER_VIDEO_CACHE.pop(session_id)
            if media_contents:
                logger.info(
                    f"[{self.Model_ID}] media injection summary, "
                    f"blocks={self._summarize_user_media_blocks(media_contents)}"
                )
                media_contents.append({"type": "text", "text": query})
                query = media_contents

            session = self.sessions.session_query(query, session_id)
            if self.use_responses_api and self.model not in const.DOUBAO_BOT_LIST:
                pending_video_url = session.pending_video_url
                response_query = self._add_pending_video_input(query, pending_video_url)
                blender_mcp_enabled = conf().get("blender_mcp_enabled", False)
                blender_ready = bool(blender_mcp_enabled and conf().get("blender_mcp_command"))

                kling_ready = False
                if kling_mcp_enabled:
                    from bot.kling.kling_mcp_auth import get_kling_oauth_manager

                    kling_ready = get_kling_oauth_manager().has_valid_tokens()
                    if not kling_ready:
                        logger.warning(
                            f"[{self.Model_ID}] Kling MCP is enabled, but OAuth authorization is unavailable; "
                            "continuing without Kling tools"
                        )

                video_info = None
                if kling_ready or blender_ready:
                    try:
                        response, video_info = asyncio.run(
                            self._create_response_with_mcp(
                                session, response_query, enable_kling=kling_ready, enable_blender=blender_ready
                            )
                        )
                    except Exception as error:
                        if not self._contains_kling_auth_required_error(error):
                            raise
                        logger.warning(
                            f"[{self.Model_ID}] Kling MCP authorization is required; "
                            "continuing this request without Kling tools",
                            exc_info=True,
                        )
                        if blender_ready:
                            response, video_info = asyncio.run(
                                self._create_response_with_mcp(
                                    session, response_query, enable_kling=False, enable_blender=True
                                )
                            )
                        else:
                            response = self._create_response(session, response_query)
                else:
                    response = self._create_response(session, response_query)
                reply_text = self._extract_response_text(response)
                total_tokens = self._extract_response_total_tokens(response)
                session.previous_response_id = getattr(response, "id", None)
                session.remote_history_outdated = False
                if pending_video_url:
                    session.pending_video_url = None
                    logger.info(f"[{self.Model_ID}] sent pending generated video as incremental Responses input")
                logger.info(
                    f"[{self.Model_ID}] Responses API response_id={getattr(response, 'id', None)}, "
                    f"previous_response_id={getattr(response, 'previous_response_id', None)}, requester={session_id}"
                )
            else:
                client_attr = 'bot_chat' if self.model in const.DOUBAO_BOT_LIST else 'chat'
                # 如果调用bot,去掉messages列表中的system prompt
                if client_attr == 'bot_chat' and session.messages[0].get('role') == 'system':
                    session.messages.pop(0)
                completion = getattr(self.client, client_attr).completions.create(
                    model=self.model,
                    messages=session.messages,
                    #thinking={"type": self.thinking}
                )
                reply_text = completion.choices[0].message.content
                total_tokens = completion.usage.total_tokens if client_attr == 'chat' else completion.bot_usage.model_usage[0].total_tokens
            logger.info(f"[{self.Model_ID}] reply={reply_text}, requester={session_id}")
            self.sessions.session_reply(reply_text, session_id, total_tokens)
            if self.use_responses_api and self.model not in const.DOUBAO_BOT_LIST:
                if video_info and video_info.get("video_url"):
                    from bot.kling.kling_mcp_client import KlingMCPClient

                    session.pending_video_url = video_info["video_url"]
                    logger.info(f"[{self.Model_ID}] queued generated video for next Responses request")
                    clean_reply = KlingMCPClient.strip_media_links(reply_text)
                    return Reply(
                        ReplyType.VIDEO_URL,
                        (
                            video_info.get("duration", 5),
                            video_info["video_url"],
                            clean_reply,
                        ),
                    )
            return Reply(ReplyType.TEXT, reply_text)

        except Exception as e:
            logger.error(f"[{self.Model_ID}] fetch reply error, {e}")
            return Reply(ReplyType.ERROR, self._format_ark_error_message(e))

    def _add_pending_video_input(self, current_query, video_url):
        if not video_url:
            return current_query

        if isinstance(current_query, list):
            content = list(current_query)
        else:
            content = [{"type": "text", "text": current_query}]
        content.insert(0, {
            "type": "video_url",
            "video_url": {
                "url": video_url,
                "fps": 1,
            },
        })
        return content

    @staticmethod
    def _contains_kling_auth_required_error(error):
        if "可灵 MCP 尚未授权" in str(error):
            return True
        nested_errors = getattr(error, "exceptions", ())
        return any(
            VolcengineArkBot._contains_kling_auth_required_error(nested_error)
            for nested_error in nested_errors
        )

    def _create_response(self, session, current_query, tools=None, instructions=None):
        request_kwargs = {
            "model": self.model,
            "store": True,
        }
        if instructions:
            request_kwargs["instructions"] = instructions
        if tools:
            request_kwargs["tools"] = tools
            request_kwargs["tool_choice"] = "auto"
        if session.previous_response_id and not session.remote_history_outdated:
            request_kwargs["input"] = self._to_response_input([{"role": "user", "content": current_query}])
            request_kwargs["previous_response_id"] = session.previous_response_id
            try:
                return self.client.responses.create(**request_kwargs)
            except Exception as e:
                if self._is_ark_sdk_error(e):
                    raise
                logger.warning(
                    f"[{self.Model_ID}] Responses API continuation failed, fallback to local history replay: {e}"
                )
                session.previous_response_id = None
                session.remote_history_outdated = True

        replay_messages = list(session.messages)
        if replay_messages and replay_messages[-1].get("role") == "user":
            replay_messages[-1] = {
                "role": "user",
                "content": current_query,
            }
        else:
            replay_messages.append({
                "role": "user",
                "content": current_query,
            })
        request_kwargs["input"] = self._to_response_input(replay_messages)
        request_kwargs.pop("previous_response_id", None)
        return self.client.responses.create(**request_kwargs)

    async def _create_response_with_kling(self, session, current_query):
        return await self._create_response_with_mcp(
            session, current_query, enable_kling=True, enable_blender=False
        )

    async def _create_response_with_mcp(
        self, session, current_query, enable_kling: bool = True, enable_blender: bool = False
    ):
        import contextlib
        from bot.kling.kling_mcp_client import KlingMCPClient
        from bot.blender.blender_mcp_client import BlenderMCPClient

        max_rounds = int(
            conf().get("kling_mcp_max_tool_rounds")
            or conf().get("blender_mcp_max_tool_rounds")
            or 20
        )
        if max_rounds < 1:
            raise ValueError("mcp max_tool_rounds must be at least 1")

        reference_urls = []
        instructions_parts = []
        if enable_kling:
            reference_urls = KlingMCPClient.extract_reference_urls(session.messages)
            instructions_parts.append(
                "Kling MCP tools are available. Call a generation tool only when the user explicitly asks "
                "to create an image or video, not for ordinary discussion. For asynchronous generation, "
                "call query_tasks until the final media URL is available. Use the user's supplied Kling "
                "public reference URLs when the request depends on uploaded images or video. Never call "
                "file_upload; use the provided public reference URLs directly in the image URL parameters."
            )
        if enable_blender:
            instructions_parts.append(
                "Blender MCP tools are available. You can inspect the active Blender scene, "
                "objects, datablocks, materials, and execute Python scripts in Blender using execute_blender_code "
                "to automate 3D modeling, rendering, scene inspection, and modifications."
            )
        instructions = "\n\n".join(instructions_parts)

        pending_generation_id = None
        generation_duration = 5
        captured_video = None

        async with contextlib.AsyncExitStack() as stack:
            # mcp_tools: tool_name -> (connection, is_kling)
            mcp_tools = {}
            all_tool_defs = []

            # 1. 挂载 Blender 工具 (若启用)
            if enable_blender:
                try:
                    blender_client = BlenderMCPClient()
                    blender_conn = await stack.enter_async_context(blender_client.connect())
                    b_tools = blender_conn.tool_definitions()
                    for t in b_tools:
                        mcp_tools[t["name"]] = (blender_conn, False)
                    all_tool_defs.extend(b_tools)
                    logger.info(
                        f"[{self.Model_ID}] Blender MCP 已连接并为 Ark 载入工具: "
                        f"{[t['name'] for t in b_tools]}"
                    )
                except Exception as e:
                    logger.warning(f"[{self.Model_ID}] Ark 连接 Blender MCP 失败: {e}")
                    if not enable_kling:
                        raise

            # 2. 挂载 Kling 工具 (若启用)
            if enable_kling:
                kling_client = KlingMCPClient()
                kling_conn = await stack.enter_async_context(kling_client.connect())
                k_tools = [
                    tool
                    for tool in kling_conn.tool_definitions()
                    if tool["name"] != "file_upload"
                ]
                for t in k_tools:
                    mcp_tools[t["name"]] = (kling_conn, True)
                all_tool_defs.extend(k_tools)
                logger.info(
                    f"[{self.Model_ID}] Kling MCP 已连接并为 Ark 载入工具: "
                    f"{[t['name'] for t in k_tools]}"
                )

            if not mcp_tools:
                raise RuntimeError("未提供任何可用的 MCP 工具")

            tools = [
                {
                    "type": "function",
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"],
                }
                for tool in all_tool_defs
            ]
            tool_names = set(mcp_tools.keys())
            response = self._create_response(
                session,
                current_query,
                tools=tools,
                instructions=instructions,
            )

            for _ in range(max_rounds):
                tool_calls = [
                    item
                    for item in (self._get_attr_or_key(response, "output", []) or [])
                    if self._get_attr_or_key(item, "type") == "function_call"
                ]
                if not tool_calls:
                    if (
                        pending_generation_id
                        and not captured_video
                        and "query_tasks" in tool_names
                    ):
                        kling_conn_item = mcp_tools.get("query_tasks")
                        if kling_conn_item:
                            captured_video = await kling_conn_item[0].poll_task(
                                pending_generation_id,
                                duration=generation_duration,
                            )
                    return response, captured_video

                response_id = self._get_attr_or_key(response, "id")
                if not response_id:
                    raise RuntimeError("Ark Responses API 工具调用响应缺少 response id")

                tool_outputs = []
                for tool_call in tool_calls:
                    tool_name = self._get_attr_or_key(tool_call, "name")
                    call_id = self._get_attr_or_key(tool_call, "call_id")
                    raw_arguments = self._get_attr_or_key(tool_call, "arguments", "{}")
                    if not call_id:
                        raise RuntimeError(f"Ark 工具调用缺少 call_id: {tool_name}")
                    if tool_name not in tool_names:
                        raise RuntimeError(f"Ark 请求了未注册的 MCP 工具：{tool_name}")

                    try:
                        arguments = (
                            json.loads(raw_arguments)
                            if isinstance(raw_arguments, str)
                            else raw_arguments
                        )
                    except json.JSONDecodeError as error:
                        raise ValueError(f"Ark 工具参数不是有效 JSON：{tool_name}") from error
                    if not isinstance(arguments, dict):
                        raise ValueError(f"Ark 工具参数必须是 JSON 对象：{tool_name}")

                    conn, is_kling = mcp_tools[tool_name]

                    if (
                        is_kling
                        and tool_name == "omni_ref_video"
                        and len(reference_urls) > 1
                    ):
                        missing_urls = KlingMCPClient.missing_reference_urls(
                            arguments,
                            reference_urls,
                        )
                        if missing_urls:
                            output = (
                                f"本轮收到 {len(reference_urls)} 张用户参考图，但 omni_ref_video "
                                f"工具参数只包含 {len(reference_urls) - len(missing_urls)} 张。"
                                "请重新调用并将所有“可灵公网图片地址”原样填入多图参考参数。"
                            )
                            tool_outputs.append({
                                "type": "function_call_output",
                                "call_id": call_id,
                                "output": output,
                            })
                            continue

                    if is_kling and tool_name == "query_tasks":
                        await asyncio.sleep(15)

                    result = await conn.call_tool(tool_name, arguments)

                    if is_kling:
                        serialized_result = KlingMCPClient.serialize_result(result)
                        if getattr(result, "is_error", getattr(result, "isError", False)):
                            serialized_result = json.dumps({
                                "is_error": True,
                                "result": serialized_result,
                            }, ensure_ascii=False)
                        video_url, cover_url = KlingMCPClient.extract_video_info(serialized_result)
                        if video_url:
                            captured_video = {
                                "video_url": video_url,
                                "cover_url": cover_url,
                                "duration": arguments.get("duration", generation_duration),
                            }

                        if tool_name in KlingMCPClient.GENERATION_TOOLS and not video_url:
                            pending_generation_id = KlingMCPClient.extract_generation_id(
                                serialized_result
                            )
                            generation_duration = arguments.get("duration", 5)
                        elif tool_name == "query_tasks":
                            pending_generation_id = (
                                None
                                if video_url
                                else pending_generation_id
                                or KlingMCPClient.extract_generation_id(arguments)
                            )
                    else:
                        serialized_result = BlenderMCPClient.serialize_result(result)

                    tool_outputs.append({
                        "type": "function_call_output",
                        "call_id": call_id,
                        "output": serialized_result,
                    })

                response = self.client.responses.create(
                    model=self.model,
                    store=True,
                    previous_response_id=response_id,
                    input=tool_outputs,
                    tools=tools,
                    tool_choice="auto",
                    instructions=instructions,
                )

            remaining_tool_calls = [
                item
                for item in (self._get_attr_or_key(response, "output", []) or [])
                if self._get_attr_or_key(item, "type") == "function_call"
            ]
            if not remaining_tool_calls:
                if pending_generation_id and not captured_video and "query_tasks" in tool_names:
                    kling_conn_item = mcp_tools.get("query_tasks")
                    if kling_conn_item:
                        captured_video = await kling_conn_item[0].poll_task(
                            pending_generation_id,
                            duration=generation_duration,
                        )
                return response, captured_video

        raise RuntimeError(f"MCP 工具调用超过最大轮次（{max_rounds}）")

    def _format_ark_error_message(self, error):
        if is_ark_status_error(error):
            return format_ark_api_error(error, self.model, service_name="火山方舟")
        if is_ark_timeout_error(error):
            return format_ark_connection_error(error, self.model, error_type="timeout")
        if is_ark_connection_error(error):
            return format_ark_connection_error(error, self.model, error_type="connection")
        return f"[{self.Model_ID}] {error}"

    def _is_ark_sdk_error(self, error):
        return is_ark_status_error(error) or is_ark_timeout_error(error) or is_ark_connection_error(error)

    def _to_response_input(self, messages):
        response_input = []
        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")
            if isinstance(content, str):
                response_input.append({
                    "role": role,
                    "content": content,
                })
                continue

            if isinstance(content, list):
                if role == "assistant":
                    response_input.append(self._convert_assistant_multimodal_message(content))
                    continue
                response_input.append({
                    "role": role,
                    "content": self._normalize_content_blocks(content),
                })
                continue

            response_input.append({
                "role": role,
                "content": str(content),
            })
        return response_input

    def _convert_assistant_multimodal_message(self, blocks):
        normalized_blocks = self._normalize_content_blocks(blocks)
        has_media = any(
            isinstance(block, dict) and block.get("type") in {"input_image", "input_video"}
            for block in normalized_blocks
        )
        if has_media:
            return {
                "role": "user",
                "content": normalized_blocks,
            }

        return {
            "role": "assistant",
            "content": self._summarize_assistant_multimodal_content(blocks),
        }

    def _summarize_assistant_multimodal_content(self, blocks):
        text_parts = []
        image_count = 0
        video_count = 0

        for block in blocks:
            if not isinstance(block, dict):
                text_parts.append(str(block))
                continue

            block_type = block.get("type")
            if block_type == "text":
                text = block.get("text", "")
                if text:
                    text_parts.append(text)
            elif block_type == "image_url":
                image_count += 1
            elif block_type == "video_url":
                video_count += 1

        summary_parts = []
        if image_count:
            summary_parts.append(f"[assistant injected {image_count} image]")
        if video_count:
            summary_parts.append(f"[assistant injected {video_count} video]")
        summary_parts.extend(text_parts)
        return "\n".join(summary_parts) if summary_parts else "[assistant multimodal content]"

    def _normalize_content_blocks(self, blocks):
        normalized = []
        for block in blocks:
            if not isinstance(block, dict):
                normalized.append({"type": "input_text", "text": str(block)})
                continue

            block_type = block.get("type")
            if block_type == "text":
                normalized.append({
                    "type": "input_text",
                    "text": block.get("text", ""),
                })
                continue

            if block_type == "image_url":
                image_url = block.get("image_url", {}).get("url")
                if image_url:
                    normalized.append({
                        "type": "input_image",
                        "image_url": image_url,
                    })
                continue

            if block_type == "video_url":
                video_payload = block.get("video_url", {})
                video_url = video_payload.get("url")
                if video_url:
                    normalized.append({
                        "type": "input_video",
                        "video_url": video_url,
                        "fps": video_payload.get("fps", 1),
                    })
                continue

            if block_type == "file":
                file_payload = block.get("file", {})
                file_content = self._normalize_file_block(file_payload)
                if file_content:
                    normalized.append(file_content)
                continue

            normalized.append(block)
        return normalized

    def _build_file_contents(self, file_cache):
        contents = []
        for cached_file in file_cache.get("files", []):
            if not isinstance(cached_file, dict):
                logger.warning(f"[{self.Model_ID}] unsupported cached file type: {type(cached_file).__name__}")
                continue

            mime_type = cached_file.get("mime_type", "")
            file_path = cached_file.get("path", "")
            file_name = os.path.basename(file_path) if file_path else "unknown"
            file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0
            logger.info(
                f"[{self.Model_ID}] preparing file block, "
                f"name={file_name}, mime_type={mime_type}, size={file_size}"
            )
            if mime_type == "application/pdf":
                file_block = self._build_pdf_file_block(cached_file)
                if file_block is not None:
                    contents.append(file_block)
                continue

            if mime_type in ("application/docx", "application/doc", "application/plain"):
                raw_text = cached_file.get("data", "")
                if raw_text:
                    contents.append({
                        "type": "text",
                        "text": f"<document>\n{raw_text}\n</document>",
                    })
                    logger.info(
                        f"[{self.Model_ID}] document text block added, "
                        f"name={file_name}, mime_type={mime_type}, text_length={len(raw_text)}"
                    )
                continue

            logger.warning(f"[{self.Model_ID}] unsupported file mime_type: {mime_type}")
        return contents

    def _build_pdf_file_block(self, cached_file):
        file_path = cached_file.get("path", "")
        raw_data = cached_file.get("data", "")
        file_name = os.path.basename(file_path) if file_path else "upload.pdf"
        file_size = os.path.getsize(file_path) if file_path and os.path.exists(file_path) else 0

        if file_size > self._ARK_MAX_FILE_API_PDF_LIMIT_BYTES:
            raise ValueError("Ark 官方目前只支持 512MB 以内的文档，请压缩或拆分后再试。")

        if file_size >= self._ARK_INLINE_PDF_LIMIT_BYTES:
            file_id = self._upload_pdf_to_ark(file_path)
            logger.info(f"[{self.Model_ID}] PDF exceeds 50MB, use Ark Files API, file_id={file_id}, path={file_path}")
            return {
                "type": "file",
                "file": {
                    "file_id": file_id,
                    "filename": file_name,
                    "mime_type": "application/pdf",
                }
            }

        return {
            "type": "file",
            "file": {
                "filename": file_name,
                "file_data": f"data:application/pdf;base64,{raw_data}",
            }
        }

    def _summarize_user_media_blocks(self, blocks):
        summary = {
            "text": 0,
            "image": 0,
            "video": 0,
            "file": 0,
        }
        for block in blocks:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type")
            if block_type == "text":
                summary["text"] += 1
            elif block_type == "image_url":
                summary["image"] += 1
            elif block_type == "video_url":
                summary["video"] += 1
            elif block_type == "file":
                summary["file"] += 1
        return summary

    def _build_kling_image_reference_contents(self, image_cache, label):
        references = []
        image_paths = image_cache.get("path", [])
        for index, image_path in enumerate(image_paths, start=1):
            public_url = build_public_media_url(image_path) if image_path else None
            if not public_url:
                logger.warning(
                    "[%s] could not build public URL for Kling %s %s: %s",
                    self.Model_ID,
                    label,
                    index,
                    image_path,
                )
                continue

            references.append({
                "type": "text",
                "text": (
                    f"[{label} {index} 的可灵公网图片地址：{public_url}。"
                    "如果调用 Kling 的 image_to_video、omni_ref_video 或其他图片 URL 工具，"
                    "请直接使用此地址，不要调用 file_upload。]"
                ),
            })
            logger.info(
                "[%s] Kling %s %s public_url=%s",
                self.Model_ID,
                label,
                index,
                public_url,
            )
        return references

    def _upload_pdf_to_ark(self, file_path):
        if not file_path or not os.path.exists(file_path):
            raise ValueError("Ark Files API upload failed: missing local PDF path")

        with open(file_path, "rb") as pdf_file:
            uploaded_file = self.client.files.create(
                file=pdf_file,
                purpose="user_data",
            )
        file_id = getattr(uploaded_file, "id", None)
        if not file_id:
            raise ValueError(f"Ark Files API upload failed: missing file id in response {uploaded_file}")
        self._wait_for_ark_file_ready(file_id)
        return file_id

    def _wait_for_ark_file_ready(self, file_id):
        deadline = time.time() + self._ARK_FILE_POLL_TIMEOUT_SECONDS
        last_status = None

        while time.time() < deadline:
            file_obj = self.client.files.retrieve(file_id)
            status = getattr(file_obj, "status", None)
            last_status = status
            logger.info(f"[{self.Model_ID}] Ark file status, file_id={file_id}, status={status}")
            if status in (None, "active"):
                return
            if status == "failed":
                error = getattr(file_obj, "error", None)
                raise ValueError(f"Ark file preprocessing failed, file_id={file_id}, status={status}, error={error}")
            time.sleep(self._ARK_FILE_POLL_INTERVAL_SECONDS)

        raise ValueError(f"Ark file preprocessing timeout, file_id={file_id}, last_status={last_status}")

    def _normalize_file_block(self, file_payload):
        file_id = file_payload.get("file_id")
        file_name = file_payload.get("filename", "upload.pdf")
        mime_type = file_payload.get("mime_type", "application/pdf")
        file_data = file_payload.get("file_data")

        normalized = {
            "type": "input_file",
        }
        if file_id:
            normalized["file_id"] = file_id
        elif file_data:
            normalized["filename"] = file_name
            normalized["file_data"] = file_data
        else:
            return None
        return normalized

    def _extract_response_text(self, response):
        output_text = getattr(response, "output_text", None)
        if output_text:
            return output_text

        collected = []
        for item in getattr(response, "output", []) or []:
            item_type = self._get_attr_or_key(item, "type")
            if item_type != "message":
                continue

            for content in self._get_attr_or_key(item, "content", []) or []:
                text = self._extract_text_from_content(content)
                if text:
                    collected.append(text)

        if collected:
            return "".join(collected)

        try:
            return json.dumps(response, ensure_ascii=False)
        except TypeError:
            return str(response)

    def _extract_text_from_content(self, content):
        content_type = self._get_attr_or_key(content, "type")
        if content_type in {"output_text", "text", "input_text"}:
            text = self._get_attr_or_key(content, "text")
            if isinstance(text, str):
                return text
            if hasattr(text, "value"):
                return getattr(text, "value", "")

        text_obj = self._get_attr_or_key(content, "text")
        if isinstance(text_obj, str):
            return text_obj
        if hasattr(text_obj, "value"):
            return getattr(text_obj, "value", "")
        return ""

    def _extract_response_total_tokens(self, response):
        usage = getattr(response, "usage", None)
        if usage is None:
            return None

        total_tokens = getattr(usage, "total_tokens", None)
        if total_tokens is not None:
            return total_tokens

        input_tokens = getattr(usage, "input_tokens", 0) or 0
        output_tokens = getattr(usage, "output_tokens", 0) or 0
        total = input_tokens + output_tokens
        return total if total else None

    def _get_attr_or_key(self, data, name, default=None):
        if isinstance(data, dict):
            return data.get(name, default)
        return getattr(data, name, default)
