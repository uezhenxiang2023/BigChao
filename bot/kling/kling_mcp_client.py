import asyncio
import json
import re
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional, Tuple

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from bot.kling.kling_mcp_auth import get_kling_oauth_manager
from common.log import logger
from config import conf


class KlingMCPConnection:
    """A live MCP session and the tools advertised by the Kling server."""

    def __init__(self, session, tools):
        self.session = session
        self.tools = tools

    def tool_definitions(self) -> List[Dict[str, Any]]:
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
            }
            for tool in self.tools
        ]

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        logger.info(f"[KlingMCP] 发起工具调用: {tool_name}, args={arguments}")
        result = await self.session.call_tool(tool_name, arguments)
        logger.info(f"[KlingMCP] 工具调用完成: {tool_name}")
        return result

    async def poll_task(
        self,
        generation_id: str,
        duration: int = 5,
        interval_seconds: int = 15,
        max_polls: int = 24,
    ) -> Optional[Dict[str, Any]]:
        """Poll an asynchronous Kling task and return its media when complete."""
        for poll_index in range(max_polls):
            await asyncio.sleep(interval_seconds)
            arguments = {"generationId": generation_id}
            logger.info(
                "[KlingMCP] 轮询任务 %s (%s/%s)",
                generation_id,
                poll_index + 1,
                max_polls,
            )
            try:
                result = await self.call_tool("query_tasks", arguments)
                result_text = KlingMCPClient.serialize_result(result)
                video_url, cover_url = KlingMCPClient.extract_video_info(result_text)
                if video_url:
                    return {
                        "video_url": video_url,
                        "cover_url": cover_url,
                        "duration": duration,
                    }

                if any(
                    status in result_text.lower()
                    for status in ('"failed"', '"error"', '"cancelled"', '"timeout"')
                ):
                    logger.warning("[KlingMCP] 任务失败: %s", result_text[:300])
                    return None
            except Exception:
                logger.exception("[KlingMCP] 轮询任务失败: %s", generation_id)
                return None

        logger.warning("[KlingMCP] 任务轮询超时: %s", generation_id)
        return None


class KlingMCPClient:
    """Shared Streamable HTTP client for the official Kling MCP service."""

    GENERATION_TOOLS = frozenset({
        "text_to_video",
        "image_to_video",
        "omni_ref_video",
        "motion_control",
        "text_to_image",
        "image_to_image",
    })

    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url or conf().get("kling_mcp_server_url") or "https://klingai.com/mcp"
        self.oauth_manager = get_kling_oauth_manager()

    @asynccontextmanager
    async def connect(self):
        provider = self.oauth_manager.get_auth_provider()
        timeout = httpx2.Timeout(connect=20.0, read=360.0, write=60.0, pool=360.0)
        async with httpx2.AsyncClient(auth=provider, timeout=timeout) as http_client:
            async with streamable_http_client(
                self.server_url, http_client=http_client
            ) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_list = await session.list_tools()
                    connection = KlingMCPConnection(session, tool_list.tools)
                    logger.info(
                        "[KlingMCP] 成功获取可灵工具列表: %s",
                        [tool.name for tool in tool_list.tools],
                    )
                    yield connection

    async def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """Return server tool schemas; also used to initiate OAuth authorization."""
        async with self.connect() as connection:
            return connection.tool_definitions()

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """Call a tool in its own short-lived MCP session."""
        async with self.connect() as connection:
            return await connection.call_tool(tool_name, arguments)

    @staticmethod
    def serialize_result(result: Any) -> str:
        content_items = []
        for block in getattr(result, "content", []) or []:
            if hasattr(block, "model_dump"):
                content_items.append(block.model_dump(exclude_none=True))
            elif isinstance(block, dict):
                content_items.append(block)
            else:
                content_items.append({"type": "text", "text": str(block)})

        payload = {"content": content_items}
        structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
        if structured is not None:
            payload["structured_content"] = structured
        return json.dumps(payload, ensure_ascii=False, default=str)

    parse_tool_result_content = serialize_result

    @staticmethod
    def extract_video_info(result_text_or_obj: Any) -> Tuple[Optional[str], Optional[str]]:
        text = str(result_text_or_obj)
        video_match = re.search(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', text)
        cover_match = re.search(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png)[^\s"\'<>]*', text)
        return (
            video_match.group(0) if video_match else None,
            cover_match.group(0) if cover_match else None,
        )

    @staticmethod
    def extract_generation_id(value: Any) -> Optional[str]:
        if isinstance(value, dict):
            for key in (
                "generationId", "generation_id", "generationIds", "generation_ids",
                "taskId", "task_id", "taskIds", "task_ids",
            ):
                candidate = value.get(key)
                if isinstance(candidate, str) and len(candidate) >= 10:
                    return candidate
                if isinstance(candidate, (list, tuple)):
                    candidate_id = KlingMCPClient.extract_generation_id(candidate)
                    if candidate_id:
                        return candidate_id
            for child in value.values():
                candidate_id = KlingMCPClient.extract_generation_id(child)
                if candidate_id:
                    return candidate_id
            return None
        if isinstance(value, (list, tuple)):
            for child in value:
                candidate_id = KlingMCPClient.extract_generation_id(child)
                if candidate_id:
                    return candidate_id
            return None
        if isinstance(value, str):
            if re.fullmatch(r"[A-Za-z0-9_-]{10,}", value):
                return value
            match = re.search(
                r'(?:generationIds?|generation_ids?|taskIds?|task_ids?|任务ID)["\s:：`]*["`]?([A-Za-z0-9_-]{10,})',
                value,
            )
            return match.group(1) if match else None
        return None

    @staticmethod
    def extract_reference_urls(messages: Any) -> List[str]:
        messages = messages if isinstance(messages, list) else [messages]
        for message in reversed(messages):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", [])
            if not isinstance(content, list):
                continue
            urls = []
            for block in content:
                if not isinstance(block, dict) or block.get("type") != "text":
                    continue
                text = block.get("text", "")
                if not isinstance(text, str) or "可灵公网图片地址" not in text:
                    continue
                match = re.search(r"可灵公网图片地址：([^\s。]+)", text)
                if match and match.group(1) not in urls:
                    urls.append(match.group(1))
            if urls:
                return urls
        return []

    @staticmethod
    def extract_http_urls(value: Any) -> List[str]:
        if isinstance(value, dict):
            return [
                url
                for child in value.values()
                for url in KlingMCPClient.extract_http_urls(child)
            ]
        if isinstance(value, (list, tuple)):
            return [
                url
                for child in value
                for url in KlingMCPClient.extract_http_urls(child)
            ]
        if isinstance(value, str):
            return re.findall(r"https?://[^\s\"'\]})]+", value)
        return []

    @staticmethod
    def missing_reference_urls(tool_arguments: Any, reference_urls: List[str]) -> List[str]:
        submitted_urls = KlingMCPClient.extract_http_urls(tool_arguments)
        return [url for url in reference_urls if url not in submitted_urls]

    @staticmethod
    def strip_media_links(reply_content: str) -> str:
        if not isinstance(reply_content, str):
            return reply_content

        media_label_pattern = (
            r"(?:"
            r"视频地址(?:\s*[（(][^)）]*[)）])?"
            r"|成片(?:链接|地址)?(?:\s*[（(][^)）]*[)）])?"
            r"|无水印(?:版|视频|地址|链接)?(?:\s*[（(][^)）]*[)）])?"
            r"|封面(?:图|链接|地址)?(?:\s*[（(][^)）]*[)）])?"
            r")"
        )
        cleaned_lines = []
        skip_next_link = False
        for line in reply_content.splitlines():
            if re.match(
                rf"^\s*{media_label_pattern}\s*[:：]\s*"
                r"(?:https?://\S+|\[[^\]]*\]\(https?://[^)]+\))\s*$",
                line,
            ):
                continue
            if re.match(rf"^\s*{media_label_pattern}\s*[:：]\s*$", line):
                skip_next_link = True
                continue
            if skip_next_link:
                if not line.strip():
                    continue
                if re.match(r"^\s*(?:https?://|\[[^\]]*\]\(https?://)", line):
                    skip_next_link = False
                    continue
                skip_next_link = False
            cleaned_lines.append(line)
        return re.sub(r"\n{3,}", "\n\n", "\n".join(cleaned_lines)).strip()
