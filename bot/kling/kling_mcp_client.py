import json
import re
from typing import Any, Dict, List, Optional, Tuple
import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from common.log import logger
from config import conf
from bot.kling.kling_mcp_auth import get_kling_oauth_manager


class KlingMCPClient:
    """
    可灵 AI MCP (Model Context Protocol) 客户端
    使用 Streamable HTTP 协议与官方 https://klingai.com/mcp 通信
    """

    def __init__(self, server_url: str = "https://klingai.com/mcp"):
        self.server_url = server_url
        self.oauth_manager = get_kling_oauth_manager()

    async def get_tool_definitions(self) -> List[Dict[str, Any]]:
        """获取适用于 Claude Messages API 的 Tool 定义列表"""
        provider = self.oauth_manager.get_auth_provider()
        timeout = httpx2.Timeout(connect=15.0, read=120.0, write=60.0, pool=60.0)

        async with httpx2.AsyncClient(auth=provider, timeout=timeout) as http_client:
            async with streamable_http_client(self.server_url, http_client=http_client) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    tool_list = await session.list_tools()
                    definitions = []
                    for tool in tool_list.tools:
                        definitions.append({
                            "name": tool.name,
                            "description": tool.description or "",
                            "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
                        })
                    logger.info(f"[KlingMCP] 成功获取可灵工具列表: {[t['name'] for t in definitions]}")
                    return definitions

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """调用指定的可灵 MCP 工具（如 text_to_video）"""
        provider = self.oauth_manager.get_auth_provider()
        # 视频生成时间通常较长，设置超时 360 秒（6 分钟）
        timeout = httpx2.Timeout(connect=20.0, read=360.0, write=60.0, pool=360.0)

        logger.info(f"[KlingMCP] 发起工具调用: {tool_name}, args={arguments}")
        async with httpx2.AsyncClient(auth=provider, timeout=timeout) as http_client:
            async with streamable_http_client(self.server_url, http_client=http_client) as (read_stream, write_stream):
                async with ClientSession(read_stream, write_stream) as session:
                    await session.initialize()
                    result = await session.call_tool(tool_name, arguments)
                    logger.info(f"[KlingMCP] 工具调用完成: {tool_name}")
                    return result

    @staticmethod
    def parse_tool_result_content(result: Any) -> str:
        """格式化 MCP 工具返回的内容为 JSON/字符串供 Claude 阅读"""
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

    @staticmethod
    def extract_video_info(result_text_or_obj: Any) -> Tuple[Optional[str], Optional[str]]:
        """
        从工具返回值中提取 video_url 与 cover_url
        返回: (video_url, cover_url)
        """
        text = str(result_text_or_obj)

        # 1. 尝试正则匹配视频链接 (.mp4)
        video_url = None
        video_match = re.search(r'https?://[^\s"\'<>]+\.mp4[^\s"\'<>]*', text)
        if video_match:
            video_url = video_match.group(0)

        # 2. 尝试正则匹配封面图片链接 (.jpg / .jpeg / .png)
        cover_url = None
        cover_match = re.search(r'https?://[^\s"\'<>]+\.(?:jpg|jpeg|png)[^\s"\'<>]*', text)
        if cover_match:
            cover_url = cover_match.group(0)

        return video_url, cover_url
