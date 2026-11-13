"""
Unreal Engine MCP 客户端模块

连接本机 Unreal 编辑器内置的 MCP 服务（Streamable HTTP，无需 OAuth）。
该服务只暴露 4 个“元工具”：list_toolsets / describe_toolsets / describe_tools / call_tool，
真正的编辑器功能都在各个工具集里，由模型通过 call_tool 间接调用。

和 Blender / Kling 客户端保持相同的形状（connect() -> connection，
connection.tool_definitions() / call_tool()，以及静态的 serialize_result()），
方便后续统一抽取。
"""
import asyncio
import contextlib
import json
import logging
import socket
import threading
import time
from urllib.parse import urlparse

import httpx2
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

try:  # 新版 SDK 的异常类名
    from mcp.shared.exceptions import MCPError as McpError
except ImportError:  # 旧版 SDK
    from mcp.shared.exceptions import McpError

from common.log import logger
from config import conf

DEFAULT_URL = "http://127.0.0.1:8000/mcp"

# 暴露给模型的工具名统一加前缀，避免与其他 MCP（如 Blender）的通用名字冲突
TOOL_PREFIX = "unreal_"

_BUSY_MARKER = "slow task"

# Unreal 编辑器是单实例、有状态的资源：同一时刻只允许一个会话操作它
_unreal_lock = threading.Lock()


class _IgnoreSessionClose202(logging.Filter):
    """Unreal 对会话关闭的 DELETE 返回 202，SDK 只认 200/204/405，会打一条无害的警告。"""

    def filter(self, record):
        return "Session termination failed: 202" not in record.getMessage()


logging.getLogger("mcp.client.streamable_http").addFilter(_IgnoreSessionClose202())


class _SyntheticResult:
    """不经过 Unreal、由本地生成的工具结果（权限拒绝、MCP 错误等）。"""

    def __init__(self, text: str, is_error: bool = True):
        self.content = [{"type": "text", "text": text}]
        self.is_error = is_error


class UnrealMCPConnection:
    """一次存活的 MCP 会话及其工具列表。"""

    def __init__(self, session, tools):
        self.session = session
        self.tools = tools

    def tool_definitions(self) -> list[dict]:
        """返回统一的工具定义（名字带 unreal_ 前缀）。"""
        return [
            {
                "name": TOOL_PREFIX + tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
            }
            for tool in self.tools
        ]

    async def call_tool(self, name: str, arguments: dict):
        """name 可以带或不带 unreal_ 前缀。"""
        raw_name = name[len(TOOL_PREFIX):] if name.startswith(TOOL_PREFIX) else name
        arguments = arguments or {}

        retries = int(conf().get("unreal_mcp_busy_retries") or 3)
        logger.info(f"[UnrealMCP] 发起工具调用: {raw_name}, args={arguments}")
        for attempt in range(retries + 1):
            try:
                result = await self.session.call_tool(raw_name, arguments)
                text = UnrealMCPClient.result_text(result)
            except McpError as e:
                result = None
                text = str(e)

            if _BUSY_MARKER in text.lower():
                if attempt < retries:
                    logger.info(f"[UnrealMCP] 编辑器正忙，3 秒后重试 ({attempt + 1}/{retries})")
                    await asyncio.sleep(3)
                    continue
                return _SyntheticResult("Unreal 编辑器持续繁忙（可能正在编译着色器），请稍后再试。")

            if result is None:
                return _SyntheticResult(f"Unreal MCP 错误：{text}")
            logger.info(f"[UnrealMCP] 工具调用完成: {raw_name}")
            return result

        return _SyntheticResult("Unreal 编辑器持续繁忙（可能正在编译着色器），请稍后再试。")


class UnrealMCPClient:
    """本机 Unreal 编辑器 MCP 服务的 Streamable HTTP 客户端。"""

    _ready_cache: dict = {}

    def __init__(self, url: str = None):
        self.url = url or conf().get("unreal_mcp_url") or DEFAULT_URL

    # ---------- 就绪探测 ----------
    def is_ready(self) -> bool:
        """轻量 TCP 探测（结果缓存 5 秒），编辑器没开就直接跳过 Unreal 工具。"""
        now = time.monotonic()
        cached = self._ready_cache.get(self.url)
        if cached and now - cached[0] < 5:
            return cached[1]

        parsed = urlparse(self.url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            with socket.create_connection((host, port), timeout=0.5):
                ok = True
        except OSError:
            ok = False
        self._ready_cache[self.url] = (now, ok)
        return ok

    # ---------- 权限 ----------
    @staticmethod
    def is_session_allowed(session_id) -> bool:
        """unreal_mcp_allowed_users 为空表示不限制；否则只有列表内的会话可用。"""
        users = conf().get("unreal_mcp_allowed_users") or []
        return not users or session_id in users

    # ---------- 连接 ----------
    @contextlib.asynccontextmanager
    async def connect(self):
        """
        异步上下文管理器：获取全局锁 -> 建立会话 -> 返回 UnrealMCPConnection。
        每条消息新建会话，Unreal 重启后不会遇到失效的旧会话。
        """
        lock_timeout = float(conf().get("unreal_mcp_lock_timeout") or 120)
        acquired = await asyncio.to_thread(_unreal_lock.acquire, True, lock_timeout)
        if not acquired:
            raise RuntimeError("Unreal 编辑器正被其他会话占用，请稍后再试")
        try:
            timeout = httpx2.Timeout(connect=5.0, read=300.0, write=30.0, pool=300.0)
            async with httpx2.AsyncClient(timeout=timeout) as http_client:
                async with streamable_http_client(
                    self.url, http_client=http_client
                ) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tool_list = await session.list_tools()
                        logger.info(
                            f"[UnrealMCP] 成功连接并获取工具列表: {[t.name for t in tool_list.tools]}"
                        )
                        yield UnrealMCPConnection(session, tool_list.tools)
        finally:
            _unreal_lock.release()

    # ---------- 结果处理 ----------
    @staticmethod
    def result_text(result) -> str:
        parts = []
        for block in getattr(result, "content", []) or []:
            text = block.get("text") if isinstance(block, dict) else getattr(block, "text", None)
            if text:
                parts.append(text)
        return "\n".join(parts)

    @staticmethod
    def serialize_result(result) -> str:
        """序列化 MCP 结果：图片块替换为占位符，整体按 unreal_mcp_max_result_chars 截断。"""
        content = []
        for block in getattr(result, "content", []) or []:
            if hasattr(block, "model_dump"):
                block = block.model_dump(exclude_none=True)
            if isinstance(block, dict):
                if block.get("type") == "image":
                    content.append({"type": "image", "note": "图片内容已省略"})
                else:
                    content.append(block)
            else:
                content.append({"type": "text", "text": str(block)})
        payload = {"content": content}
        structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
        if structured is not None:
            payload["structured_content"] = structured
        text = json.dumps(payload, ensure_ascii=False, default=str)

        # list_toolsets 的完整输出约 3 万字符，默认阈值要能装下
        limit = int(conf().get("unreal_mcp_max_result_chars") or 40000)
        if len(text) > limit:
            text = text[:limit] + f"...[结果过长已截断，原长 {len(text)} 字符；请缩小查询范围]"
        return text

    # ---------- 提示词 ----------
    @classmethod
    def instructions(cls) -> str:
        return (
            "\n\n[Unreal Engine MCP 工具调用规范]\n"
            "你已连接用户本机运行的 Unreal 编辑器，可以通过 unreal_* 工具查看并修改当前项目。\n"
            "1. 仅当用户明确要求查看或操作 Unreal 项目（关卡、Actor、资产、材质等）时才调用；"
            "闲聊和概念讨论不要调用。\n"
            "2. 这些是“元工具”：unreal_list_toolsets 列出工具集，unreal_describe_toolsets / "
            "unreal_describe_tools 查看具体工具及其参数，unreal_call_tool 才真正执行。"
            "每个具体工具的参数名各不相同，调用 unreal_call_tool 前必须先用 unreal_describe_tools "
            "查看参数，不要猜。\n"
            "3. 调用 unreal_call_tool 时必须同时给出 toolset_name 和 tool_name，"
            "工具集名使用 unreal_list_toolsets 返回的完整名字。\n"
            "4. 先读后写：先查询现状再修改；只做用户要求的事，不要擅自删除、移动或覆盖已有内容。\n"
            "5. 不要主动保存资产或关卡，除非用户明确要求。\n"
            "6. 如果工具提示编辑器繁忙，系统会自动重试；仍失败时如实告诉用户。\n"
            "7. 完成后用简短文字汇报：做了什么、结果如何（例如新建了哪些资产、Actor 的路径）。"
        )
