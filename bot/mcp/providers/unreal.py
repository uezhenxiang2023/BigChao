# encoding:utf-8
from typing import Any, Dict, Optional, Tuple

from bot.mcp.provider import MCPProvider, safe_conf_get
from bot.unreal.unreal_mcp_client import UnrealMCPClient
from common.log import logger
from config import conf


class UnrealProvider(MCPProvider):
    name = "unreal"

    def enabled(self) -> bool:
        return bool(safe_conf_get("unreal_mcp_enabled", False))

    async def ready(self, session_id: Optional[str] = None) -> bool:
        if not UnrealMCPClient.is_session_allowed(session_id):
            logger.info(f"[UnrealMCP] 会话 {session_id} 不在 allowed_users 列表中，跳过 Unreal 工具")
            return False
        if not UnrealMCPClient().is_ready():
            logger.info("[UnrealMCP] 本机 Unreal 编辑器 MCP 端口不可达，跳过 Unreal 工具")
            return False
        return True

    def connect(self):
        client = UnrealMCPClient()
        return client.connect()

    def instructions(self) -> str:
        return UnrealMCPClient.instructions()

    async def on_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        is_error = bool(getattr(raw_result, "is_error", getattr(raw_result, "isError", False)))
        serialized = UnrealMCPClient.serialize_result(raw_result)
        return serialized, is_error

    @property
    def max_tool_rounds(self) -> int:
        return int(safe_conf_get("unreal_mcp_max_tool_rounds") or 50)

    @property
    def lock_timeout(self) -> Optional[float]:
        return float(safe_conf_get("unreal_mcp_lock_timeout") or 120.0)

    def is_cacheable_tool(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        # 元工具查询（列出工具集、描述工具集和工具参数等）结果稳定且体量较大，缓存能显著提速并节省 Token
        lower = tool_name.lower()
        if lower in ("unreal_list_toolsets", "unreal_describe_toolsets", "unreal_describe_tools", "list_toolsets", "describe_toolsets", "describe_tools"):
            return True
        return False
