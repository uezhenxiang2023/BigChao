# encoding:utf-8
import os
from typing import Any, Dict, Optional

from bot.blender.blender_mcp_client import BlenderMCPClient
from bot.mcp.provider import MCPProvider, safe_conf_get
from common.log import logger
from config import conf


class BlenderProvider(MCPProvider):
    name = "blender"

    def enabled(self) -> bool:
        return bool(safe_conf_get("blender_mcp_enabled", False))

    async def ready(self, session_id: Optional[str] = None) -> bool:
        cmd = safe_conf_get("blender_mcp_command")
        if not cmd:
            return False
        # 如果配置的是本地文件路径，校验其是否存在
        if os.path.sep in cmd and not os.path.exists(cmd):
            logger.warning(f"[BlenderMCP] 配置的 blender_mcp_command 不存在: {cmd}")
            return False
        return True

    def connect(self):
        client = BlenderMCPClient()
        return client.connect()

    def instructions(self) -> str:
        return (
            "\n\n[Blender MCP 工具调用规范]\n"
            "你可以通过 Blender MCP 工具查看、创建和修改当前 Blender 场景中的对象、材质、骨骼与动画。\n"
            "可以使用 execute_blender_code 运行 Python 脚本来实现 3D 建模、渲染与场景修改自动化。\n"
        )

    @property
    def max_tool_rounds(self) -> int:
        return int(safe_conf_get("blender_mcp_max_tool_rounds") or 8)

    @property
    def lock_timeout(self) -> Optional[float]:
        # Blender 本地 stdio 服务通常为单进程独占
        return float(safe_conf_get("blender_mcp_lock_timeout") or 60.0)
