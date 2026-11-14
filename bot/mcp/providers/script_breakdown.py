# encoding:utf-8
import json
import os
from typing import Any, Dict, Optional, Tuple

from bot.mcp.provider import MCPProvider, safe_conf_get
from bot.script_breakdown.script_breakdown_mcp_client import ScriptBreakdownMCPClient
from common.log import logger
from config import conf


class ScriptBreakdownProvider(MCPProvider):
    name = "script_breakdown"

    def enabled(self) -> bool:
        if bool(safe_conf_get("script_breakdown_mcp_enabled", False)):
            return True
        from common.tool_button import tool_state
        return any(
            isinstance(s, dict) and s.get("printing")
            for s in getattr(tool_state, "_user_states", {}).values()
        )

    async def ready(self, session_id: Optional[str] = None) -> bool:
        client = ScriptBreakdownMCPClient()
        if not client.is_ready():
            logger.info("[ScriptBreakdownMCP] 本地 FastMCP 运行环境不可达或配置目录不存在，跳过工具")
            return False
        return True

    def connect(self):
        client = ScriptBreakdownMCPClient()
        return client.connect()

    def instructions(self) -> str:
        return (
            "\n\n[剧本拆解与标准化排版 MCP 工具调用规范]\n"
            "当你需要对剧本进行工业级排版并生成文档时，请调用 `screenplay_formatter` 工具。\n"
            "调用前请先通读剧本全文，把每一个段落严格归类为 scene_setting / action / character / dialogue。\n"
            "其中 scene_setting 必须包含 scene_heading（场号、环境、场景名称、时间）。\n"
            "工具会按照好莱坞工业标准生成 A4 尺寸的 .docx 文件（含封面与右上角页码），并返回本地文件路径。\n"
            "若剧本篇幅较长，请使用 append / final 分批传入。\n"
        )

    async def on_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        is_error = bool(getattr(raw_result, "is_error", getattr(raw_result, "isError", False)))
        serialized = ScriptBreakdownMCPClient.serialize_result(raw_result)

        # 尝试捕获生成的文件路径，便于上层 Bot 直接构造文件回复
        try:
            target_json = None
            if isinstance(raw_result, dict):
                target_json = raw_result
            else:
                for block in getattr(raw_result, "content", []) or []:
                    text = getattr(block, "text", None)
                    if isinstance(block, dict):
                        text = block.get("text")
                    if text and isinstance(text, str) and "file_path" in text:
                        target_json = json.loads(text)
                        break

            if target_json and isinstance(target_json, dict) and "file_path" in target_json:
                file_path = target_json["file_path"]
                state["generated_file_path"] = file_path
                state["generated_files"] = [file_path]
                # 兼容现有渠道对 function_response 的结构预期
                state["function_response"] = {
                    "file_pathes": [file_path],
                    "total_paragraphs": target_json.get("total_paragraphs"),
                    "total_scenes": target_json.get("total_scenes"),
                }
                logger.info(f"[ScriptBreakdownMCP] 成功捕获生成的剧本文件: {file_path}")
        except Exception as e:
            logger.warning(f"[ScriptBreakdownMCP] 解析工具返回值中的文件路径失败: {e}")

        return serialized, is_error

    @property
    def max_tool_rounds(self) -> int:
        return int(safe_conf_get("script_breakdown_mcp_max_tool_rounds") or 15)

    @property
    def lock_timeout(self) -> Optional[float]:
        # 针对同一剧本 draft 暂存写操作加锁保护
        return float(safe_conf_get("script_breakdown_mcp_lock_timeout") or 60.0)

    def is_cacheable_tool(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        # 排版写文件工具不可缓存
        return False
