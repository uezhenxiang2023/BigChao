"""
Script Breakdown MCP 客户端模块
"""
import contextlib
import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional
from common.log import logger
from config import conf


class ScriptBreakdownMCPConnection:
    """封装与 Script_Breakdown MCP Server 的会话及工具定义"""

    def __init__(self, session, tools):
        self.session = session
        self.tools = tools

    def tool_definitions(self) -> List[Dict[str, Any]]:
        """返回统一的标准工具定义列表，供各大模型格式化"""
        from bot.gemini.gemini_mcp import dereference_schema
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": dereference_schema(getattr(tool, "input_schema", getattr(tool, "inputSchema", {}))),
            }
            for tool in self.tools
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]):
        logger.info(f"[ScriptBreakdownMCP] 发起工具调用: {name}, args_keys={list((arguments or {}).keys())}")
        result = await self.session.call_tool(name, arguments or {})
        logger.info(f"[ScriptBreakdownMCP] 工具调用完成: {name}")
        return result


class ScriptBreakdownMCPClient:
    """
    Script_Breakdown MCP 本地 stdio 客户端封装。
    通过子进程启动 FastMCP 服务 (如 uv run script-breakdown-mcp)。
    """

    DEFAULT_COMMAND = "uv"
    DEFAULT_DIR = "/Users/fu/work/MCP/script_breakdown_mcp"

    def __init__(self, command: str = None, args: list = None, cwd: str = None, mcp_dir: str = None):
        self.mcp_dir = mcp_dir or conf().get("script_breakdown_mcp_dir") or self.DEFAULT_DIR
        self.command = command or conf().get("script_breakdown_mcp_command") or self.DEFAULT_COMMAND

        # 优先使用显式配置的 args，若未配置则根据 mcp_dir 自动组装
        configured_args = args or conf().get("script_breakdown_mcp_args")
        if not configured_args:
            configured_args = ["run", "--directory", self.mcp_dir, "script-breakdown-mcp"]
        self.configured_args = configured_args

        # 工作目录默认为 mcp_dir
        self.cwd = cwd or conf().get("script_breakdown_mcp_cwd") or self.mcp_dir
        if not self.cwd and "--directory" in (self.configured_args or []):
            try:
                idx = self.configured_args.index("--directory")
                if idx + 1 < len(self.configured_args):
                    self.cwd = self.configured_args[idx + 1]
            except Exception:
                pass

    def is_ready(self) -> bool:
        """检查启动命令及目标 MCP 目录是否就绪"""
        cmd = self.command
        # 如果 command 是绝对或相对路径，校验是否存在
        if os.path.sep in cmd:
            if not os.path.exists(cmd):
                return False
        else:
            # 命令名（如 uv），检查 PATH 中是否存在
            if not shutil.which(cmd):
                return False

        # 检查 MCP 服务目录是否存在
        target_dir = self.mcp_dir or self.cwd
        if target_dir and not os.path.exists(target_dir):
            logger.warning(f"[ScriptBreakdownMCP] MCP 服务目录不存在: {target_dir}")
            return False

        # 如果 args 里配置了 --directory，检查该目录是否存在
        args = self.configured_args or []
        if "--directory" in args:
            try:
                idx = args.index("--directory")
                if idx + 1 < len(args):
                    dir_path = args[idx + 1]
                    if not os.path.exists(dir_path):
                        logger.warning(f"[ScriptBreakdownMCP] 配置的目录不存在: {dir_path}")
                        return False
            except Exception:
                pass
        return True

    @contextlib.asynccontextmanager
    async def connect(self):
        """
        异步上下文管理器：启动 stdio 子进程，建立 MCP ClientSession 会话，并返回 ScriptBreakdownMCPConnection
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not self.command:
            raise ValueError("未配置 script_breakdown_mcp_command")

        # 准备子进程环境变量
        server_env = os.environ.copy()
        output_dir = conf().get("script_breakdown_output_dir")
        if output_dir:
            output_dir_expanded = os.path.abspath(os.path.expanduser(output_dir))
            os.makedirs(output_dir_expanded, exist_ok=True)
            server_env["SCRIPT_BREAKDOWN_OUTPUT_DIR"] = output_dir_expanded

        server_params = StdioServerParameters(
            command=self.command,
            args=self.configured_args,
            cwd=self.cwd,
            env=server_env,
        )

        stderr_log_path = os.path.join(os.getcwd(), "logs", "script_breakdown_mcp_stderr.log")
        os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

        async with contextlib.AsyncExitStack() as stack:
            stderr_log = stack.enter_context(open(stderr_log_path, "a", encoding="utf-8"))
            stderr_log.write("\n--- Script_Breakdown MCP session started ---\n")
            stderr_log.flush()

            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params, errlog=stderr_log)
            )
            session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await session.initialize()
            tool_list = await session.list_tools()
            logger.info(f"[ScriptBreakdownMCP] 成功连接并获取工具列表: {[t.name for t in tool_list.tools]}")
            connection = ScriptBreakdownMCPConnection(session, tool_list.tools)
            yield connection

    @staticmethod
    def serialize_result(result) -> str:
        """统一序列化 MCP 工具返回结果"""
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
