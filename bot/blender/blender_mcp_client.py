"""
Blender MCP 客户端模块
"""
import contextlib
import json
import os
from pathlib import Path
from common.log import logger
from config import conf


class BlenderMCPConnection:
    """A live MCP session and the tools advertised by the Blender stdio server."""

    def __init__(self, session, tools):
        self.session = session
        self.tools = tools

    def tool_definitions(self) -> list[dict]:
        """返回统一的标准工具定义列表，供各大模型格式化"""
        return [
            {
                "name": tool.name,
                "description": tool.description or "",
                "input_schema": getattr(tool, "input_schema", getattr(tool, "inputSchema", {})),
            }
            for tool in self.tools
        ]

    async def call_tool(self, name: str, arguments: dict):
        logger.info(f"[BlenderMCP] 发起工具调用: {name}, args={arguments}")
        result = await self.session.call_tool(name, arguments)
        logger.info(f"[BlenderMCP] 工具调用完成: {name}")
        return result


class BlenderMCPClient:
    """
    Blender MCP 本地 stdio 客户端封装
    通过子进程 stdio 启动并与本地 Blender MCP Server (如 blmcp) 通信
    """

    def __init__(self, command: str = None, args: list = None, cwd: str = None):
        self.command = command or conf().get("blender_mcp_command")
        self.configured_args = args or conf().get("blender_mcp_args") or ["-m", "blmcp"]
        self.cwd = cwd or conf().get("blender_mcp_cwd") or None

    @contextlib.asynccontextmanager
    async def connect(self):
        """
        异步上下文管理器：启动 stdio 子进程，建立 MCP ClientSession 会话，并返回 BlenderMCPConnection
        """
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        if not self.command:
            raise ValueError("未配置 blender_mcp_command")

        mcp_cwd = self.cwd
        configured_args = self.configured_args

        # Windows 下若为 -m blmcp 模式，生成 entrypoint 脚本避免环境路径丢失
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
                logger.warning(f"[BlenderMCP] 写入 MCP entrypoint 失败，回退到默认参数: {e}")
                server_args = configured_args
        else:
            server_args = configured_args

        if mcp_cwd:
            _venv_site = str(Path(self.command).parent.parent / "Lib" / "site-packages")
            server_env = {
                "PYTHONPATH": _venv_site + os.pathsep + mcp_cwd,
                "PYDEVD_DISABLE_FILE_VALIDATION": "1",
                "PYDEVD_DONT_TRACE_CHILD_PROCESSES": "1",
            }
        else:
            server_env = None

        server_params = StdioServerParameters(
            command=self.command,
            args=server_args,
            cwd=mcp_cwd,
            env=server_env,
        )

        stderr_log_path = os.path.join(os.getcwd(), "logs", "blender_mcp_stderr.log")
        os.makedirs(os.path.dirname(stderr_log_path), exist_ok=True)

        async with contextlib.AsyncExitStack() as stack:
            stderr_log = stack.enter_context(open(stderr_log_path, "a", encoding="utf-8"))
            stderr_log.write("\n--- Blender MCP session started ---\n")
            stderr_log.flush()

            read_stream, write_stream = await stack.enter_async_context(
                stdio_client(server_params, errlog=stderr_log)
            )
            blender_session = await stack.enter_async_context(
                ClientSession(read_stream, write_stream)
            )
            await blender_session.initialize()
            tool_list = await blender_session.list_tools()
            logger.info(f"[BlenderMCP] 成功连接并获取工具列表: {[t.name for t in tool_list.tools]}")
            connection = BlenderMCPConnection(blender_session, tool_list.tools)
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
