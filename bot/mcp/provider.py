# encoding:utf-8
from abc import ABC, abstractmethod
import contextlib
import json
from typing import Any, Dict, List, Optional, Tuple

from common.log import logger
from config import conf


def safe_conf_get(key: str, default: Any = None) -> Any:
    try:
        val = conf().get(key, default)
        return default if val is None else val
    except Exception:
        return default


def serialize_mcp_result(result: Any, max_chars: Optional[int] = None) -> str:
    """
    通用 MCP 工具返回结果序列化方法：
    1. 支持 pydantic model_dump、dict 以及原始对象；
    2. 对大型图片块替换为占位符，避免撑爆模型上下文；
    3. 保留 structured_content / structuredContent；
    4. 支持按最大字符数截断保护。
    """
    content_items = []
    blocks = getattr(result, "content", None)
    if blocks is not None:
        for block in blocks or []:
            if hasattr(block, "model_dump"):
                block_dict = block.model_dump(exclude_none=True)
            elif isinstance(block, dict):
                block_dict = block
            else:
                block_dict = {"type": "text", "text": str(block)}

            if isinstance(block_dict, dict) and block_dict.get("type") == "image":
                content_items.append({"type": "image", "note": "图片内容已省略"})
            else:
                content_items.append(block_dict)
    elif isinstance(result, (dict, list)):
        content_items.append(result)
    elif result is not None:
        content_items.append({"type": "text", "text": str(result)})

    payload = {"content": content_items}
    structured = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if structured is not None:
        payload["structured_content"] = structured

    text = json.dumps(payload, ensure_ascii=False, default=str)

    limit = max_chars or int(safe_conf_get("mcp_max_result_chars") or safe_conf_get("unreal_mcp_max_result_chars") or 40000)
    if len(text) > limit:
        text = text[:limit] + f"...[结果过长已截断，原长 {len(text)} 字符；请缩小查询范围]"
    return text


class MCPProvider(ABC):
    """
    MCP Provider 抽象基类。
    每个 Provider 封装具体 MCP 服务（如 Blender、Kling、Unreal）的生命周期、权限检查与结果解析。
    """

    name: str

    @abstractmethod
    def enabled(self) -> bool:
        """检查配置中该 MCP 是否已开启"""
        pass

    @abstractmethod
    async def ready(self, session_id: Optional[str] = None) -> bool:
        """
        轻量就绪/权限探测。
        若返回 False，当前轮次将跳过该 Provider，不影响其他 Provider。
        """
        pass

    @abstractmethod
    @contextlib.asynccontextmanager
    async def connect(self):
        """
        异步上下文管理器，建立连接并 yield connection 对象。
        connection 需至少实现:
            - tool_definitions() -> list[dict]
            - call_tool(name: str, arguments: dict) -> Any
        """
        pass

    @abstractmethod
    def instructions(self) -> str:
        """返回追加到 system prompt 或 instructions 的规范提示词"""
        pass

    def check(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        工具调用前的权限或参数校验。
        若通过返回 None；若拒绝，返回拒绝原因字符串（将作为 tool_result 返回给模型）。
        """
        return None

    async def on_call_tool(
        self,
        connection: Any,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        """
        执行工具调用，可覆盖添加等待或特定处理。
        """
        return await connection.call_tool(tool_name, arguments)

    async def on_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        """
        工具调用完成后的后处理。
        返回 (serialized_result_str, is_error)，并可向 state 中写入捕获的数据（如 captured_video）。
        """
        is_error = bool(getattr(raw_result, "is_error", getattr(raw_result, "isError", False)))
        serialized = serialize_mcp_result(raw_result)
        return serialized, is_error

    async def finalize(
        self,
        connection: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        当对话轮次结束（模型未发起进一步 tool_use）时的兜底收尾（如异步任务轮询）。
        若生成了媒体，返回包含 media 信息的字典；否则返回 None。
        """
        return None

    @property
    def max_tool_rounds(self) -> int:
        """建议的单条消息内最多工具调用轮数"""
        return 20

    @property
    def lock_timeout(self) -> Optional[float]:
        """独占锁等待超时秒数，None 表示不需要独占锁"""
        return None

    def is_cacheable_tool(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        判断工具是否为幂等只读查询工具，默认缓存 get_、list_、describe_ 等无害查询。
        """
        lower = tool_name.lower()
        if any(lower.startswith(prefix) or f"_{prefix}" in lower for prefix in ("get_", "list_", "describe_", "read_", "inspect_")):
            return True
        return False
