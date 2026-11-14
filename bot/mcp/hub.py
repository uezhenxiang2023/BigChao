# encoding:utf-8
import asyncio
import contextlib
import json
import re
import threading
from typing import Any, Dict, List, Optional, Tuple

from bot.mcp.provider import MCPProvider, safe_conf_get, serialize_mcp_result
from bot.mcp.providers.blender import BlenderProvider
from bot.mcp.providers.kling import KlingProvider
from bot.mcp.providers.unreal import UnrealProvider
from bot.mcp.providers.script_breakdown import ScriptBreakdownProvider
from common.log import logger
from config import conf


class MCPSession:
    """
    一个具体的对话轮次中激活的 MCP 聚合会话。
    负责统一路由工具、前置校验、并发加锁、工具调用重试、结果截断与缓存。
    """

    def __init__(
        self,
        hub: "MCPHub",
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        self.hub = hub
        self.session_id = session_id
        self.context = context or {}

        # 工具映射表：tool_name -> (provider, connection)
        self.tool_map: Dict[str, Tuple[MCPProvider, Any]] = {}
        self.tool_definitions: List[Dict[str, Any]] = []
        self.active_providers: List[MCPProvider] = []
        self.state: Dict[str, Any] = {}
        self.instructions: str = ""
        self.max_tool_rounds: int = 20

        # 本轮次内的工具结果缓存：(tool_name, args_json) -> (result_text, is_error)
        self._cache: Dict[Tuple[str, str], Tuple[str, bool]] = {}

    @property
    def has_tools(self) -> bool:
        return bool(self.tool_map)

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> Tuple[str, bool]:
        """
        统一调用工具：
        1. 检查结果缓存（针对只读/元工具等幂等查询）；
        2. 执行 Provider 前置校验（权限、参数缺失等）；
        3. 执行工具调用（支持自动重试机制与正忙重试）；
        4. Provider 后处理（媒体捕获、状态记录）；
        5. 写入缓存并返回 (serialized_result, is_error)。
        """
        if tool_name not in self.tool_map:
            error_msg = f"未注册的 MCP 工具: {tool_name}"
            logger.error(f"[MCPHub] {error_msg}")
            return json.dumps({"is_error": True, "content": error_msg}, ensure_ascii=False), True

        provider, connection = self.tool_map[tool_name]
        arguments = arguments or {}

        # 1. 结果缓存匹配
        args_key = json.dumps(arguments, sort_keys=True, ensure_ascii=False, default=str)
        cache_key = (tool_name, args_key)
        if provider.is_cacheable_tool(tool_name, arguments) and cache_key in self._cache:
            logger.info(f"[MCPHub] 命中工具结果缓存: {tool_name}")
            return self._cache[cache_key]

        # 2. 前置权限/参数检查
        check_error = provider.check(tool_name, arguments, self.context)
        if check_error:
            logger.warning(f"[MCPHub] 工具检查未通过: {tool_name}, reason={check_error}")
            return check_error, True

        # 3. 工具调用与重试控制
        max_retries = int(safe_conf_get("mcp_tool_retries") or safe_conf_get("unreal_mcp_busy_retries") or 3)
        busy_markers = ("slow task", "editor is busy", "temporarily busy")
        raw_result = None
        last_error = None

        logger.info(f"[MCPHub] 发起工具调用: [{provider.name}] {tool_name}, args={arguments}")
        for attempt in range(max_retries + 1):
            try:
                raw_result = await provider.on_call_tool(connection, tool_name, arguments, self.context)
                last_error = None
            except Exception as e:
                last_error = e
                logger.warning(f"[MCPHub] 工具调用异常: {tool_name}, attempt={attempt + 1}/{max_retries + 1}, err={e}")

            # 检查是否有正忙需要重试的标记
            result_str = str(raw_result) if raw_result is not None else ""
            if any(marker in result_str.lower() for marker in busy_markers):
                if attempt < max_retries:
                    retry_wait = 2 * (attempt + 1)
                    logger.info(f"[MCPHub] 资源繁忙，{retry_wait} 秒后重试 ({attempt + 1}/{max_retries})")
                    await asyncio.sleep(retry_wait)
                    continue

            # 如果调用成功，跳出重试
            if last_error is None:
                break
            elif attempt < max_retries:
                await asyncio.sleep(1)

        if last_error is not None:
            err_text = f"MCP 工具执行失败：{last_error}"
            logger.error(f"[MCPHub] {err_text}")
            return json.dumps({"is_error": True, "content": err_text}, ensure_ascii=False), True

        logger.info(f"[MCPHub] 工具调用完成: [{provider.name}] {tool_name}")

        # 4. 后处理与媒体提取
        serialized_result, is_error = await provider.on_result(
            tool_name, arguments, raw_result, self.state, self.context
        )

        # 5. 写入缓存（如果该工具允许缓存且无错误）
        if not is_error and provider.is_cacheable_tool(tool_name, arguments):
            self._cache[cache_key] = (serialized_result, is_error)

        return serialized_result, is_error

    async def finalize(self) -> Optional[Dict[str, Any]]:
        """对话轮次结束时的收尾，如检测并轮询未完成的视频异步生成"""
        captured_video = self.state.get("captured_video")
        for provider in self.active_providers:
            # 找到该 provider 对应的任意 connection
            conn = None
            for p, c in self.tool_map.values():
                if p == provider:
                    conn = c
                    break
            media = await provider.finalize(conn, self.state, self.context)
            if media:
                captured_video = media
        return captured_video


class MCPHub:
    """
    MCP 服务中枢（Hub）：
    统一管理 Provider 注册、并发锁、生命周期与连接。
    """

    def __init__(self, providers: Optional[List[MCPProvider]] = None):
        self.providers: List[MCPProvider] = providers or [
            BlenderProvider(),
            KlingProvider(),
            UnrealProvider(),
            ScriptBreakdownProvider(),
        ]
        # 并发锁字典：provider_name -> threading.Lock
        self._provider_locks: Dict[str, threading.Lock] = {}
        for p in self.providers:
            if p.lock_timeout is not None:
                self._provider_locks[p.name] = threading.Lock()

    def get_provider(self, name: str) -> Optional[MCPProvider]:
        for p in self.providers:
            if p.name == name:
                return p
        return None

    @contextlib.asynccontextmanager
    async def session(
        self,
        session_id: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
    ):
        """
        异步上下文管理器：
        并发加锁 -> 激活并连接所有可用 Provider -> 聚合工具列表与 instructions -> 返回 MCPSession。
        """
        context = context or {}
        mcp_session = MCPSession(self, session_id=session_id, context=context)
        acquired_locks: List[threading.Lock] = []

        async with contextlib.AsyncExitStack() as stack:
            instructions_parts = []
            max_rounds = 20

            for provider in self.providers:
                if not provider.enabled():
                    continue

                try:
                    is_ready = await provider.ready(session_id=session_id)
                except Exception as e:
                    logger.warning(f"[MCPHub] Provider {provider.name} ready() 检查异常: {e}")
                    is_ready = False

                if not is_ready:
                    continue

                # 独占锁处理（如 Unreal、Blender 单实例资源）
                lock_timeout = provider.lock_timeout
                if lock_timeout and provider.name in self._provider_locks:
                    p_lock = self._provider_locks[provider.name]
                    acquired = await asyncio.to_thread(p_lock.acquire, True, lock_timeout)
                    if not acquired:
                        logger.warning(f"[MCPHub] Provider {provider.name} 获取独占锁超时（{lock_timeout}s），跳过该 Provider")
                        continue
                    acquired_locks.append(p_lock)

                # 建立连接
                try:
                    conn = await stack.enter_async_context(provider.connect())
                    tools = conn.tool_definitions() if hasattr(conn, "tool_definitions") else []

                    # 收集工具定义，Unreal 等已有前缀的保留，避免重名冲突
                    for tool in tools:
                        t_name = tool["name"]
                        mcp_session.tool_map[t_name] = (provider, conn)
                        mcp_session.tool_definitions.append(tool)

                    mcp_session.active_providers.append(provider)
                    instr = provider.instructions()
                    if instr:
                        instructions_parts.append(instr.strip())

                    max_rounds = max(max_rounds, provider.max_tool_rounds)
                    logger.info(
                        f"[MCPHub] Provider [{provider.name}] 已挂载，载入工具: {[t['name'] for t in tools]}"
                    )
                except Exception as e:
                    logger.warning(f"[MCPHub] 初始化 Provider {provider.name} 失败，跳过: {e}")

            mcp_session.instructions = "\n\n".join(instructions_parts)
            mcp_session.max_tool_rounds = max_rounds

            try:
                yield mcp_session
            finally:
                for lock in reversed(acquired_locks):
                    try:
                        lock.release()
                    except Exception as e:
                        logger.warning(f"[MCPHub] 释放 Provider 锁失败: {e}")

    @staticmethod
    def strip_media_links(reply_content: str) -> str:
        """剥除模型输出中冗余的视频、封面裸链接，保持最终文本整洁"""
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


# 全局单例 Hub 实例
mcp_hub = MCPHub()
