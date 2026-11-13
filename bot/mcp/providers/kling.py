# encoding:utf-8
import asyncio
from typing import Any, Dict, Optional, Tuple

from bot.kling.kling_mcp_auth import get_kling_oauth_manager
from bot.kling.kling_mcp_client import KlingMCPClient
from bot.mcp.provider import MCPProvider, safe_conf_get, serialize_mcp_result
from common.log import logger
from config import conf


class KlingProvider(MCPProvider):
    name = "kling"

    def enabled(self) -> bool:
        return bool(safe_conf_get("kling_mcp_enabled", False))

    async def ready(self, session_id: Optional[str] = None) -> bool:
        return get_kling_oauth_manager().has_valid_tokens()

    def connect(self):
        client = KlingMCPClient()
        return client.connect()

    def instructions(self) -> str:
        return (
            "\n\n[可灵 MCP 工具调用规范]\n"
            "你已成功连接可灵（Kling）官方 MCP 服务，挂载了 text_to_video（文生视频）、image_to_video（图生视频）、"
            "text_to_image（文生图）、omni_ref_video（多图全能参考）等工具能力。\n"
            "1. 当用户明确要求制作、生成或想要查看视频/图片时，你可以根据对话内容自动润色分镜提示词并直接调用对应的可灵工具完成生成。\n"
            "2. 绝对不要引导或提示用户使用 '//' 或 '/' 等前缀（前缀为旧系统机制，当前已升级为全自然语言 Agent 调用）。\n"
            "3. 如果用户只是咨询视效技术、探讨剧本或闲聊，未明确要求生成，则只提供文字讨论，不触发工具。\n"
            "4. 调用 text_to_video 时，请根据用户意图合理填充 prompt、aspect_ratio、duration 等参数。\n"
            "5. 【异步任务轮询】生成类工具是异步的，调用后返回 generationId（任务ID）。你必须紧接着调用 query_tasks "
            "工具（传入该 generationId）来轮询任务状态，直到任务完成并获取到最终的视频/图片 URL。不要在拿到 generationId 后就停下来。\n"
            "6. 拿到最终视频/图片 URL 后，将 URL 包含在你的回复文本中，以便系统自动提取并下发给用户。\n"
            "7. 用户上传的参考图会在消息中附带“可灵公网图片地址”。图生视频或多图参考时，必须将这些地址直接填入相应 Kling 工具的图片 URL 参数，绝不能要求用户再次提供链接。\n"
            "8. 用户上传或引用的视频会在消息中附带“可灵公网视频地址”。需要基于该视频进行参考或编辑时，必须将此地址直接填入对应 Kling 工具的视频 URL 参数。\n"
        )

    def check(self, tool_name: str, arguments: Dict[str, Any], context: Optional[Dict[str, Any]] = None) -> Optional[str]:
        # omni_ref_video 多图全能参考检查
        if tool_name == "omni_ref_video" and context:
            messages = context.get("messages") or []
            reference_urls = KlingMCPClient.extract_reference_urls(messages)
            if len(reference_urls) > 1:
                missing_urls = KlingMCPClient.missing_reference_urls(arguments, reference_urls)
                if missing_urls:
                    logger.warning(f"[KlingMCP] omni_ref_video 缺少参考图参数: missing={len(missing_urls)}")
                    return (
                        f"本轮收到 {len(reference_urls)} 张用户参考图，但 omni_ref_video 工具参数只包含 "
                        f"{len(reference_urls) - len(missing_urls)} 张。请重新调用 omni_ref_video，"
                        f"并将每一张“可灵公网图片地址”全部原样填入多图参考参数后再提交。"
                    )
        return None

    async def on_call_tool(
        self,
        connection: Any,
        tool_name: str,
        arguments: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Any:
        # 对 query_tasks 轮询请求，等待 15 秒（视频生成需要一定耗时）
        if tool_name == "query_tasks":
            poll_interval = 15
            logger.info(f"[KlingMCP] 可灵任务轮询中，等待 {poll_interval} 秒后查询...")
            await asyncio.sleep(poll_interval)
        return await connection.call_tool(tool_name, arguments)

    async def on_result(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
        raw_result: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[str, bool]:
        is_error = bool(getattr(raw_result, "is_error", getattr(raw_result, "isError", False)))
        serialized = KlingMCPClient.serialize_result(raw_result)

        video_url, cover_url = KlingMCPClient.extract_video_info(serialized)
        duration = arguments.get("duration", state.get("_kling_generation_duration", 5))

        if video_url:
            state["captured_video"] = {
                "video_url": video_url,
                "cover_url": cover_url,
                "duration": duration,
            }
            state["_kling_pending_generation_id"] = None
            logger.info(f"[KlingMCP] 捕获到可灵生成视频: {video_url}")

        if tool_name in KlingMCPClient.GENERATION_TOOLS and not video_url:
            gen_id = KlingMCPClient.extract_generation_id(serialized) or KlingMCPClient.extract_generation_id(arguments)
            if gen_id:
                state["_kling_pending_generation_id"] = gen_id
                state["_kling_generation_duration"] = duration
                logger.info(f"[KlingMCP] 追踪到可灵异步任务 generationId={gen_id}")

        if tool_name == "query_tasks":
            if not video_url and not state.get("_kling_pending_generation_id"):
                gen_id = KlingMCPClient.extract_generation_id(arguments)
                if gen_id:
                    state["_kling_pending_generation_id"] = gen_id
                    logger.info(f"[KlingMCP] 从 query_tasks 入参恢复任务 generationId={gen_id}")
            if video_url:
                state["_kling_pending_generation_id"] = None

        return serialized, is_error

    async def finalize(
        self,
        connection: Any,
        state: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        gen_id = state.get("_kling_pending_generation_id")
        captured = state.get("captured_video")
        if gen_id and not captured and hasattr(connection, "poll_task"):
            logger.info(f"[KlingMCP] 模型已结束工具调用，启动底层代码强制轮询未完成任务: {gen_id}")
            duration = state.get("_kling_generation_duration", 5)
            polled_video = await connection.poll_task(gen_id, duration=duration)
            if polled_video:
                state["captured_video"] = polled_video
                return polled_video
        return captured

    @property
    def max_tool_rounds(self) -> int:
        return int(safe_conf_get("kling_mcp_max_tool_rounds") or 20)

    def is_cacheable_tool(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        # 可灵生成与轮询状态是实时的，不走静态缓存
        return False
