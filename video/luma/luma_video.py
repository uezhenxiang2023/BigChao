import base64
import io
import os
import time

from PIL import Image
from luma_agents import APIStatusError, Luma

from bot.bot import Bot
from bot.ark.ark_media import size_calculator, size_calculator_from_data_urls, aspect_ratio_from_size
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from common import const, memory
from common.aspect_ratio import parse_aspect_ratio_from_prompt
from common.log import logger
from common.model_status import model_state
from common.utils import (
    get_chat_session_manager,
    get_image_urls_from_session,
    infer_aspect_ratio_from_video_cache,
    infer_resolution_from_video_cache,
    url_to_base64,
)
from common.video_status import video_state
from config import conf


_LUMA_VIDEO_RATIO_MAP = {
    "9:16": 9 / 16,
    "3:4": 3 / 4,
    "1:1": 1.0,
    "4:3": 4 / 3,
    "16:9": 16 / 9,
    "21:9": 21 / 9,
}
_LUMA_VIDEO_RESOLUTIONS = {"360p", "540p", "720p", "1080p"}
_LUMA_VIDEO_DURATIONS = {"5s", "10s"}
_LUMA_REFERENCE_IMAGE_MAX_BYTES = 50 * 1024 * 1024
_LUMA_REFERENCE_VIDEO_MAX_BYTES = 200 * 1024 * 1024


class LumaVideoClientFacingError(Exception):
    pass


class LumaVideoBot(Bot):
    def __init__(self):
        super().__init__()
        self.client = Luma(
            auth_token=conf().get("luma_agents_api_key"),
            timeout=conf().get("request_timeout", 180),
        )

    def reply(self, query, context: Context = None) -> Reply:
        model = const.LUMA_RAY_32
        try:
            session_id = context["session_id"]
            model = model_state.get_video_state(session_id) or const.LUMA_RAY_32
            video_mode = model_state.get_video_mode(session_id)
            session_manager = get_chat_session_manager(session_id)
            logger.info(f"[{model.upper()}] query={query}, video_mode={video_mode}, requester={session_id}")
            session_manager.session_query(query, session_id)

            params, request_meta = self._build_generation_params(query, session_id, model, video_mode, session_manager)
            logger.info(
                f"[{model.upper()}] 参考素材统计: reference_images={request_meta['reference_image_count']}, "
                f"reference_videos={request_meta['reference_video_count']}, video_mode={video_mode}, "
                f"request_type={params.get('type')}"
            )
            logger.info(
                f"[{model.upper()}] 请求参数: resolution={request_meta['resolution']}, "
                f"ratio={request_meta['aspect_ratio']}, duration={request_meta['duration']}"
            )

            try:
                generation = self.client.generations.create(**params)
            except APIStatusError as e:
                error_message = self._format_sync_api_error(e, model, endpoint="POST")
                logger.warning(f"[{model.upper()}] Luma video create request failed: {error_message}")
                return Reply(ReplyType.ERROR, error_message)

            generation = self._poll_generation(generation, model)
            video_url = self._extract_output_url(generation)
            if not video_url:
                raise ValueError("Luma video response missing video url")

            try:
                base64_data = url_to_base64(video_url)
                session_manager.session_inject_media(
                    session_id=session_id,
                    media_type="video",
                    data=base64_data,
                    source_model=model,
                    remote_url=video_url,
                )
                logger.info(f"[{model.upper()}] video injected to session, model={model}, session_id={session_id}")
            except Exception as e:
                logger.warning(f"[{model.upper()}] failed to inject video to session: {e}")

            return Reply(ReplyType.VIDEO_URL, (self._duration_to_seconds(request_meta["duration"]), video_url))
        except LumaVideoClientFacingError as e:
            logger.warning(f"[{model.upper()}] client-facing luma video error: {e}")
            return Reply(ReplyType.ERROR, str(e))
        except Exception as e:
            logger.error(f"[{model.upper()}] fetch reply error: {e}")
            return Reply(ReplyType.ERROR, self._format_error_message(model, e))

    def _build_generation_params(self, query, session_id, model, video_mode, session_manager):
        quoted_cache = memory.USER_QUOTED_IMAGE_CACHE.get(session_id)
        file_cache = memory.USER_IMAGE_CACHE.get(session_id)
        quoted_video_cache = memory.USER_QUOTED_VIDEO_CACHE.get(session_id)
        video_cache = memory.USER_VIDEO_CACHE.get(session_id)
        has_direct_image_reference_source = bool(quoted_cache or file_cache)
        can_use_reference_video = self._should_use_reference_video(
            video_mode,
            has_direct_image_reference_source=has_direct_image_reference_source,
        )

        prompt_ratio = self._parse_aspect_ratio_from_prompt(query)
        if prompt_ratio:
            logger.info(f"[{model.upper()}] 从 prompt 中解析到比例: {prompt_ratio}")

        duration = self._normalize_duration(video_state.get_video_duration(session_id), model)
        resolution = self._normalize_resolution(video_state.get_video_resolution(session_id), model)
        aspect_ratio = prompt_ratio or self._normalize_ratio(conf().get("image_aspect_ratio", "16:9"), model)

        image_refs, image_source_files, image_source_name = self._collect_image_refs(
            session_id,
            session_manager,
            model,
            quoted_cache,
            file_cache,
        )
        reference_image_count = len(image_refs)
        if reference_image_count:
            if not prompt_ratio:
                aspect_ratio = self._infer_ratio_from_image_source(image_source_files, image_source_name, model)
            logger.info(f"[{model.upper()}] {image_source_name}取参考图, count={reference_image_count}")

        selected_video_cache = None
        source_video = None
        if can_use_reference_video:
            if quoted_video_cache:
                selected_video_cache = quoted_video_cache
                source_video = self._build_source_video(quoted_video_cache, model)
                logger.info(f"[{model.upper()}] 从回复引用视频取参考视频, count={1 if source_video else 0}")
                memory.USER_QUOTED_VIDEO_CACHE.pop(session_id, None)
            elif video_cache:
                selected_video_cache = video_cache
                source_video = self._build_source_video(video_cache, model)
                logger.info(f"[{model.upper()}] 从内存参考视频取参考视频, count={1 if source_video else 0}")
                memory.USER_VIDEO_CACHE.pop(session_id, None)
        else:
            if quoted_video_cache or video_cache:
                logger.info(f"[{model.upper()}] 当前为首尾帧模式且存在图片缓存，已跳过参考视频素材")
            memory.USER_QUOTED_VIDEO_CACHE.pop(session_id, None)
            memory.USER_VIDEO_CACHE.pop(session_id, None)

        if source_video and selected_video_cache:
            video_ratio = self._infer_aspect_ratio_from_video_cache(selected_video_cache, model)
            if video_ratio:
                aspect_ratio = video_ratio
                logger.info(f"[{model.upper()}] 从参考视频推断比例: {video_ratio}")
            video_resolution = self._infer_resolution_from_video_cache(selected_video_cache, model)
            if video_resolution:
                resolution = video_resolution
                logger.info(f"[{model.upper()}] 从参考视频推断分辨率: {video_resolution}")

        if reference_image_count:
            logger.info(
                f"[{model.upper()}] 图片角色识别结果: "
                f"{self._summarize_image_roles(image_refs, video_mode, bool(source_video))}"
            )

        if source_video:
            params = self._build_video_edit_params(query, model, resolution, source_video, image_refs, video_mode)
            request_type = "video_edit"
        else:
            duration = self._normalize_duration_for_image_anchors(duration, image_refs, video_mode, model)
            params = self._build_video_generation_params(query, model, aspect_ratio, resolution, duration, image_refs, video_mode)
            request_type = "video"

        return params, {
            "request_type": request_type,
            "reference_image_count": reference_image_count,
            "reference_video_count": 1 if source_video else 0,
            "resolution": resolution,
            "aspect_ratio": aspect_ratio,
            "duration": duration,
        }

    def _collect_image_refs(self, session_id, session_manager, model, quoted_cache, file_cache):
        if quoted_cache:
            image_refs = self._build_image_refs_from_pil(quoted_cache.get("files", []), model)
            memory.USER_QUOTED_IMAGE_CACHE.pop(session_id, None)
            return image_refs, quoted_cache.get("files", []), "从回复引用图"
        if file_cache:
            image_refs = self._build_image_refs_from_pil(file_cache.get("files", []), model)
            memory.USER_IMAGE_CACHE.pop(session_id, None)
            return image_refs, file_cache.get("files", []), "从内存参考图"

        session_images = get_image_urls_from_session(session_id, session_manager)
        if session_images:
            return self._build_image_refs_from_data_urls(session_images, model), session_images, "从 session 历史"
        return [], [], "无图片"

    def _build_video_generation_params(self, query, model, aspect_ratio, resolution, duration, image_refs, video_mode):
        video = {
            "resolution": resolution,
            "duration": duration,
        }
        normalized_mode = self._normalize_video_mode(video_mode)
        if image_refs and normalized_mode == "first_last":
            video["start_frame"] = image_refs[0]
            if len(image_refs) >= 2:
                video["end_frame"] = image_refs[1]
            if len(image_refs) > 2:
                logger.warning(
                    f"[{model.upper()}] 首尾帧模式仅使用前两张图片，其余 {len(image_refs) - 2} 张图片将忽略"
                )
        elif image_refs:
            video["keyframes"] = image_refs[:64]
            video["keyframe_indexes"] = self._build_keyframe_indexes(len(video["keyframes"]), duration)

        return {
            "model": model,
            "type": "video",
            "prompt": query,
            "aspect_ratio": aspect_ratio,
            "video": video,
        }

    def _build_video_edit_params(self, query, model, resolution, source_video, image_refs, video_mode):
        video = {
            "resolution": resolution,
            "edit": {
                "auto_controls": True,
            },
        }
        normalized_mode = self._normalize_video_mode(video_mode)
        if image_refs and normalized_mode == "first_last":
            video["start_frame"] = image_refs[0]
            if len(image_refs) > 1:
                logger.warning(
                    f"[{model.upper()}] Luma video_edit 仅支持单张 start_frame 指导图，"
                    f"其余 {len(image_refs) - 1} 张图片将忽略"
                )
        elif image_refs:
            video["edit"]["keyframes"] = image_refs[:64]
            video["edit"]["keyframe_indexes"] = self._build_keyframe_indexes(len(video["edit"]["keyframes"]), "5s")

        return {
            "model": model,
            "type": "video_edit",
            "prompt": query,
            "source": source_video,
            "video": video,
        }

    def _build_image_refs_from_pil(self, images, model):
        image_refs = []
        for image in images:
            image_ref = self._encode_pil_image_ref(image, model)
            if image_ref:
                image_refs.append(image_ref)
        return image_refs[:64]

    def _build_image_refs_from_data_urls(self, data_urls, model):
        image_refs = []
        for data_url in data_urls:
            image_ref = self._data_url_to_image_ref(data_url, model)
            if image_ref:
                image_refs.append(image_ref)
        return image_refs[:64]

    def _encode_pil_image_ref(self, image, model):
        try:
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")
            buf = io.BytesIO()
            image.save(buf, format="JPEG", quality=95)
            image_data = self._ensure_reference_image_within_limit(buf.getvalue(), model)
            return {
                "data": base64.b64encode(image_data).decode("utf-8"),
                "media_type": "image/jpeg",
            }
        except Exception as e:
            logger.warning(f"[{model.upper()}] failed to encode cached image: {e}")
            return None

    def _data_url_to_image_ref(self, data_url, model):
        try:
            header, data = str(data_url).split(",", 1)
            media_type = "image/jpeg"
            if header.startswith("data:") and ";" in header:
                media_type = header[5:header.index(";")]
            return {
                "data": data,
                "media_type": media_type,
            }
        except Exception as e:
            logger.warning(f"[{model.upper()}] failed to parse session image data url: {e}")
            return None

    def _ensure_reference_image_within_limit(self, image_data, model):
        if len(image_data) <= _LUMA_REFERENCE_IMAGE_MAX_BYTES:
            return image_data
        logger.warning(
            f"[{model.upper()}] reference image too large, start compressing, "
            f"size={len(image_data)} bytes, limit={_LUMA_REFERENCE_IMAGE_MAX_BYTES}"
        )
        return self._compress_image_bytes(image_data, model)

    def _compress_image_bytes(self, image_data, model):
        try:
            image = Image.open(io.BytesIO(image_data))
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            quality = 90
            width, height = image.size
            resized = image
            while True:
                buf = io.BytesIO()
                resized.save(buf, format="JPEG", quality=quality, optimize=True)
                compressed = buf.getvalue()
                if len(compressed) <= _LUMA_REFERENCE_IMAGE_MAX_BYTES:
                    logger.info(f"[{model.upper()}] reference image compressed, size={len(compressed)} bytes")
                    return compressed
                if quality > 55:
                    quality -= 10
                    continue
                width = max(int(width * 0.85), 512)
                height = max(int(height * 0.85), 512)
                if (width, height) == resized.size:
                    logger.warning(f"[{model.upper()}] reference image still exceeds limit after compression")
                    return compressed
                resized = image.resize((width, height), Image.LANCZOS)
                quality = 85
        except Exception as e:
            logger.warning(f"[{model.upper()}] failed to compress reference image: {e}")
            return image_data

    def _build_source_video(self, video_cache, model):
        for video_file in video_cache.get("files", []):
            media_type = video_file.get("mime_type") or "video/mp4"
            path = video_file.get("path")
            if path:
                try:
                    if os.path.getsize(path) > _LUMA_REFERENCE_VIDEO_MAX_BYTES:
                        logger.warning(f"[{model.upper()}] reference video exceeds Luma 200MB limit, skipped: {path}")
                        continue
                    with open(path, "rb") as video:
                        logger.info(f"[{model.upper()}] reference video encoded from local path={path}")
                        return {
                            "data": base64.b64encode(video.read()).decode("utf-8"),
                            "media_type": media_type,
                        }
                except Exception as e:
                    logger.warning(f"[{model.upper()}] failed to encode reference video: {e}")

            public_url = video_file.get("public_url")
            if public_url:
                logger.info(f"[{model.upper()}] reference video fallback public_url={public_url}")
                return {
                    "url": public_url,
                    "media_type": media_type,
                }

            logger.warning(f"[{model.upper()}] reference video missing public_url/path, skipped")
        return None

    def _build_keyframe_indexes(self, image_count, duration):
        if image_count <= 1:
            return [0] if image_count == 1 else []
        max_index = 240 if duration == "10s" else 120
        return [round(i * max_index / (image_count - 1)) for i in range(image_count)]

    def _summarize_image_roles(self, image_refs, video_mode, has_source_video):
        normalized_mode = self._normalize_video_mode(video_mode)
        if normalized_mode == "first_last":
            if has_source_video:
                return ",".join(["start_frame"] + ["ignored"] * max(len(image_refs) - 1, 0))
            roles = ["first_frame"]
            if len(image_refs) >= 2:
                roles.append("last_frame")
            roles.extend(["ignored"] * max(len(image_refs) - 2, 0))
            return ",".join(roles)
        if has_source_video:
            return ",".join(["edit_keyframe"] * len(image_refs))
        return ",".join(["keyframe"] * len(image_refs))

    def _poll_generation(self, generation, model, max_retries=120, interval=2):
        last_log_time = 0
        for i in range(max_retries):
            state = getattr(generation, "state", None)
            if state == "completed":
                logger.info(f"[{model.upper()}] task completed, id={generation.id}")
                return generation
            if state == "failed":
                raise LumaVideoClientFacingError(self._format_async_failure(generation, model))

            time.sleep(interval)
            try:
                generation = self.client.generations.get(generation.id)
            except APIStatusError as e:
                error_message = self._format_sync_api_error(e, model, endpoint="GET")
                logger.warning(f"[{model.upper()}] Luma video poll request failed: {error_message}")
                raise LumaVideoClientFacingError(error_message)
            now = time.time()
            if now - last_log_time >= 10:
                logger.info(
                    f"[{model.upper()}] 轮询中 ({i + 1}/{max_retries}), "
                    f"state={generation.state}, id={generation.id}"
                )
                last_log_time = now

        raise TimeoutError("任务超时，请稍后重试")

    def _extract_output_url(self, generation):
        output = getattr(generation, "output", None) or []
        for item in output:
            url = getattr(item, "url", None)
            if url:
                return url
            if isinstance(item, dict) and item.get("url"):
                return item["url"]
        return None

    def _normalize_video_mode(self, video_mode):
        normalized = str(video_mode or "").strip().lower()
        if normalized == "firstlast":
            return "first_last"
        if normalized == "reference":
            return "reference"
        return "reference"

    def _should_use_reference_video(self, video_mode, has_direct_image_reference_source):
        normalized_mode = self._normalize_video_mode(video_mode)
        if normalized_mode == "reference":
            return True
        if normalized_mode == "first_last":
            return not has_direct_image_reference_source
        return False

    def _parse_aspect_ratio_from_prompt(self, prompt):
        ratio_candidates = {value: key for key, value in _LUMA_VIDEO_RATIO_MAP.items()}
        aspect_ratio = parse_aspect_ratio_from_prompt(
            prompt,
            ratio_map=ratio_candidates,
            decimal_tolerance=0.25,
            ratio_tolerance=0.3,
        )
        return self._normalize_ratio(aspect_ratio, const.LUMA_RAY_32) if aspect_ratio else None

    def _infer_ratio_from_image_source(self, image_source_files, image_source_name, model):
        if not image_source_files:
            return self._normalize_ratio(conf().get("image_aspect_ratio", "16:9"), model)
        if isinstance(image_source_files[0], str):
            ratio = size_calculator_from_data_urls(image_source_files)
        else:
            ratio = size_calculator(image_source_files)
        normalized = self._normalize_ratio(ratio, model)
        logger.info(f"[{model.upper()}] {image_source_name}推断比例: {normalized}")
        return normalized

    def _infer_aspect_ratio_from_video_cache(self, video_cache, model):
        ratio = infer_aspect_ratio_from_video_cache(
            video_cache,
            lambda video_size: self._normalize_ratio(aspect_ratio_from_size(video_size), model),
        )
        if ratio is None:
            logger.warning(f"[{model.upper()}] failed to infer aspect ratio from reference video cache")
        return ratio

    def _infer_resolution_from_video_cache(self, video_cache, model):
        resolution = infer_resolution_from_video_cache(video_cache, _LUMA_VIDEO_RESOLUTIONS)
        if resolution is None:
            logger.warning(f"[{model.upper()}] failed to infer resolution from reference video cache")
            return None
        return self._normalize_resolution(resolution, model)

    def _normalize_ratio(self, ratio, model):
        if ratio in _LUMA_VIDEO_RATIO_MAP:
            return ratio
        ratio_value = self._ratio_to_float(ratio)
        if ratio_value is None:
            logger.warning(f"[{model.upper()}] unsupported ratio={ratio}, fallback to 16:9")
            return "16:9"
        normalized_ratio = min(_LUMA_VIDEO_RATIO_MAP, key=lambda key: abs(_LUMA_VIDEO_RATIO_MAP[key] - ratio_value))
        logger.info(f"[{model.upper()}] ratio {ratio} 不在白名单内，已映射为 {normalized_ratio}")
        return normalized_ratio

    def _normalize_duration(self, duration, model):
        try:
            seconds = int(duration)
        except (TypeError, ValueError):
            logger.warning(f"[{model.upper()}] invalid duration={duration}, fallback to 5s")
            return "5s"
        normalized = f"{seconds}s"
        if normalized in _LUMA_VIDEO_DURATIONS:
            return normalized
        logger.warning(f"[{model.upper()}] invalid duration={duration}, fallback to 5s")
        return "5s"

    def _normalize_duration_for_image_anchors(self, duration, image_refs, video_mode, model):
        if (
            duration == "10s"
            and image_refs
            and self._normalize_video_mode(video_mode) == "first_last"
        ):
            logger.warning(f"[{model.upper()}] Luma start_frame/end_frame 不支持 10s，已回退到 5s")
            return "5s"
        return duration

    def _normalize_resolution(self, resolution, model):
        normalized = str(resolution).strip().lower()
        if normalized in _LUMA_VIDEO_RESOLUTIONS:
            return normalized
        logger.warning(f"[{model.upper()}] invalid resolution={resolution}, fallback to 720p")
        return "720p"

    def _ratio_to_float(self, ratio):
        try:
            width, height = str(ratio).split(":", 1)
            return float(width) / float(height)
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    def _duration_to_seconds(self, duration):
        try:
            return int(str(duration).rstrip("s"))
        except (TypeError, ValueError):
            return 5

    def _format_sync_api_error(self, error, model, endpoint="POST"):
        status_code = self._extract_luma_status_code(error)
        detail = self._extract_luma_error_detail(error)
        retry_after = self._extract_luma_header(error, "Retry-After")
        request_id = self._extract_luma_header(error, "X-Request-Id")

        detail_text = f"（{detail}）" if detail else ""
        retry_text = f"，建议 {retry_after} 秒后再试" if retry_after else ""
        request_text = f" request_id={request_id}" if request_id else ""

        if endpoint == "GET":
            status_messages = {
                401: f"Luma 轮询鉴权失败{detail_text}，请检查 API Key 配置。",
                404: f"Luma 任务不存在或无权访问{detail_text}，请确认任务 ID 和 API Key 是否匹配。",
            }
        else:
            status_messages = {
                400: f"Luma 请求参数有误{detail_text}，请调整后重试。",
                401: f"Luma API 鉴权失败{detail_text}，请检查 API Key 配置。",
                402: f"Luma 账户额度不足{detail_text}，请充值后再试。",
                403: f"Luma 访问被拒绝{detail_text}，请检查账号状态或权限。",
                413: f"参考素材超过 Luma 限制{detail_text}，请压缩后再试。",
                422: f"Luma 无法处理当前视频请求{detail_text}，请检查参考素材是否有效、可读取，或调整参数组合。",
                429: f"Luma 当前请求过多{detail_text}{retry_text}。",
                502: f"Luma 上游服务暂时不可用{detail_text}，请稍后重试。",
                503: f"Luma 视频接入服务暂时不可用{detail_text}，请稍后重试。",
            }
        message = status_messages.get(
            status_code,
            f"Luma {endpoint} 请求失败(status={status_code}){detail_text}，请稍后重试。",
        )
        return f"[{model.upper()}] {message}{request_text}"

    def _format_async_failure(self, generation, model):
        generation_id = getattr(generation, "id", None)
        reason = getattr(generation, "failure_reason", None) or "Luma 未返回失败原因"
        code = getattr(generation, "failure_code", None) or "unknown"

        code_messages = {
            "content_moderated": "Luma 内容审核未通过，请调整提示词或参考素材后重新提交。",
            "generation_failed": "Luma 视频生成过程中发生临时错误，可以稍后重试同一请求。",
            "budget_exhausted": "Luma 账户额度在生成过程中耗尽，请充值后重新提交。",
            "output_not_found": "Luma 生成结果暂时无法取回，可以稍后重试同一请求。",
        }
        message = code_messages.get(code, "Luma 视频任务失败，请根据失败原因调整后重试。")
        id_text = f" generation_id={generation_id}" if generation_id else ""
        return f"[{model.upper()}] {message}（{reason}，failure_code={code}）{id_text}"

    def _format_error_message(self, model, error):
        error_text = str(error)
        if error_text.startswith(f"[{model.upper()}]"):
            return error_text
        return f"[{model.upper()}] {error_text}"

    def _extract_luma_status_code(self, error):
        status_code = getattr(error, "status_code", None)
        response = getattr(error, "response", None)
        if status_code is None and response is not None:
            status_code = getattr(response, "status_code", None)
        return status_code or "unknown"

    def _extract_luma_error_detail(self, error):
        body = getattr(error, "body", None)
        if isinstance(body, dict):
            detail = body.get("detail")
            if detail:
                return str(detail)

        response = getattr(error, "response", None)
        if response is not None:
            try:
                data = response.json()
                if isinstance(data, dict) and data.get("detail"):
                    return str(data["detail"])
            except Exception:
                pass

        message = str(error).strip()
        return message or "Luma API 未返回错误详情"

    def _extract_luma_header(self, error, header_name):
        response = getattr(error, "response", None)
        headers = getattr(response, "headers", None) if response is not None else None
        if not headers:
            return None
        return headers.get(header_name) or headers.get(header_name.lower())
