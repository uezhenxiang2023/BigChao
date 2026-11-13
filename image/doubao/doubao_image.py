import base64
import io
import re

from PIL import Image
from volcenginesdkarkruntime import Ark
from volcenginesdkarkruntime.types.images.images import SequentialImageGenerationOptions

from bot.bot import Bot
from bot.ark.ark_error import (
    format_ark_api_error,
    format_ark_connection_error,
    is_ark_connection_error,
    is_ark_status_error,
    is_ark_timeout_error,
)
from bot.ark.ark_media import (
    build_seedream_size,
    encode_image,
    size_calculator,
)
from bridge.context import Context
from bridge.reply import Reply, ReplyType
from common import const
from common import memory
from common.aspect_ratio import parse_aspect_ratio_from_prompt
from common.log import logger
from common.model_status import model_state
from common.utils import get_chat_session_manager, url_to_base64
from config import conf


class DoubaoImageBot(Bot):
    _MAX_REFERENCE_IMAGE_BYTES = int(9.5 * 1024 * 1024)
    _SEQUENTIAL_IMAGE_MAX_COUNT = 15
    _SEQUENTIAL_IMAGE_CN_NUM_MAP = {
        "两": 2,
        "俩": 2,
        "仨": 3,
        "一": 1,
        "二": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
        "十": 10,
    }

    def __init__(self):
        super().__init__()
        self.client = Ark(api_key=conf().get("ark_api_key"))

    def reply(self, query, context: Context = None) -> Reply:
        error_model = const.DOUBAO_SEEDREAM_55
        try:
            session_id = context["session_id"]
            model = model_state.get_image_model(session_id)
            image_mode = model_state.get_image_mode(session_id)
            error_model = model
            session_manager = get_chat_session_manager(session_id)
            logger.info(f"[{model.upper()}] query={query}, requester={session_id}")
            session_manager.session_query(query, session_id)

            params = {
                "model": model,
                "prompt": query,
                "response_format": "url",
                "watermark": False,
            }
            # 仅在支持组图特性的模型（如 Seedream 4.0/4.5/5.0）上传递 sequential_image_generation
            if model in {const.DOUBAO_SEEDREAM_4, const.DOUBAO_SEEDREAM_45, const.DOUBAO_SEEDREAM_5}:
                params["sequential_image_generation"] = "disabled"
                sequential_image_count = self._parse_sequential_image_count_from_prompt(query)
                if sequential_image_count:
                    params.update({
                        "sequential_image_generation": "auto",
                        "sequential_image_generation_options": SequentialImageGenerationOptions(
                            max_images=sequential_image_count
                        )
                    })
                    logger.info(
                        f"[{model.upper()}] 从 prompt 中解析到组图数量: {sequential_image_count}, "
                        "已启用 sequential_image_generation=auto"
                    )

            # Seedream 5.5 / 5.0-pro 原生透明背景支持
            if model in {const.DOUBAO_SEEDREAM_55, const.DOUBAO_SEEDREAM_5_PRO} and self._is_transparent_background_requested(query):
                params["extra_body"] = {"background": "transparent"}
                params["output_format"] = "png"
                logger.info(
                    f"[{model.upper()}] 从 prompt 中解析到透明背景需求, "
                    "已启用 background=transparent, output_format=png"
                )
            default_aspect_ratio = conf().get("image_aspect_ratio", "16:9")
            prompt_aspect_ratio = self._parse_aspect_ratio_from_prompt(query)
            if prompt_aspect_ratio:
                logger.info(f"[{model.upper()}] 从 prompt 中解析到比例: {prompt_aspect_ratio}")

            images = []
            aspect_ratio_files = None
            quoted_cache = memory.USER_QUOTED_IMAGE_CACHE.get(session_id)
            file_cache = memory.USER_IMAGE_CACHE.get(session_id)
            if quoted_cache:
                quoted_images = [
                    encode_image(path, file)
                    for path, file in zip(quoted_cache["path"], quoted_cache["files"])
                ]
                quoted_images = [
                    self._ensure_reference_image_within_limit(image_url, model)
                    for image_url in quoted_images
                ]
                images.extend(quoted_images)
                aspect_ratio_files = quoted_cache["files"]
                logger.info(f"[{model.upper()}] 从回复引用图取参考图, count={len(quoted_images)}")
                memory.USER_QUOTED_IMAGE_CACHE.pop(session_id)

            if file_cache:
                cached_images = [
                    encode_image(path, file)
                    for path, file in zip(file_cache["path"], file_cache["files"])
                ]
                cached_images = [
                    self._ensure_reference_image_within_limit(image_url, model)
                    for image_url in cached_images
                ]
                images.extend(cached_images)
                if aspect_ratio_files is None:
                    aspect_ratio_files = file_cache["files"]
                logger.info(f"[{model.upper()}] 从内存参考图追加入参, count={len(cached_images)}")
                memory.USER_IMAGE_CACHE.pop(session_id)

            if images:
                aspect_ratio = prompt_aspect_ratio or size_calculator(aspect_ratio_files)
                image_size = self._build_image_size(session_id, model, aspect_ratio)
                params.update({
                    "image": images,
                    "size": image_size
                })
                logger.info(
                    f"[{model.upper()}] request summary: mode={image_mode}, reference_count={len(images)}, "
                    f"aspect_ratio={aspect_ratio}, image_size={image_size}"
                )
                if not prompt_aspect_ratio:
                    logger.info(
                        f"[{model.upper()}] 从参考图推断比例: {aspect_ratio}, "
                        f"size={image_size}, count={len(aspect_ratio_files)}"
                    )
            else:
                aspect_ratio = prompt_aspect_ratio or default_aspect_ratio
                image_size = self._build_image_size(session_id, model, aspect_ratio)
                params["size"] = image_size
                logger.info(
                    f"[{model.upper()}] 当前为文生图模式, mode={image_mode}, aspect_ratio={aspect_ratio}, image_size={image_size}"
                )

            try:
                response = self.client.images.generate(**params)
            except Exception as e:
                if is_ark_status_error(e):
                    error_message = format_ark_api_error(e, model, service_name="豆包图片")
                elif is_ark_timeout_error(e):
                    error_message = format_ark_connection_error(e, model, error_type="timeout")
                elif is_ark_connection_error(e):
                    error_message = format_ark_connection_error(e, model, error_type="connection")
                else:
                    raise
                logger.warning(f"[{model.upper()}] Ark image request failed: {error_message}")
                return Reply(ReplyType.ERROR, error_message)
            image_urls = [item.url for item in getattr(response, "data", []) if getattr(item, "url", None)]
            if not image_urls:
                raise ValueError("Doubao image response missing image url")

            try:
                for image_url in image_urls:
                    base64_data = url_to_base64(image_url)
                    session_manager.session_inject_media(
                        session_id=session_id,
                        media_type="image",
                        data=base64_data,
                        source_model=model
                    )
                logger.info(
                    f"[{model.upper()}] image injected to session, model={model}, "
                    f"session_id={session_id}, image_count={len(image_urls)}"
                )
            except Exception as e:
                logger.warning(f"[{model.upper()}] failed to inject image to session: {e}")

            logger.info(f"[{model.upper()}] image generation finished, image_count={len(image_urls)}")
            if len(image_urls) == 1:
                return Reply(ReplyType.IMAGE_URL, image_urls[0])
            return Reply(ReplyType.IMAGE_URL, image_urls)
        except Exception as e:
            logger.error(f"[{error_model.upper()}] fetch reply error: {e}")
            return Reply(ReplyType.ERROR, f"[{error_model.upper()}] {e}")

    def _build_image_size(self, session_id, model, aspect_ratio):
        image_size = model_state.get_image_size(session_id)
        if model in const.DOUBAO_SEEDREAM_LIST:
            return build_seedream_size(model, image_size, aspect_ratio)
        return image_size

    def _ensure_reference_image_within_limit(self, image_url, model):
        original_size = len(image_url.encode("utf-8"))
        if original_size <= self._MAX_REFERENCE_IMAGE_BYTES:
            return image_url

        logger.warning(
            f"[{model.upper()}] reference image too large, start compressing, size={original_size} bytes, "
            f"limit={self._MAX_REFERENCE_IMAGE_BYTES}"
        )
        compressed_url = self._compress_data_url(image_url, model)
        compressed_size = len(compressed_url.encode("utf-8"))
        logger.info(
            f"[{model.upper()}] reference image compressed, before={original_size} bytes, "
            f"after={compressed_size} bytes"
        )
        return compressed_url

    def _compress_data_url(self, image_url, model):
        try:
            header, b64_data = image_url.split(",", 1)
            image = Image.open(io.BytesIO(base64.b64decode(b64_data)))
            if image.mode in ("RGBA", "LA", "P"):
                image = image.convert("RGB")

            quality = 90
            width, height = image.size
            resized = image
            while True:
                buf = io.BytesIO()
                resized.save(buf, format="JPEG", quality=quality, optimize=True)
                encoded = base64.b64encode(buf.getvalue()).decode("utf-8")
                compressed_url = f"data:image/jpeg;base64,{encoded}"
                if len(compressed_url.encode("utf-8")) <= self._MAX_REFERENCE_IMAGE_BYTES:
                    return compressed_url
                if quality > 55:
                    quality -= 10
                    continue
                width = max(int(width * 0.85), 512)
                height = max(int(height * 0.85), 512)
                if (width, height) == resized.size:
                    logger.warning(f"[{model.upper()}] reference image still exceeds limit after compression")
                    return compressed_url
                resized = image.resize((width, height), Image.LANCZOS)
                quality = 85
        except Exception as e:
            logger.warning(f"[{model.upper()}] failed to compress reference image: {e}")
            return image_url

    def _parse_aspect_ratio_from_prompt(self, prompt):
        return parse_aspect_ratio_from_prompt(prompt)

    def _parse_sequential_image_count_from_prompt(self, prompt):
        if not prompt:
            return None

        normalized_prompt = prompt.replace("张图", "张").replace("幅图", "幅")

        digit_patterns = [
            r"(?:组图|一组|做|生成|输出|画|给我|来|要)\s*(\d{1,2})\s*(?:张|幅)",
            r"(\d{1,2})\s*(?:张|幅)\s*(?:组图|系列图|连环图)",
            r"(?:共|一共|最多|至多|最多生成)\s*(\d{1,2})\s*(?:张|幅)",
        ]
        for pattern in digit_patterns:
            match = re.search(pattern, normalized_prompt, re.IGNORECASE)
            if match:
                return self._clamp_sequential_image_count(int(match.group(1)))

        cn_pattern = r"(两|俩|仨|[一二三四五六七八九十])\s*(?:张|幅)\s*(?:组图|系列图|连环图)?"
        match = re.search(cn_pattern, normalized_prompt)
        if match:
            parsed = self._SEQUENTIAL_IMAGE_CN_NUM_MAP.get(match.group(1))
            if parsed:
                return self._clamp_sequential_image_count(parsed)

        return None

    def _clamp_sequential_image_count(self, count):
        if count <= 1:
            return None
        return min(count, self._SEQUENTIAL_IMAGE_MAX_COUNT)

    @staticmethod
    def _is_transparent_background_requested(prompt: str) -> bool:
        if not prompt:
            return False
        keywords = [
            "透明背景", "透明底", "透明图", "透明通道", "免抠", "抠图", 
            "透明png", "transparent background", "transparent"
        ]
        prompt_lower = prompt.lower()
        return any(kw in prompt_lower for kw in keywords)
