import base64
import io

from PIL import Image

from common import const
from common.utils import get_ark_sessions
from common.log import logger
from config import conf


def encode_image_content(image_path, image_file):
    """将图片组装为多模态会话消息块。"""
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{_get_image_mime_type(image_file)};base64,{base64_image}"
        }
    }


def encode_image(image_path, image_file):
    """将seedream中引用的内存图片转为生成模型可用的 data URL。"""
    with open(image_path, "rb") as file:
        base64_image = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{_get_image_mime_type(image_file)};base64,{base64_image}"


def process_image_files(file_cache):
    """处理seed和seedance中的图片缓存，返回按原顺序编码后的图片内容列表。"""
    if not file_cache:
        return []

    image_contents = []
    image_files = file_cache["files"]
    image_paths = file_cache["path"]
    for image_path, image_file in zip(image_paths, image_files):
        image_contents.append(encode_image_content(image_path, image_file))
    return image_contents


def encode_video_content(video_path, video_file, fps=1):
    """将视频组装为多模态会话消息块，优先使用公网 URL，回退到 data URL。"""
    video_url = encode_video(video_path, video_file)
    return {
        "type": "video_url",
        "video_url": {
            "url": video_url,
            "fps": fps,
        }
    }


def encode_video(video_path, video_file):
    """将视频转为模型可用的 URL 或 data URL。"""
    public_url = video_file.get("public_url") if isinstance(video_file, dict) else None
    input_mode = str(conf().get("ark_video_input_mode", "base64")).strip().lower()
    if input_mode not in {"base64", "public_url"}:
        logger.warning(f"[ArkMedia] invalid ark_video_input_mode={input_mode}, fallback to base64")
        input_mode = "base64"
    if input_mode == "public_url" and public_url:
        return public_url
    if public_url:
        logger.info("[ArkMedia] ark video input mode=base64, using inline data URL")
    mime_type = _get_video_mime_type(video_path, video_file)
    with open(video_path, "rb") as file:
        base64_video = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{base64_video}"


def process_video_files(file_cache, fps=1):
    """处理视频缓存，返回按原顺序编码后的视频内容列表。"""
    if not file_cache:
        return []

    video_contents = []
    for video_file in file_cache.get("files", []):
        video_path = video_file.get("path") if isinstance(video_file, dict) else None
        if not video_path:
            logger.warning("[ArkMedia] video cache item missing path, skipped")
            continue
        video_contents.append(encode_video_content(video_path, video_file, fps=fps))
    return video_contents


def size_calculator(files):
    """根据参考图推断最接近的豆包出图分辨率。"""
    sizes_list = [file.size for file in files]
    best_size = sorted(sizes_list, key=lambda x: x[0] * x[1], reverse=True)[0]
    return aspect_ratio_from_size(best_size)


def size_calculator_from_data_urls(image_urls):
    """根据 session 中的 data URL 图片推断最接近的豆包出图比例。"""
    sizes_list = [_decode_image_size(image_url) for image_url in image_urls]
    best_size = sorted(sizes_list, key=lambda x: x[0] * x[1], reverse=True)[0]
    return aspect_ratio_from_size(best_size)


def get_image_from_session(session_id):
    """从会话历史中倒序提取最近一条图片消息。"""
    session = get_ark_sessions().build_session(session_id)
    for msg in reversed(session.messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        images = [
            item["image_url"]["url"]
            for item in content
            if item.get("type") == "image_url"
        ]
        if images:
            return images
    return []


def _get_image_mime_type(image_file):
    image_type = type(image_file).__name__
    if image_type == "PngImageFile":
        return "image/png"
    return "image/jpeg"


def _get_video_mime_type(video_path, video_file):
    mime_type = video_file.get("mime_type") if isinstance(video_file, dict) else None
    if isinstance(mime_type, str) and mime_type:
        return mime_type
    suffix = video_path.rsplit(".", 1)[-1].lower() if "." in video_path else "mp4"
    return f"video/{suffix}"


def _decode_image_size(image_url):
    b64_data = image_url.split(",", 1)[1]
    img = Image.open(io.BytesIO(base64.b64decode(b64_data)))
    return img.size


def aspect_ratio_from_size(size):
    ratio_map = {
        1.0: "1:1",
        1.33: "4:3",
        0.75: "3:4",
        1.78: "16:9",
        0.56: "9:16",
        1.5: "3:2",
        0.67: "2:3",
        2.33: "21:9",
    }
    ratio = round(size[0] / size[1], 2)
    closest_ratio = min(ratio_map.keys(), key=lambda x: abs(x - ratio))
    return ratio_map[closest_ratio]


def build_seedream_size(model, configured_size, aspect_ratio):
    normalized_size = str(configured_size).strip().upper()
    normalized_ratio = str(aspect_ratio).strip()
    supported_sizes = _get_seedream_supported_sizes(model)
    if normalized_size not in supported_sizes:
        fallback_size = supported_sizes[0]
        logger.warning(
            f"[DoubaoImage] invalid size={configured_size} for model={model}, fallback to {fallback_size}"
        )
        normalized_size = fallback_size

    ratio_size_map = _get_seedream_ratio_size_map(model, normalized_size)
    if normalized_ratio in ratio_size_map:
        return ratio_size_map[normalized_ratio]

    if normalized_ratio and normalized_ratio != "auto":
        logger.warning(
            f"[DoubaoImage] unsupported aspect_ratio={aspect_ratio} for model={model}, size={normalized_size}, "
            f"fallback to resolution tier {normalized_size}"
        )
    return normalized_size


def _get_seedream_supported_sizes(model):
    if model == const.DOUBAO_SEEDREAM_5:
        return ["2K", "3K", "4K"]
    if model == const.DOUBAO_SEEDREAM_45:
        return ["2K", "4K"]
    if model == const.DOUBAO_SEEDREAM_4:
        return ["1K", "2K", "4K"]
    return ["2K"]


def _get_seedream_ratio_size_map(model, image_size):
    if model == const.DOUBAO_SEEDREAM_5:
        return {
            "2K": {
                "1:1": "2048x2048",
                "3:4": "1728x2304",
                "4:3": "2304x1728",
                "16:9": "2848x1600",
                "9:16": "1600x2848",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "21:9": "3136x1344",
            },
            "3K": {
                "1:1": "3072x3072",
                "3:4": "2592x3456",
                "4:3": "3456x2592",
                "16:9": "4096x2304",
                "9:16": "2304x4096",
                "3:2": "3744x2496",
                "2:3": "2496x3744",
                "21:9": "4704x2016",
            },
            "4K": {
                "1:1": "4096x4096",
                "3:4": "3520x4704",
                "4:3": "4704x3520",
                "16:9": "5504x3040",
                "9:16": "3040x5504",
                "3:2": "4992x3328",
                "2:3": "3328x4992",
                "21:9": "6240x2656",
            },
        }.get(image_size, {})
    if model in {const.DOUBAO_SEEDREAM_45, const.DOUBAO_SEEDREAM_4}:
        return {
            "1K": {
                "1:1": "1024x1024",
                "3:4": "864x1152",
                "4:3": "1152x864",
                "16:9": "1312x736",
                "9:16": "736x1312",
                "3:2": "1248x832",
                "2:3": "832x1248",
                "21:9": "1568x672",
            },
            "2K": {
                "1:1": "2048x2048",
                "3:4": "1728x2304",
                "4:3": "2304x1728",
                "16:9": "2848x1600",
                "9:16": "1600x2848",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "21:9": "3136x1344",
            },
            "4K": {
                "1:1": "4096x4096",
                "3:4": "3520x4704",
                "4:3": "4704x3520",
                "16:9": "5504x3040",
                "9:16": "3040x5504",
                "3:2": "4992x3328",
                "2:3": "3328x4992",
                "21:9": "6240x2656",
            },
        }.get(image_size, {})
    return {}
