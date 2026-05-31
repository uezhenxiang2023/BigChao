import io
import os
import requests, base64
import re
import subprocess

from urllib.parse import urlparse
from PIL import Image


def fsize(file):
    if isinstance(file, io.BytesIO):
        return file.getbuffer().nbytes
    elif isinstance(file, str):
        return os.path.getsize(file)
    elif hasattr(file, "seek") and hasattr(file, "tell"):
        pos = file.tell()
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(pos)
        return size
    else:
        raise TypeError("Unsupported type")


def compress_imgfile(file, max_size):
    if fsize(file) <= max_size:
        return file
    file.seek(0)
    img = Image.open(file)
    rgb_image = img.convert("RGB")
    quality = 95
    while True:
        out_buf = io.BytesIO()
        rgb_image.save(out_buf, "JPEG", quality=quality)
        if fsize(out_buf) <= max_size:
            return out_buf
        quality -= 5


def split_string_by_utf8_length(string, max_length, max_split=0):
    encoded = string.encode("utf-8")
    start, end = 0, 0
    result = []
    while end < len(encoded):
        if max_split > 0 and len(result) >= max_split:
            result.append(encoded[start:].decode("utf-8"))
            break
        end = min(start + max_length, len(encoded))
        # 如果当前字节不是 UTF-8 编码的开始字节，则向前查找直到找到开始字节为止
        while end < len(encoded) and (encoded[end] & 0b11000000) == 0b10000000:
            end -= 1
        result.append(encoded[start:end].decode("utf-8"))
        start = end
    return result


def get_path_suffix(path):
    path = urlparse(path).path
    return os.path.splitext(path)[-1].lstrip('.')

def url_to_base64(url: str) -> str:
        """下载 URL 内容并转为 base64 字符串"""
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return base64.b64encode(response.content).decode('utf-8')

# utils.py
def get_ark_sessions():
    from bot.ark.volcengine_ark_bot import _ark_sessions
    return _ark_sessions


def get_chat_session_manager(session_id):
    try:
        from bridge.bridge import Bridge

        bot = Bridge(session_id).get_bot("chat")
        sessions = getattr(bot, "sessions", None)
        if sessions is not None:
            return sessions
    except Exception:
        pass
    return get_ark_sessions()


def get_image_urls_from_session(session_id, session_manager=None):
    session_manager = session_manager or get_chat_session_manager(session_id)
    session = session_manager.build_session(session_id)
    images = []
    for msg in session.messages:
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "image_url":
                continue
            image_info = item.get("image_url", {})
            image_url = image_info.get("url")
            if isinstance(image_url, str):
                images.append(image_url)
    return images


def get_video_urls_from_session(session_id, session_manager=None, include_data_urls=False):
    session_manager = session_manager or get_chat_session_manager(session_id)
    session = session_manager.build_session(session_id)
    for msg in reversed(session.messages):
        content = msg.get("content")
        if not isinstance(content, list):
            continue
        videos = []
        for item in content:
            if not isinstance(item, dict) or item.get("type") != "video_url":
                continue
            video_info = item.get("video_url", {})
            video_url = video_info.get("remote_url") or video_info.get("url")
            if not isinstance(video_url, str):
                continue
            if include_data_urls or video_url.startswith(("http://", "https://")):
                videos.append(video_url)
        if videos:
            return videos
    return []


def get_video_dimensions(video_path):
    try:
        result = subprocess.run(
            ["ffmpeg", "-i", video_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        output = f"{result.stdout}\n{result.stderr}"
        match = re.search(r"(\d{2,5})x(\d{2,5})", output)
        if not match:
            return None
        return int(match.group(1)), int(match.group(2))
    except Exception:
        return None


def infer_aspect_ratio_from_video_cache(video_cache, size_to_ratio):
    if not video_cache:
        return None
    for video_file in video_cache.get("files", []):
        video_path = video_file.get("path")
        if not video_path:
            continue
        video_size = get_video_dimensions(video_path)
        if not video_size:
            continue
        return size_to_ratio(video_size)
    return None


def infer_resolution_from_video_cache(video_cache, allowed_resolutions=None):
    if not video_cache:
        return None
    resolution_levels = [480, 720, 1080]
    if allowed_resolutions:
        normalized = []
        for resolution in allowed_resolutions:
            try:
                normalized.append(int(str(resolution).rstrip("pP")))
            except (TypeError, ValueError):
                continue
        if normalized:
            resolution_levels = sorted(set(normalized))
    for video_file in video_cache.get("files", []):
        video_path = video_file.get("path")
        if not video_path:
            continue
        video_size = get_video_dimensions(video_path)
        if not video_size:
            continue
        target = min(video_size)
        best = min(resolution_levels, key=lambda level: abs(level - target))
        return f"{best}p"
    return None

def expand_path(path: str) -> str:
    """
    Expand user path with proper Windows support.
    
    On Windows, os.path.expanduser('~') may not work properly in some shells (like PowerShell).
    This function provides a more robust path expansion.
    
    Args:
        path: Path string that may contain ~
        
    Returns:
        Expanded absolute path
    """
    if not path:
        return path
    
    # Try standard expansion first
    expanded = os.path.expanduser(path)
    
    # If expansion didn't work (path still starts with ~), use HOME or USERPROFILE
    if expanded.startswith('~'):
        import platform
        if platform.system() == 'Windows':
            # On Windows, try USERPROFILE first, then HOME
            home = os.environ.get('USERPROFILE') or os.environ.get('HOME')
        else:
            # On Unix-like systems, use HOME
            home = os.environ.get('HOME')
        
        if home:
            # Replace ~ with home directory
            if path == '~':
                expanded = home
            elif path.startswith('~/') or path.startswith('~\\'):
                expanded = os.path.join(home, path[2:])
    
    return expanded