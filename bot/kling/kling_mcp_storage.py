import base64
import json
import os
import time
import urllib.parse
import urllib.request
from common.log import logger
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken


class KlingTokenFileStorage:
    """
    实现 MCP SDK TokenStorage 协议，支持持久化存储可灵 OAuth Client 注册信息与 Access/Refresh Token
    """
    def __init__(self, file_path="tmp/kling_mcp_auth.json"):
        self.file_path = file_path
        self._ensure_dir()

    def _ensure_dir(self):
        dirname = os.path.dirname(self.file_path)
        if dirname:
            os.makedirs(dirname, exist_ok=True)

    def _read_data(self) -> dict:
        if not os.path.exists(self.file_path):
            return {}
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"[KlingTokenStorage] 读取 Token 文件失败: {e}")
            return {}

    def _write_data(self, data: dict):
        self._ensure_dir()
        tmp_file = f"{self.file_path}.tmp"
        try:
            with open(tmp_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2, default=str)
            os.replace(tmp_file, self.file_path)
        except Exception as e:
            logger.error(f"[KlingTokenStorage] 保存 Token 文件失败: {e}")
            if os.path.exists(tmp_file):
                try:
                    os.remove(tmp_file)
                except Exception:
                    pass

    @staticmethod
    def extract_token_exp(access_token: str) -> float | None:
        """从 JWT Access Token 中解析 exp 过期时间戳"""
        try:
            parts = access_token.split(".")
            if len(parts) >= 2:
                payload = parts[1]
                payload += "=" * ((4 - len(payload) % 4) % 4)
                data = json.loads(base64.urlsafe_b64decode(payload.encode("utf-8")))
                return float(data.get("exp", 0))
        except Exception:
            pass
        return None

    def refresh_tokens_sync(self) -> bool:
        """使用 refresh_token 向可灵端点主动刷新 access_token"""
        data = self._read_data()
        tokens_data = data.get("tokens") or {}
        client_data = data.get("client_info") or {}
        refresh_token = tokens_data.get("refresh_token")
        client_id = client_data.get("client_id")
        if not refresh_token or not client_id:
            logger.warning("[KlingTokenStorage] 缺少 refresh_token 或 client_id，无法自动刷新")
            return False

        try:
            body = urllib.parse.urlencode({
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": client_id,
            }).encode("utf-8")
            req = urllib.request.Request(
                "https://klingai.com/auth/token",
                data=body,
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                new_tokens = json.loads(resp.read().decode("utf-8"))

            if not new_tokens.get("refresh_token"):
                new_tokens["refresh_token"] = refresh_token
            if not new_tokens.get("scope") and tokens_data.get("scope"):
                new_tokens["scope"] = tokens_data.get("scope")

            data["tokens"] = new_tokens
            self._write_data(data)
            logger.info("[KlingTokenStorage] 可灵 OAuth Token 自动静默刷新成功")
            return True
        except Exception as e:
            logger.warning(f"[KlingTokenStorage] 自动刷新可灵 Token 失败: {e}")
            return False

    def has_valid_tokens_sync(self) -> bool:
        """同步检查是否已存在可灵 OAuth Token，若已过期则尝试自动刷新"""
        data = self._read_data()
        token_data = data.get("tokens")
        if not token_data or not token_data.get("access_token"):
            return False

        exp = self.extract_token_exp(token_data.get("access_token"))
        if exp is not None and time.time() >= exp - 60:
            logger.info(f"[KlingTokenStorage] Token 即将过期或已超期 (exp={exp})，尝试自动刷新...")
            return self.refresh_tokens_sync()
        return True

    async def get_tokens(self) -> OAuthToken | None:
        data = self._read_data()
        token_data = data.get("tokens")
        if not token_data:
            return None

        # 检查是否过期并主动静默刷新
        access_token = token_data.get("access_token")
        if access_token:
            exp = self.extract_token_exp(access_token)
            if exp is not None and time.time() >= exp - 60:
                logger.info(f"[KlingTokenStorage] get_tokens 检测到 Token 已超期，触发自动静默刷新")
                if self.refresh_tokens_sync():
                    data = self._read_data()
                    token_data = data.get("tokens")

        if not token_data:
            return None
        try:
            return OAuthToken.model_validate(token_data)
        except Exception as e:
            logger.warning(f"[KlingTokenStorage] 解析已保存 tokens 失败: {e}")
            return None

    async def set_tokens(self, tokens: OAuthToken) -> None:
        data = self._read_data()
        data["tokens"] = tokens.model_dump(mode="json", exclude_none=True)
        self._write_data(data)
        logger.info("[KlingTokenStorage] 可灵 OAuth Tokens 已成功保存/更新")

    async def get_client_info(self) -> OAuthClientInformationFull | None:
        data = self._read_data()
        client_data = data.get("client_info")
        if not client_data:
            return None
        try:
            return OAuthClientInformationFull.model_validate(client_data)
        except Exception as e:
            logger.warning(f"[KlingTokenStorage] 解析已保存 client_info 失败: {e}")
            return None

    async def set_client_info(self, client_info: OAuthClientInformationFull) -> None:
        data = self._read_data()
        data["client_info"] = client_info.model_dump(mode="json", exclude_none=True)
        self._write_data(data)
        logger.info(f"[KlingTokenStorage] 可灵 DCR 客户端信息已保存: client_id={client_info.client_id}")
