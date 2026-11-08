import json
import os
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

    def has_valid_tokens_sync(self) -> bool:
        """同步检查是否已存在可灵 OAuth Token"""
        data = self._read_data()
        token_data = data.get("tokens")
        return bool(token_data and token_data.get("access_token"))

    async def get_tokens(self) -> OAuthToken | None:
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
