import asyncio
import os
from typing import Callable, Optional
from urllib.parse import parse_qs, urlparse
import webbrowser

from common.log import logger
from config import conf
from mcp.client.auth.oauth2 import OAuthClientProvider
from mcp.shared.auth import AuthorizationCodeResult, OAuthClientMetadata

from bot.kling.kling_mcp_storage import KlingTokenFileStorage


_SUCCESS_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>可灵 AI 授权成功</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
            background: #f0f2f5;
            display: flex;
            align-items: center;
            justify-content: center;
            height: 100vh;
            margin: 0;
        }
        .container {
            background: #ffffff;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 4px 24px rgba(0,0,0,0.08);
            text-align: center;
            max-width: 420px;
            width: 85%;
        }
        .icon {
            font-size: 54px;
            margin-bottom: 20px;
        }
        h2 {
            margin: 0 0 12px 0;
            color: #1a1a1a;
            font-size: 22px;
        }
        p {
            margin: 0;
            color: #666;
            font-size: 14px;
            line-height: 1.6;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">✅</div>
        <h2>可灵 AI 账号授权成功</h2>
        <p>您已成功连接可灵 MCP 视频服务。<br>您可以关闭本页面，返回聊天窗口继续使用。</p>
    </div>
</body>
</html>
"""

_ERROR_HTML = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>可灵 AI 授权失败</title>
    <style>
        body { font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 100vh; margin: 0; background: #fafafa; }
        .container { background: white; padding: 40px; border-radius: 12px; text-align: center; box-shadow: 0 4px 20px rgba(0,0,0,0.08); }
        h2 { color: #d32f2f; }
        p { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <h2>❌ 授权失败</h2>
        <p>{error_msg}</p>
    </div>
</body>
</html>
"""


class KlingOAuthManager:
    """
    可灵 AI MCP 服务 OAuth 2.0 授权管理器 (RFC 8252 Loopback + DCR)
    """

    def __init__(
        self,
        server_url: str = "https://klingai.com/mcp",
        storage_path: str = "tmp/kling_mcp_auth.json",
        redirect_port: int = 3344,
    ):
        self.server_url = server_url
        self.storage_path = storage_path
        self.redirect_port = int(conf().get("kling_mcp_redirect_port") or redirect_port)
        self.redirect_uri = f"http://127.0.0.1:{self.redirect_port}/callback"

        self.storage = KlingTokenFileStorage(self.storage_path)
        self.latest_auth_url: Optional[str] = None
        self._auth_notify_callback: Optional[Callable[[str], None]] = None
        self._interactive_auth_requested = False
        self._open_browser_on_auth = True

        self._pending_future: Optional[asyncio.Future] = None
        self._server: Optional[asyncio.Server] = None
        self._provider: Optional[OAuthClientProvider] = None

    def has_valid_tokens(self) -> bool:
        """检查本地是否已有有效 Token"""
        return self.storage.has_valid_tokens_sync()

    def set_auth_notify_callback(self, callback: Optional[Callable[[str], None]]):
        """设置接收授权链接的通知回调（如向当前会话下发卡片/链接）"""
        self._auth_notify_callback = callback

    def get_auth_provider(self) -> OAuthClientProvider:
        if self._provider is None:
            metadata = OAuthClientMetadata(
                client_name="BigChao Kling MCP Client",
                redirect_uris=[self.redirect_uri],
                grant_types=["authorization_code", "refresh_token"],
                response_types=["code"],
                scope="generation.create generation.read account.credit.read",
            )
            self._provider = OAuthClientProvider(
                server_url=self.server_url,
                client_metadata=metadata,
                storage=self.storage,
                redirect_handler=self._redirect_handler,
                callback_handler=self._callback_handler,
            )
        return self._provider

    async def _redirect_handler(self, auth_url: str):
        """当 MCP 鉴权流程触发重定向时被调用"""
        if not self._interactive_auth_requested:
            raise RuntimeError("可灵 MCP 尚未授权，请点击飞书视频模型菜单中的 Kling 发起授权。")

        self.latest_auth_url = auth_url
        logger.info(
            f"\n"
            f"======================================================================\n"
            f"[Kling MCP OAuth] 请在浏览器中打开以下链接完成可灵账号授权：\n"
            f"{auth_url}\n"
            f"======================================================================\n"
        )

        if self._auth_notify_callback:
            try:
                self._auth_notify_callback(auth_url)
            except Exception as e:
                logger.warning(f"[KlingOAuth] 触发外部通知回调异常: {e}")

        if self._open_browser_on_auth:
            try:
                webbrowser.open(auth_url)
            except Exception as e:
                logger.warning(f"[KlingOAuth] 打开授权页面失败: {e}")

    async def _callback_handler(self) -> AuthorizationCodeResult:
        """启动本地 loopback http 服务等待可灵重定向返回 code"""
        loop = asyncio.get_running_loop()
        self._pending_future = loop.create_future()

        # 启动本地回调服务
        try:
            self._server = await asyncio.start_server(
                self._handle_http_request,
                host="127.0.0.1",
                port=self.redirect_port,
            )
            logger.info(f"[KlingOAuth] 本地授权回调监听已启动: {self.redirect_uri}")
        except Exception as e:
            logger.error(f"[KlingOAuth] 无法在端口 {self.redirect_port} 启动回调服务器: {e}")
            raise RuntimeError(f"启动可灵 OAuth 本地回调监听失败，端口 {self.redirect_port} 可能被占用: {e}")

        try:
            # 最多等待 5 分钟供用户在网页授权
            result = await asyncio.wait_for(self._pending_future, timeout=300.0)
            return result
        except asyncio.TimeoutError:
            logger.error("[KlingOAuth] 等待用户授权超时（5分钟）")
            raise RuntimeError("可灵账号授权超时，请重试。")
        finally:
            if self._server:
                self._server.close()
                await self._server.wait_closed()
                self._server = None
                logger.info("[KlingOAuth] 本地授权回调监听已关闭")

    async def _handle_http_request(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        """处理来自浏览器的重定向 GET 请求"""
        try:
            data = await reader.read(4096)
            if not data:
                writer.close()
                return

            request_line = data.decode("utf-8", errors="ignore").split("\r\n")[0]
            parts = request_line.split(" ")
            if len(parts) < 2:
                writer.close()
                return

            path = parts[1]
            parsed = urlparse(path)
            query = parse_qs(parsed.query)

            code = query.get("code", [None])[0]
            state = query.get("state", [None])[0]
            iss = query.get("iss", [None])[0]
            error = query.get("error", [None])[0]
            error_description = query.get("error_description", [None])[0]

            if error:
                msg = error_description or error
                logger.error(f"[KlingOAuth] 用户授权拒绝或失败: {msg}")
                body = _ERROR_HTML.format(error_msg=msg).encode("utf-8")
                response = (
                    b"HTTP/1.1 400 Bad Request\r\n"
                    b"Content-Type: text/html; charset=utf-8\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(response)
                await writer.drain()
                writer.close()
                if self._pending_future and not self._pending_future.done():
                    self._pending_future.set_exception(RuntimeError(f"可灵授权失败: {msg}"))
                return

            if code:
                logger.info(f"[KlingOAuth] 成功捕获授权 code: {code[:8]}...")
                body = _SUCCESS_HTML.encode("utf-8")
                response = (
                    b"HTTP/1.1 200 OK\r\n"
                    b"Content-Type: text/html; charset=utf-8\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(response)
                await writer.drain()
                writer.close()

                if self._pending_future and not self._pending_future.done():
                    self._pending_future.set_result(
                        AuthorizationCodeResult(code=code, state=state, iss=iss)
                    )
            else:
                body = b"Invalid request"
                response = (
                    b"HTTP/1.1 400 Bad Request\r\n"
                    b"Content-Length: " + str(len(body)).encode("ascii") + b"\r\n"
                    b"Connection: close\r\n\r\n" + body
                )
                writer.write(response)
                await writer.drain()
                writer.close()

        except Exception as e:
            logger.exception(f"[KlingOAuth] 处理重定向请求异常: {e}")
            writer.close()


# 全局单例管理器
_default_oauth_manager = KlingOAuthManager()

def get_kling_oauth_manager() -> KlingOAuthManager:
    return _default_oauth_manager


def trigger_kling_auth_if_needed(
    notify_channel_fn: Optional[Callable[[str], None]] = None,
    open_browser: bool = True,
    force: bool = False,
) -> bool:
    """
    当用户通过菜单或 #video_model 命令切换至可灵视频模型时调用此函数。
    force=True 时，即使已有 Token 也会通过 MCP 握手验证/刷新。
    返回 True 表示已启动授权握手，False 表示已有流程进行中或无需启动。
    """
    manager = get_kling_oauth_manager()
    if manager._interactive_auth_requested:
        logger.info("[KlingOAuth] 可灵 OAuth 授权流程已在进行中")
        return False
    if manager.has_valid_tokens() and not force:
        logger.info("[KlingOAuth] 可灵 MCP 已授权，无需重新触发鉴权")
        return False

    manager.set_auth_notify_callback(notify_channel_fn)
    manager._interactive_auth_requested = True
    manager._open_browser_on_auth = open_browser

    import threading

    def _worker():
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            from bot.kling.kling_mcp_client import KlingMCPClient
            client = KlingMCPClient(manager.server_url)
            # 访问工具定义将触发 OAuth 握手并持久化保存 Token
            loop.run_until_complete(client.get_tool_definitions())
            logger.info("[KlingOAuth] 可灵 MCP 显式授权成功并已获取工具列表！")
        except Exception as e:
            logger.warning(f"[KlingOAuth] 显式授权握手流程结束或异常: {e}")
        finally:
            manager._interactive_auth_requested = False
            manager._auth_notify_callback = None
            manager._open_browser_on_auth = True
            loop.close()

    thread = threading.Thread(target=_worker, name="kling-oauth-worker", daemon=True)
    thread.start()
    return True
