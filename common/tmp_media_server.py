import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote, urlparse

from common.log import logger
from common.tmp_dir import TmpDir


_SERVER = None
_SERVER_THREAD = None


def start_tmp_media_server(port):
    global _SERVER, _SERVER_THREAD
    if _SERVER is not None:
        return True

    try:
        server = _ReusableThreadingHTTPServer(("0.0.0.0", int(port)), _TmpMediaRequestHandler)
    except Exception as e:
        logger.warning(f"[TmpMediaServer] failed to start on port {port}: {e}")
        return False

    thread = threading.Thread(target=server.serve_forever, name="tmp-media-server", daemon=True)
    thread.start()
    _SERVER = server
    _SERVER_THREAD = thread
    logger.info(f"[TmpMediaServer] serving tmp media on port {port}")
    return True


class _ReusableThreadingHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _TmpMediaRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self._serve_file(send_body=True)

    def do_HEAD(self):
        self._serve_file(send_body=False)

    def log_message(self, format, *args):
        logger.debug("[TmpMediaServer] " + format, *args)

    def _serve_file(self, send_body):
        parsed = urlparse(self.path)
        prefix = "/tmp_media/"
        if not parsed.path.startswith(prefix):
            self.send_error(404)
            return

        relative_path = unquote(parsed.path[len(prefix):])
        tmp_root = os.path.abspath(TmpDir().path())
        target_path = os.path.abspath(os.path.join(tmp_root, relative_path))
        if not target_path.startswith(tmp_root + os.sep) and target_path != tmp_root:
            self.send_error(403)
            return
        if not os.path.isfile(target_path):
            self.send_error(404)
            return

        content_type = mimetypes.guess_type(target_path)[0] or "application/octet-stream"
        file_size = os.path.getsize(target_path)
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_size))
        self.end_headers()
        if not send_body:
            return

        with open(target_path, "rb") as file:
            while True:
                chunk = file.read(1024 * 1024)
                if not chunk:
                    break
                self.wfile.write(chunk)
