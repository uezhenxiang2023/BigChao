# encoding:utf-8

import os
import signal
import sys

from channel import channel_factory
from common import const
from config import load_config
from plugins import *

SUPPORTED_PLUGIN_CHANNELS = [
    "wx",
    "wxy",
    "terminal",
    "wechatmp",
    "web",
    "wechatmp_service",
    "wechatcom_app",
    "wework",
    const.WEIXIN,
    const.FEISHU,
    const.DINGTALK,
    const.TELEGRAM,
]

_BOOTSTRAPPED = False
_BOOTSTRAP_CHANNEL_NAME = None
_PLUGINS_LOADED = False


def is_feishu_webhook_debug_enabled():
    return os.environ.get("DEBUG_FEISHU_WEBHOOK", "").lower() in {"1", "true", "yes", "on"}


def sigterm_handler_wrap(_signo):
    old_handler = signal.getsignal(_signo)

    def func(_signo, _stack_frame):
        logger.info("signal {} received, exiting...".format(_signo))
        conf().save_user_datas()
        if callable(old_handler):  #  check old_handler
            return old_handler(_signo, _stack_frame)
        sys.exit(0)

    signal.signal(_signo, func)


def resolve_channel_name():
    channel_name = conf().get("channel_type", "wx")
    if "--cmd" in sys.argv:
        channel_name = "terminal"
    return channel_name


def bootstrap(register_signals=False):
    global _BOOTSTRAPPED, _BOOTSTRAP_CHANNEL_NAME, _PLUGINS_LOADED
    if not _BOOTSTRAPPED:
        load_config()
        _BOOTSTRAP_CHANNEL_NAME = resolve_channel_name()
        if _BOOTSTRAP_CHANNEL_NAME == "wxy":
            os.environ["WECHATY_LOG"] = "warn"
            # os.environ['WECHATY_PUPPET_SERVICE_ENDPOINT'] = '127.0.0.1:9001'
        _BOOTSTRAPPED = True

    if register_signals:
        sigterm_handler_wrap(signal.SIGINT)
        sigterm_handler_wrap(signal.SIGTERM)

    if _BOOTSTRAP_CHANNEL_NAME in SUPPORTED_PLUGIN_CHANNELS and not _PLUGINS_LOADED:
        PluginManager().load_plugins()
        _PLUGINS_LOADED = True

    return _BOOTSTRAP_CHANNEL_NAME


def is_feishu_webhook_mode(channel_name=None):
    current_channel = channel_name or bootstrap()
    return current_channel == const.FEISHU and conf().get("feishu_websocket") is False


def create_channel(channel_name=None):
    current_channel = channel_name or bootstrap()
    return channel_factory.create_channel(current_channel)


def create_wsgi_application():
    if not is_feishu_webhook_mode():
        return None
    channel = create_channel(const.FEISHU)
    return channel.app


application = create_wsgi_application()


def run():
    try:
        channel_name = bootstrap(register_signals=True)
        channel = create_channel(channel_name)
        if is_feishu_webhook_mode(channel_name) and not is_feishu_webhook_debug_enabled():
            logger.info("Feishu webhook mode detected, please start with gunicorn app:application")
            logger.info("For local breakpoint debugging, set DEBUG_FEISHU_WEBHOOK=1 and run python app.py")
            return

        channel.startup()
    except Exception as e:
        logger.error("App startup failed!")
        logger.exception(e)


if __name__ == "__main__":
    run()
