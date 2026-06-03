import os
import pathlib

from config import conf
from common.log import logger


class TmpDir(object):
    """A temporary directory that is deleted when the object is destroyed."""

    tmpFilePath = pathlib.Path("./tmp/")

    def __init__(self):
        pathExists = os.path.exists(self.tmpFilePath)
        if not pathExists:
            os.makedirs(self.tmpFilePath)

    def path(self):
        return str(self.tmpFilePath) + "/"


def get_request_dir(channel_type, user_id):
    """Return tmp/{channel_type}/{user_id}/request/ and ensure it exists."""
    request_dir = os.path.join(TmpDir().path(), str(channel_type), str(user_id), "request")
    os.makedirs(request_dir, exist_ok=True)
    return request_dir + os.sep


def get_response_dir(channel_type, user_id):
    """Return tmp/{channel_type}/{user_id}/response/ and ensure it exists."""
    response_dir = os.path.join(TmpDir().path(), str(channel_type), str(user_id), "response")
    os.makedirs(response_dir, exist_ok=True)
    return response_dir + os.sep
    
def create_user_dir(path):
    """创建用户私有目录"""
    user_path = pathlib.Path(path)
    os.makedirs(user_path)
    info = 'Dir is created:' + path
    logger.info(f'[{conf().get("channel_type").upper()}] {info}')
