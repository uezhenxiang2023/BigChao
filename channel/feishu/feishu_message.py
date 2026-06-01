import os, json, re

import lark_oapi as lark
from lark_oapi.api.im.v1 import *

from config import conf
from bridge.context import ContextType
from channel.chat_message import ChatMessage
from common.log import logger
from common.tmp_dir import get_request_dir

def get_message_resource(*, message_id, file_key, type, file_path):
    """
    Use this method to get basic information about resource in IM and prepare it for downloading.
    """
    # 创建client
    client = lark.Client.builder() \
        .app_id(conf().get('feishu_app_id')) \
        .app_secret(conf().get('feishu_app_secret')) \
        .log_level(lark.LogLevel.INFO) \
        .build()

    # 构造请求对象
    request: GetMessageResourceRequest = GetMessageResourceRequest.builder() \
        .message_id(message_id) \
        .file_key(file_key) \
        .type(type) \
        .build()

    # 发起请求
    response: GetMessageResourceResponse = client.im.v1.message_resource.get(request)

    # 处理失败返回
    if not response.success():
        lark.logger.error(
            f"client.im.v1.message_resource.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return

    # 处理业务结果
    f = open(file_path, "wb")
    f.write(response.file.read())
    f.close()



def get_message_detail(message_id):
    client = lark.Client.builder() \
        .app_id(conf().get('feishu_app_id')) \
        .app_secret(conf().get('feishu_app_secret')) \
        .log_level(lark.LogLevel.INFO) \
        .build()

    request: GetMessageRequest = GetMessageRequest.builder() \
        .message_id(message_id) \
        .build()

    response: GetMessageResponse = client.im.v1.message.get(request)
    if not response.success():
        lark.logger.error(
            f"client.im.v1.message.get failed, code: {response.code}, msg: {response.msg}, log_id: {response.get_log_id()}, resp: \n{json.dumps(json.loads(response.raw.content), indent=4, ensure_ascii=False)}")
        return None

    items = response.data.items or []
    return items[0] if items else None


def download_referenced_image(parent_message_id, file_path):
    message = get_message_detail(parent_message_id)
    if not message or not message.body or not message.body.content:
        return None

    try:
        content = json.loads(message.body.content)
    except Exception as e:
        logger.warning(f"[Lark] failed to parse referenced message body: {e}")
        return None

    if message.msg_type == 'image':
        image_key = content.get('image_key')
        if not image_key:
            return None
        get_message_resource(message_id=parent_message_id, file_key=image_key, type='image', file_path=file_path)
        return file_path

    if message.msg_type == 'post':
        contents = content.get('content', [])
        for items in contents:
            for item in items:
                if item.get('tag') == 'img' and item.get('image_key'):
                    get_message_resource(message_id=parent_message_id, file_key=item['image_key'], type='image', file_path=file_path)
                    return file_path
    return None


def download_referenced_video(parent_message_id, user_dir):
    message = get_message_detail(parent_message_id)
    if not message or not message.body or not message.body.content:
        return None

    try:
        content = json.loads(message.body.content)
    except Exception as e:
        logger.warning(f"[Lark] failed to parse referenced message body: {e}")
        return None

    file_key = None
    file_name = None
    resource_type = None

    if message.msg_type == 'media':
        file_key = content.get("file_key")
        file_name = content.get("file_name") or f"quoted_{parent_message_id}.mp4"
        resource_type = "media"
    elif message.msg_type == 'file':
        file_key = content.get("file_key")
        file_name = content.get("file_name")
        if not file_name or os.path.splitext(file_name)[1].lstrip(".").lower() not in {"mp4", "mov", "avi", "mkv", "webm", "m4v"}:
            return None
        resource_type = "file"
    else:
        return None

    if not file_key or not file_name:
        return None

    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, f"quoted_{parent_message_id}_{os.path.basename(file_name)}")
    get_message_resource(message_id=parent_message_id, file_key=file_key, type=resource_type, file_path=file_path)
    return file_path


def download_referenced_file(parent_message_id, user_dir):
    message = get_message_detail(parent_message_id)
    if not message or not message.body or not message.body.content or message.msg_type != 'file':
        return None

    try:
        content = json.loads(message.body.content)
    except Exception as e:
        logger.warning(f"[Lark] failed to parse referenced file body: {e}")
        return None

    file_key = content.get("file_key")
    file_name = content.get("file_name")
    if not file_key or not file_name:
        return None

    suffix = os.path.splitext(file_name)[1].lstrip(".").lower()
    if suffix not in {"pdf", "doc", "docx", "plain", "txt"}:
        logger.info(f"[Lark] skip quoted file download, unsupported suffix={suffix}, parent_id={parent_message_id}")
        return None

    os.makedirs(user_dir, exist_ok=True)
    file_path = os.path.join(user_dir, f"quoted_{parent_message_id}_{os.path.basename(file_name)}")
    logger.info(
        f"[Lark] downloading quoted file, parent_id={parent_message_id}, "
        f"file_name={file_name}, suffix={suffix}, path={file_path}"
    )
    get_message_resource(message_id=parent_message_id, file_key=file_key, type='file', file_path=file_path)
    return file_path

def get_file_name(file):
    file_path = file.file_path
    file_name_index = file_path.rfind("/")
    file_name = file_path[file_name_index+1:]
    return file_name


class FeishuMessage(ChatMessage):
    def __init__(self, event: dict, is_group=False):
        super().__init__(event)
        self.msg_id = event.message.message_id
        self.create_time = event.message.create_time
        self.is_group = is_group
        self.from_user_id = event.sender.sender_id.open_id
        self.to_user_id = event.message.chat_id
        self.other_user_id = self.from_user_id
        self.user_dir = get_request_dir("feishu", self.from_user_id)
        self.parent_id = event.message.parent_id

        if event.message.message_type == 'text':
            self.ctype = ContextType.TEXT
            self.content = json.loads(event.message.content)["text"]
        elif event.message.message_type == 'audio':
            self.ctype = ContextType.VOICE
            self.content = self.user_dir + os.path.basename(event.message["FileName"])  # content直接存临时目录路径
            self._prepare_fn = lambda: event.message.download(self.content)
        elif event.message.message_type == 'image':
            self.ctype = ContextType.IMAGE
            image_key = json.loads(event.message.content)["image_key"]
            self.content = self.user_dir + image_key  # content直接存临时目录路径
            self._prepare_fn = lambda: get_message_resource(message_id=self.msg_id, file_key=image_key, type=event.message.message_type, file_path=self.content)
        elif event.message.message_type == 'media':
            self.ctype = ContextType.VIDEO
            media_content = json.loads(event.message.content)
            file_name = media_content.get("file_name") or media_content.get("file_key") or f"{self.msg_id}.mp4"
            self.content = self.user_dir + file_name
            file_key = media_content.get("file_key")
            if file_key:
                self._prepare_fn = lambda: get_message_resource(
                    message_id=self.msg_id,
                    file_key=file_key,
                    type=event.message.message_type,
                    file_path=self.content
                )
            else:
                self._prepare_fn = lambda: event.message.download(self.content)
        elif event.message.message_type == 'file':
            self.ctype = ContextType.FILE
            file_key = json.loads(event.message.content)["file_key"]
            file_name = json.loads(event.message.content)["file_name"]
            self.content = self.user_dir + file_name  # content直接存临时目录路径
            self._prepare_fn = lambda: get_message_resource(message_id=self.msg_id, file_key=file_key, type=event.message.message_type, file_path=self.content)
        elif event.message.message_type == 'post':
            contents = json.loads(event.message.content)['content']
            for items in contents:
                for item in items:
                    if item['tag'] == 'img':
                        self.ctype = ContextType.IMAGE
                        image_key = item["image_key"]
                        self.content = self.user_dir + image_key  # content直接存临时目录路径
                        self._prepare_fn = lambda: get_message_resource(message_id=self.msg_id, file_key=image_key, type='image', file_path=self.content)
        else:
            raise NotImplementedError("Unsupported message type: Type:{} MsgType:{}".format(event.message["Type"], event.message["MsgType"]))


    def get_quoted_image_path(self):
        if not self.parent_id:
            return None
        os.makedirs(self.user_dir, exist_ok=True)
        file_path = os.path.join(self.user_dir, f"quoted_{self.parent_id}.png")
        return download_referenced_image(self.parent_id, file_path)

    def get_quoted_video_path(self):
        if not self.parent_id:
            return None
        return download_referenced_video(self.parent_id, self.user_dir)

    def get_quoted_file_path(self):
        if not self.parent_id:
            return None
        return download_referenced_file(self.parent_id, self.user_dir)
