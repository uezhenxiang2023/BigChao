try:
    import openai
except Exception:
    openai = None

try:
    import requests
except Exception:
    requests = None


OPENAI_STATUS_MESSAGES = {
    400: "OpenAI 请求参数有误，请检查模型、提示词、尺寸或参考素材参数后重试。",
    401: "OpenAI API 鉴权失败，请检查 API Key 配置。",
    403: "OpenAI 访问被拒绝，请检查项目权限、模型权限或账号状态。",
    404: "OpenAI 资源不存在，请检查模型 ID、接口地址或会话引用。",
    409: "OpenAI 请求冲突，请稍后重试。",
    422: "OpenAI 请求无法处理，请检查参考素材是否有效、可读取，或调整参数组合。",
    429: "OpenAI 当前请求过多或额度受限，请稍后重试。",
    500: "OpenAI 服务内部错误，请稍后重试。",
    502: "OpenAI 上游服务暂时不可用，请稍后重试。",
    503: "OpenAI 服务繁忙或不可用，请稍后重试。",
    504: "OpenAI 服务响应超时，请稍后重试。",
}

OPENAI_ERROR_CODE_MESSAGES = {
    "invalid_api_key": "OpenAI API Key 无效，请检查 API Key 配置。",
    "insufficient_quota": "OpenAI 账户额度不足，请充值或调整配额后再试。",
    "rate_limit_exceeded": "OpenAI 当前请求过多，已触发限流，请稍后重试。",
    "context_length_exceeded": "上下文长度超过模型限制，请缩短输入后重试。",
    "model_not_found": "模型不存在或当前项目无权访问，请检查模型配置。",
    "content_policy_violation": "请求内容触发安全策略，请调整提示词或参考素材后重试。",
    "image_too_large": "参考图超过 OpenAI 限制，请压缩图片后重试。",
    "invalid_image": "参考图无效或无法读取，请更换图片后重试。",
}


OPENAI_IMAGE_MODERATION_STAGE_MESSAGES = {
    "input": "输入内容或参考图",
    "output": "生成结果",
    "unknown": "未知阶段",
}

OPENAI_IMAGE_MODERATION_CATEGORY_MESSAGES = {
    "harassment": "骚扰或攻击性内容",
    "self-harm": "自伤/自残相关内容",
    "sexual": "性相关内容",
    "violence": "暴力相关内容",
}


def format_openai_error(error, model, *, service_name="OpenAI"):
    status_code = extract_openai_status_code(error)
    error_code = extract_openai_error_code(error)
    message = extract_openai_error_message(error)
    request_id = extract_openai_request_id(error)

    if error_code == "moderation_blocked":
        return format_openai_image_moderation_error(error, model, request_id=request_id)

    request_text = f" request_id={request_id}" if request_id else ""
    code_text = f"，错误码={error_code}" if error_code else ""

    if is_openai_timeout_error(error):
        base_message = f"{service_name} 请求超时，请稍后重试。"
    elif is_openai_connection_error(error):
        base_message = f"无法连接{service_name}服务，请检查网络或稍后重试。"
    else:
        base_message = OPENAI_ERROR_CODE_MESSAGES.get(error_code)
        if not base_message:
            status_message = OPENAI_STATUS_MESSAGES.get(
                status_code,
                f"{service_name}请求失败(status={status_code})，请稍后重试。",
            )
            base_message = _prefix_service(status_message, service_name)

    if message:
        base_message = f"{base_message}（{message}）"
    return f"[{str(model).upper()}] {base_message}{code_text}{request_text}"


def format_openai_image_moderation_error(error, model, *, request_id=None):
    details = extract_openai_moderation_details(error)
    stage = details.get("moderation_stage")
    categories = details.get("categories") or []

    stage_text = OPENAI_IMAGE_MODERATION_STAGE_MESSAGES.get(stage, stage or "未知阶段")
    category_texts = [
        OPENAI_IMAGE_MODERATION_CATEGORY_MESSAGES.get(category, str(category))
        for category in categories
        if category
    ]
    reason_text = "、".join(category_texts) if category_texts else "安全策略相关内容"

    if stage == "output":
        hint = "建议调整提示词后重新生成。"
    else:
        hint = "建议调整提示词或更换参考图后重试。"

    request_text = f"\nrequest_id={request_id}" if request_id else ""
    return (
        f"[{str(model).upper()}] 图片生成被安全系统拦截，未能完成本次请求。\n"
        f"拦截阶段：{stage_text}\n"
        f"可能原因：{reason_text}\n"
        f"{hint}"
        f"{request_text}"
    )


def is_openai_error(error):
    return is_openai_sdk_error(error) or is_openai_http_error(error)


def is_openai_sdk_error(error):
    openai_error = _get_openai_class("OpenAIError")
    if openai_error is not None:
        return isinstance(error, openai_error)
    return _is_openai_exception(
        error,
        (
            "APIError",
            "APIStatusError",
            "APIConnectionError",
            "APITimeoutError",
            "BadRequestError",
            "AuthenticationError",
            "PermissionDeniedError",
            "NotFoundError",
            "ConflictError",
            "UnprocessableEntityError",
            "RateLimitError",
            "InternalServerError",
        ),
    )


def is_openai_http_error(error):
    if requests is None:
        return False
    return isinstance(error, (requests.HTTPError, requests.Timeout, requests.ConnectionError))


def is_openai_rate_limit_error(error):
    return _is_openai_exception(error, ("RateLimitError",))


def is_openai_timeout_error(error):
    return _is_openai_exception(error, ("APITimeoutError", "Timeout")) or (
        requests is not None and isinstance(error, requests.Timeout)
    )


def is_openai_connection_error(error):
    return _is_openai_exception(error, ("APIConnectionError",)) or (
        requests is not None and isinstance(error, requests.ConnectionError)
    )


def is_openai_api_error(error):
    return _is_openai_exception(error, ("APIError", "APIStatusError"))


def extract_openai_status_code(error):
    status_code = getattr(error, "status_code", None)
    response = getattr(error, "response", None)
    if status_code is None and response is not None:
        status_code = getattr(response, "status_code", None)
    return status_code or "unknown"


def extract_openai_error_payload(error):
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        return body.get("error") if isinstance(body.get("error"), dict) else body

    response = getattr(error, "response", None)
    if response is not None:
        try:
            data = response.json()
            if isinstance(data, dict):
                return data.get("error") if isinstance(data.get("error"), dict) else data
        except Exception:
            pass
    return {}


def extract_openai_error_code(error):
    payload = extract_openai_error_payload(error)
    code = getattr(error, "code", None) or payload.get("code") or payload.get("type")
    return str(code) if code else None


def extract_openai_moderation_details(error):
    payload = extract_openai_error_payload(error)
    details = payload.get("moderation_details") if isinstance(payload, dict) else None
    return details if isinstance(details, dict) else {}


def extract_openai_error_message(error):
    payload = extract_openai_error_payload(error)
    message = (
        payload.get("message")
        or payload.get("detail")
        or getattr(error, "message", None)
    )
    if message:
        return str(message)

    response = getattr(error, "response", None)
    text = getattr(response, "text", None) if response is not None else None
    if text:
        return str(text).strip()

    text = str(error).strip()
    return text or None


def extract_openai_request_id(error):
    request_id = getattr(error, "request_id", None)
    if request_id:
        return request_id

    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if not headers:
        return None
    return (
        headers.get("x-request-id")
        or headers.get("X-Request-Id")
        or headers.get("openai-request-id")
        or headers.get("Openai-Request-Id")
    )


def _get_openai_class(name):
    if openai is None:
        return None
    return getattr(openai, name, None)


def _is_openai_exception(error, class_names):
    for class_name in class_names:
        cls = _get_openai_class(class_name)
        if cls is not None and isinstance(error, cls):
            return True
    return False


def _prefix_service(message, service_name):
    if message.startswith("OpenAI") or message.startswith(service_name):
        return message
    return f"{service_name}{message}"
