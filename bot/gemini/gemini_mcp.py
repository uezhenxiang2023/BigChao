# encoding:utf-8
import copy
from typing import Any, Dict, List, Optional
from google.genai import types
from common.log import logger


def dereference_schema(schema: Dict[str, Any]) -> Dict[str, Any]:
    """
    递归解析 JSON Schema 中的 $defs / definitions 和 $ref，
    生成符合 Google GenAI SDK (types.Schema) 要求的扁平化 Schema 字典。
    """
    if not isinstance(schema, dict):
        return schema

    schema_copy = copy.deepcopy(schema)
    defs = schema_copy.pop("$defs", {})
    defs.update(schema_copy.pop("definitions", {}))

    # 规范化 $defs 内部的嵌套引用
    def resolve_ref(node: Any, depth: int = 0) -> Any:
        if depth > 20:  # 防止循环引用
            return {}
        if isinstance(node, dict):
            if "$ref" in node:
                ref = node["$ref"]
                ref_key = ref.split("/")[-1]
                if ref_key in defs:
                    resolved = copy.deepcopy(defs[ref_key])
                    # 合并同级属性（如 description）
                    for k, v in node.items():
                        if k != "$ref":
                            resolved[k] = v
                    return resolve_ref(resolved, depth + 1)
                return {k: v for k, v in node.items() if k != "$ref"}
            
            clean_dict = {}
            for k, v in node.items():
                # 过滤可能导致 Gemini 校验失败的元数据字段
                if k in ("title", "default") and isinstance(v, (dict, list)):
                    continue
                clean_dict[k] = resolve_ref(v, depth + 1)
            return clean_dict
        elif isinstance(node, list):
            return [resolve_ref(item, depth + 1) for item in node]
        return node

    return resolve_ref(schema_copy)


def mcp_tool_definitions_to_gemini(tool_definitions: List[Dict[str, Any]]) -> List[types.Tool]:
    """
    将 MCP 标准工具定义列表 (name, description, input_schema)
    转换为 Google GenAI SDK 的 Tool 对象列表。
    """
    declarations = []
    for tool in tool_definitions or []:
        raw_schema = tool.get("input_schema") or {}
        clean_schema = dereference_schema(raw_schema)
        try:
            decl = types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=clean_schema,
            )
            declarations.append(decl)
        except Exception as e:
            logger.warning(f"[GeminiMCP] 转换工具声明失败: {tool.get('name')}, err={e}")

    if not declarations:
        return []
    return [types.Tool(function_declarations=declarations)]


def mcp_tool_definitions_to_interactions_tools(tool_definitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    将 MCP 标准工具定义列表 (name, description, input_schema)
    转换为 Gemini Interactions API 接受的 Tool 格式列表。
    """
    tools = []
    for tool in tool_definitions or []:
        raw_schema = tool.get("input_schema") or {}
        clean_schema = dereference_schema(raw_schema)
        tools.append({
            "type": "function",
            "name": tool["name"],
            "description": tool.get("description", ""),
            "parameters": clean_schema,
        })
    return tools


def format_contents_for_interactions(request_contents: Any) -> Any:
    """
    将各种输入 (str, Part, 文件) 规范化为 Interactions API 接受的输入格式。
    """
    import base64
    if isinstance(request_contents, str):
        return request_contents
    if not isinstance(request_contents, list):
        request_contents = [request_contents]

    blocks = []
    for item in request_contents:
        if isinstance(item, str):
            blocks.append({"type": "text", "text": item})
        elif hasattr(item, "text") and item.text:
            blocks.append({"type": "text", "text": item.text})
        elif hasattr(item, "inline_data") and item.inline_data:
            mime = item.inline_data.mime_type or "application/octet-stream"
            raw = item.inline_data.data
            b64_data = base64.b64encode(raw).decode("utf-8") if isinstance(raw, bytes) else str(raw)
            if mime.startswith("image/"):
                blocks.append({"type": "image", "data": b64_data, "mime_type": mime})
            elif mime.startswith("audio/"):
                blocks.append({"type": "audio", "data": b64_data, "mime_type": mime})
            else:
                blocks.append({"type": "document", "data": b64_data, "mime_type": mime})
        elif hasattr(item, "file_data") and item.file_data:
            blocks.append({"type": "file", "uri": item.file_data.file_uri, "mime_type": item.file_data.mime_type})
        elif isinstance(item, dict):
            blocks.append(item)

    if not blocks:
        return request_contents
    return blocks

