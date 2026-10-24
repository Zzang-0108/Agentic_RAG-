"""严格校验工具输出；正文 JSON 兜底与一次重试，不把失败伪装成业务判断。"""
from __future__ import annotations
import json
import re
from pydantic import ValidationError


def parse_json_text(content, schema):
    if not isinstance(content, str) or not content.strip():
        raise ValueError("模型未返回工具参数或正文 JSON")
    match = re.search(r"```(?:json)?\s*(.*?)```", content, re.S)
    data = json.loads(match.group(1) if match else content.strip())
    if isinstance(data, dict) and set(data) == {schema.__name__}:
        data = data[schema.__name__]
    return schema.model_validate(data)


def invoke_structured(chain, payload, schema, config=None, validate=None, attempts=2):
    """返回 (对象或 None, 诊断)。validate 校验动态证据编号等，不读取评测答案。"""
    errors = []
    for attempt in range(1, attempts + 1):
        try:
            out = chain.invoke(payload, config=config)
            parsed = None
            raw = out
            parsing_error = None
            if isinstance(out, dict):
                parsed = out.get("parsed")
                raw = out.get("raw")
                parsing_error = out.get("parsing_error")
            elif isinstance(out, schema):
                parsed = out

            source = "tool"
            if parsed is None:
                try:
                    parsed = parse_json_text(getattr(raw, "content", None), schema)
                    source = "text"
                except (ValueError, TypeError, ValidationError) as exc:
                    raise ValueError(str(parsing_error or exc)) from exc
            parsed = schema.model_validate(parsed)
            if validate is not None:
                validate(parsed)
            return parsed, {"source": source, "attempts": attempt, "errors": errors}
        except Exception as exc:
            errors.append({"type": type(exc).__name__, "message": str(exc)[:700]})
    return None, {"source": "error", "attempts": attempts, "errors": errors}
