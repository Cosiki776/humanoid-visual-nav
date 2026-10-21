"""Local model configuration. Never return secrets to browser status endpoints."""
import json
import os
from pathlib import Path
from urllib.parse import urlparse

CONFIG_PATH = Path(__file__).resolve().parent / "model_config.json"


def load_model_settings():
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {}
    except json.JSONDecodeError:
        raise ValueError("model_config.json 格式错误，请检查双引号和逗号。") from None
    if not isinstance(config, dict):
        raise ValueError("model_config.json 必须是 JSON 对象。")
    result = {}
    for field, variable in (("base_url", "NAV_LLM_BASE_URL"), ("model", "NAV_LLM_MODEL"),
                            ("api_key", "NAV_LLM_API_KEY")):
        value = os.environ.get(variable, config.get(field, ""))
        if not isinstance(value, str):
            raise ValueError(f"模型配置 {field} 必须是字符串。")
        result[field] = value.strip()
    result["base_url"] = result["base_url"].rstrip("/")
    timeout = config.get("timeout_seconds", 120)
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not 1 <= timeout <= 600:
        raise ValueError("timeout_seconds 必须是 1 到 600 之间的数字。")
    result["timeout_seconds"] = timeout
    url = urlparse(result["base_url"])
    if result["base_url"] and (url.scheme not in {"http", "https"} or not url.netloc or url.username or url.password):
        raise ValueError("base_url 必须是有效 HTTP(S) 地址，不能在 URL 中填写密钥。")
    result["configured"] = bool(result["base_url"] and result["model"] and
                                (result["api_key"] or url.hostname in {"localhost", "127.0.0.1", "::1"}))
    return result
