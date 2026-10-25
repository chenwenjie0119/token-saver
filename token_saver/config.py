"""配置加载模块。"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class ProviderConfig:
    base_url: str = "https://api.openai.com/v1"
    api_key: str = ""
    model: str = "gpt-4o-mini"
    timeout: int = 60


@dataclass
class CompressionConfig:
    enabled: bool = True
    remove_blank_lines: bool = True
    remove_code_comments: bool = True
    collapse_whitespace: bool = True
    reserved_keywords: List[str] = field(default_factory=list)


@dataclass
class ContextConfig:
    enabled: bool = True
    max_messages: int = 10
    summary_after: int = 20
    use_llm_summary: bool = False


@dataclass
class CacheConfig:
    enabled: bool = True
    ttl_seconds: int = 3600
    similarity_threshold: float = 0.95
    max_entries: int = 512


@dataclass
class ProxyConfig:
    host: str = "127.0.0.1"
    port: int = 8000


@dataclass
class AppConfig:
    provider: ProviderConfig = field(default_factory=ProviderConfig)
    compression: CompressionConfig = field(default_factory=CompressionConfig)
    context: ContextConfig = field(default_factory=ContextConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    proxy: ProxyConfig = field(default_factory=ProxyConfig)


def _section(data: Dict[str, Any], key: str) -> Dict[str, Any]:
    value = data.get(key, {})
    return value if isinstance(value, dict) else {}


def load_config(path: Optional[str] = None) -> AppConfig:
    """加载 YAML 配置；未指定路径时优先读环境变量 TOKEN_SAVER_CONFIG，
    再回退到当前目录 config.yaml。"""
    if path is None:
        path = os.environ.get("TOKEN_SAVER_CONFIG", "")
    if not path:
        candidates = ["config.yaml", "config.yml"]
        path = next((c for c in candidates if os.path.exists(c)), "")
    if not path or not os.path.exists(path):
        return AppConfig()

    if yaml is None:
        raise RuntimeError("缺少 pyyaml 依赖，请先执行: pip install pyyaml")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    provider = _section(data, "provider")
    compression = _section(data, "compression")
    context = _section(data, "context")
    cache = _section(data, "cache")
    proxy = _section(data, "proxy")

    return AppConfig(
        provider=ProviderConfig(
            base_url=provider.get("base_url", "https://api.openai.com/v1"),
            api_key=provider.get("api_key", os.environ.get("OPENAI_API_KEY", "")),
            model=provider.get("model", "gpt-4o-mini"),
            timeout=int(provider.get("timeout", 60)),
        ),
        compression=CompressionConfig(
            enabled=bool(compression.get("enabled", True)),
            remove_blank_lines=bool(compression.get("remove_blank_lines", True)),
            remove_code_comments=bool(compression.get("remove_code_comments", True)),
            collapse_whitespace=bool(compression.get("collapse_whitespace", True)),
            reserved_keywords=list(compression.get("reserved_keywords", []) or []),
        ),
        context=ContextConfig(
            enabled=bool(context.get("enabled", True)),
            max_messages=int(context.get("max_messages", 10)),
            summary_after=int(context.get("summary_after", 20)),
            use_llm_summary=bool(context.get("use_llm_summary", False)),
        ),
        cache=CacheConfig(
            enabled=bool(cache.get("enabled", True)),
            ttl_seconds=int(cache.get("ttl_seconds", 3600)),
            similarity_threshold=float(cache.get("similarity_threshold", 0.95)),
            max_entries=int(cache.get("max_entries", 512)),
        ),
        proxy=ProxyConfig(
            host=proxy.get("host", "127.0.0.1"),
            port=int(proxy.get("port", 8000)),
        ),
    )
