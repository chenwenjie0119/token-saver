"""OpenAI 兼容客户端包装。

在调用 openai 库之前自动执行:
1. 文本压缩（compressor）
2. 上下文裁剪（context_manager）
3. 相似请求缓存（cache）

用法:
    from token_saver.client import TokenSaverClient
    client = TokenSaverClient(api_key="sk-xxx", model="gpt-4o-mini")
    resp = client.chat(messages=[{"role": "user", "content": "你好"}])
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .cache import RequestCache
from .compressor import CompressOptions, compress_messages, count_messages_tokens
from .config import AppConfig, CacheConfig, CompressionConfig, ContextConfig
from .context_manager import trim_context


class TokenSaverClient:
    """包装 OpenAI 兼容客户端的节省 token 客户端。"""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.openai.com/v1",
        model: str = "gpt-4o-mini",
        compression: Optional[CompressionConfig] = None,
        context: Optional[ContextConfig] = None,
        cache: Optional[CacheConfig] = None,
        raw_client: Optional[Any] = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.compression_cfg = compression or CompressionConfig()
        self.context_cfg = context or ContextConfig()
        self.cache_cfg = cache or CacheConfig()
        self.cache = RequestCache(
            enabled=self.cache_cfg.enabled,
            ttl_seconds=self.cache_cfg.ttl_seconds,
            similarity_threshold=self.cache_cfg.similarity_threshold,
            max_entries=self.cache_cfg.max_entries,
        )
        # 延迟导入，避免没有安装 openai 时无法导入本模块
        self._raw_client = raw_client
        self._raw_client_owner = raw_client is None
        self.stats = {"requests": 0, "cache_hits": 0, "saved_tokens": 0}

    def _ensure_client(self) -> Any:
        if self._raw_client is None:
            from openai import OpenAI

            self._raw_client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        return self._raw_client

    def _process(
        self, messages: List[Dict[str, Any]]
    ) -> tuple[List[Dict[str, Any]], int]:
        """对 messages 依次应用压缩与上下文裁剪，返回 (处理后的消息, 原始估算token)。"""
        original_tokens = count_messages_tokens(messages)

        processed = messages
        if self.compression_cfg.enabled:
            opts = CompressOptions(
                remove_blank_lines=self.compression_cfg.remove_blank_lines,
                remove_code_comments=self.compression_cfg.remove_code_comments,
                collapse_whitespace=self.compression_cfg.collapse_whitespace,
                reserved_keywords=self.compression_cfg.reserved_keywords,
            )
            processed = compress_messages(processed, opts)

        if self.context_cfg.enabled:
            processed = trim_context(
                processed,
                max_messages=self.context_cfg.max_messages,
                summary_after=self.context_cfg.summary_after,
                use_llm_summary=self.context_cfg.use_llm_summary,
                llm_summarizer=self._summarize_with_llm if self.context_cfg.use_llm_summary else None,
            )

        saved = original_tokens - count_messages_tokens(processed)
        self.stats["saved_tokens"] += max(0, saved)
        return processed, saved

    def _summarize_with_llm(self, old_messages: List[Dict[str, Any]]) -> str:
        """使用模型对旧消息做语义摘要（仅在配置开启时使用）。"""
        from .context_manager import _default_fold

        prompt = (
            "请用简洁的中文概括以下历史对话的核心信息（保留事实、结论、关键数据），"
            "不要超过200字。\n" + _default_fold(old_messages, max_chars=2000)
        )
        try:
            resp = self.chat(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                _skip_cache=True,
            )
            return resp.get("content", "")[:500]
        except Exception:
            return _default_fold(old_messages, max_chars=600)

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: Optional[int] = None,
        top_p: float = 1.0,
        _skip_cache: bool = False,
    ) -> Dict[str, Any]:
        """发送对话请求，返回 {'content': ..., 'raw': ..., 'cached': bool}。"""
        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
        }
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens

        self.stats["requests"] += 1

        # 1) 缓存命中直接返回
        if not _skip_cache:
            cached = self.cache.get(payload)
            if cached is not None:
                self.stats["cache_hits"] += 1
                return {"content": cached, "raw": cached, "cached": True}

        # 2) 压缩与裁剪
        processed_messages, _ = self._process(messages)
        payload["messages"] = processed_messages

        client = self._ensure_client()
        resp = client.chat.completions.create(
            model=payload["model"],
            messages=processed_messages,
            temperature=temperature,
            top_p=top_p,
            **( {"max_tokens": max_tokens} if max_tokens else {} ),
        )
        content = resp.choices[0].message.content if resp.choices else ""

        if not _skip_cache:
            self.cache.put(payload, content)
        return {"content": content, "raw": resp, "cached": False}

    def close(self) -> None:
        """关闭底层客户端连接。"""
        if self._raw_client_owner and self._raw_client is not None:
            try:
                self._raw_client.close()
            except Exception:
                pass

    @classmethod
    def from_config(cls, config: AppConfig) -> "TokenSaverClient":
        """从 AppConfig 构建客户端。"""
        return cls(
            api_key=config.provider.api_key,
            base_url=config.provider.base_url,
            model=config.provider.model,
            compression=config.compression,
            context=config.context,
            cache=config.cache,
        )


def make_client(config: Optional[AppConfig] = None, **kwargs: Any) -> TokenSaverClient:
    """便捷工厂函数。"""
    if config is None:
        from .config import load_config

        config = load_config()
    return TokenSaverClient.from_config(config)
