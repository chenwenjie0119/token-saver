"""OpenAI 兼容代理服务器。

将本地代理暴露为 OpenAI 兼容接口（/v1/chat/completions），
客户端只需把 base_url 指向本代理即可透明获得压缩/裁剪/缓存能力，
无需修改业务代码。

启动:
    python main.py --config config.example.yaml
然后任意 OpenAI 客户端:
    client = openai.OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="anything")
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional

from .cache import RequestCache
from .compressor import CompressOptions, compress_messages, count_messages_tokens
from .config import AppConfig
from .context_manager import trim_context


class _Stats:
    def __init__(self) -> None:
        self.requests = 0
        self.cache_hits = 0
        self.saved_tokens = 0
        self.lock = threading.Lock()

    def record(self, cache_hit: bool = False, saved: int = 0) -> None:
        with self.lock:
            self.requests += 1
            if cache_hit:
                self.cache_hits += 1
            self.saved_tokens += max(0, saved)

    def snapshot(self) -> Dict[str, int]:
        with self.lock:
            return {
                "requests": self.requests,
                "cache_hits": self.cache_hits,
                "saved_tokens": self.saved_tokens,
            }


def _forward(base_url: str, api_key: str, path: str, payload: Optional[Dict[str, Any]], timeout: int) -> Any:
    """将请求转发到真实 API。"""
    url = f"{base_url.rstrip('/')}{path}"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    req = urllib.request.Request(url, data=data, headers=headers, method="POST" if payload else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8")
            return json.loads(body)
    except urllib.error.HTTPError as e:
        err_body = e.read().decode("utf-8", errors="ignore")
        return {"error": {"status": e.code, "message": err_body}}


class TokenSaverProxy:
    """代理核心逻辑，独立于 HTTP 层便于测试。"""

    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.cache = RequestCache(
            enabled=config.cache.enabled,
            ttl_seconds=config.cache.ttl_seconds,
            similarity_threshold=config.cache.similarity_threshold,
            max_entries=config.cache.max_entries,
        )
        self.stats = _Stats()

    def handle_chat_completions(self, body: Dict[str, Any]) -> Dict[str, Any]:
        self.stats.record()
        messages = body.get("messages", [])

        # 1) 缓存命中
        cached = self.cache.get(body)
        if cached is not None:
            self.stats.record(cache_hit=True)
            return cached

        # 2) 压缩
        original_tokens = count_messages_tokens(messages)
        processed = messages
        if self.config.compression.enabled:
            opts = CompressOptions(
                remove_blank_lines=self.config.compression.remove_blank_lines,
                remove_code_comments=self.config.compression.remove_code_comments,
                collapse_whitespace=self.config.compression.collapse_whitespace,
                reserved_keywords=self.config.compression.reserved_keywords,
            )
            processed = compress_messages(processed, opts)

        # 3) 上下文裁剪
        if self.config.context.enabled:
            processed = trim_context(
                processed,
                max_messages=self.config.context.max_messages,
                summary_after=self.config.context.summary_after,
                use_llm_summary=False,
            )

        saved = original_tokens - count_messages_tokens(processed)
        self.stats.record(saved=saved)

        # 4) 转发
        body["messages"] = processed
        response = _forward(
            base_url=self.config.provider.base_url,
            api_key=self.config.provider.api_key,
            path="/v1/chat/completions",
            payload=body,
            timeout=self.config.provider.timeout,
        )
        if "error" not in response:
            self.cache.put(body, response)
        return response

    def handle_models(self) -> Dict[str, Any]:
        return _forward(
            base_url=self.config.provider.base_url,
            api_key=self.config.provider.api_key,
            path="/v1/models",
            payload=None,
            timeout=self.config.provider.timeout,
        )

    def stats_endpoint(self) -> Dict[str, Any]:
        data = self.stats.snapshot()
        data["cache"] = self.cache.stats()
        return data


def _make_handler(proxy: TokenSaverProxy) -> Any:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:
            # 精简日志
            print(f"[proxy] {self.address_string()} {fmt % args}")

        def _send_json(self, obj: Dict[str, Any], status: int = 200) -> None:
            body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:  # noqa: N802
            if self.path.startswith("/v1/models"):
                self._send_json(proxy.handle_models())
            elif self.path in ("/stats", "/v1/stats"):
                self._send_json(proxy.stats_endpoint())
            else:
                self._send_json({"error": {"message": f"not found: {self.path}"}}, 404)

        def do_POST(self) -> None:  # noqa: N802
            if not self.path.startswith("/v1/chat/completions"):
                self._send_json({"error": {"message": f"not found: {self.path}"}}, 404)
                return
            length = int(self.headers.get("Content-Length", 0))
            try:
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
            except Exception as e:
                self._send_json({"error": {"message": f"invalid json: {e}"}}, 400)
                return
            response = proxy.handle_chat_completions(body)
            if "error" in response and isinstance(response.get("error"), dict) and response["error"].get("status", 200) >= 400:
                self._send_json(response, int(response["error"]["status"]))
            else:
                self._send_json(response)

    return Handler


def start_proxy(config: AppConfig, host: Optional[str] = None, port: Optional[int] = None) -> ThreadingHTTPServer:
    """启动代理服务器（阻塞）。"""
    host = host or config.proxy.host
    port = port or config.proxy.port
    proxy = TokenSaverProxy(config)
    server = ThreadingHTTPServer((host, port), _make_handler(proxy))
    print(f"token-saver proxy listening on http://{host}:{port}")
    print(f"forwarding to {config.provider.base_url} model={config.provider.model}")
    print(f"stats: http://{host}:{port}/stats")
    server.serve_forever()
    return server
