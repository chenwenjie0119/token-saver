"""关键词/相似请求缓存模块。

核心思路: 相似或重复的请求（如相同问题、相同 prompt）不必重复调用 API，
直接复用缓存响应即可节省 tokens 与费用。

策略:
- 请求规范化: 去除无关的动态字段（时间戳、随机数等），得到规范文本。
- 精确缓存: 按规范化文本的 SHA-256 哈希命中。
- 相似缓存: 基于 difflib 的相似度匹配，超过阈值即视为相似请求复用。
- TTL: 缓存条目过期自动失效。
"""

from __future__ import annotations

import difflib
import hashlib
import json
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

# 这些字段不参与请求相似度计算（每次请求都可能变化且不影响语义）
_DYNAMIC_FIELDS = {
    "timestamp",
    "time",
    "date",
    "random",
    "seed",
    "request_id",
    "session_id",
    "n",
}


class RequestCache:
    """线程安全的相似请求缓存。"""

    def __init__(
        self,
        enabled: bool = True,
        ttl_seconds: int = 3600,
        similarity_threshold: float = 0.95,
        max_entries: int = 512,
    ) -> None:
        self.enabled = enabled
        self.ttl = ttl_seconds
        self.threshold = similarity_threshold
        self.max_entries = max_entries
        self._store: Dict[str, Tuple[float, Dict[str, Any], str]] = {}
        self._lock = threading.Lock()

    def _normalize_payload(self, payload: Dict[str, Any]) -> str:
        """去掉动态字段并序列化，得到规范文本。"""
        cleaned: Dict[str, Any] = {}
        for k, v in payload.items():
            if isinstance(v, dict):
                cleaned[k] = self._normalize_payload(v)
            elif k in _DYNAMIC_FIELDS:
                continue
            else:
                cleaned[k] = v
        return json.dumps(cleaned, ensure_ascii=False, sort_keys=True)

    def _fingerprint(self, payload: Dict[str, Any]) -> str:
        norm = self._normalize_payload(payload)
        return hashlib.sha256(norm.encode("utf-8")).hexdigest()

    def _is_expired(self, expire_at: float) -> bool:
        return expire_at < time.time()

    def _prune_locked(self) -> None:
        now = time.time()
        expired = [k for k, (exp, _, _) in self._store.items() if exp < now]
        for k in expired:
            del self._store[k]
        while len(self._store) > self.max_entries:
            oldest = min(self._store, key=lambda k: self._store[k][0])
            del self._store[oldest]

    def get(self, payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """查找精确或相似缓存。命中返回缓存响应，否则返回 None。"""
        if not self.enabled or not payload:
            return None

        with self._lock:
            self._prune_locked()

            # 1) 精确匹配
            fp = self._fingerprint(payload)
            if fp in self._store:
                exp, resp, _ = self._store[fp]
                if not self._is_expired(exp):
                    return resp
                del self._store[fp]

            # 2) 相似匹配
            norm = self._normalize_payload(payload)
            for exp, resp, stored_norm in self._store.values():
                if self._is_expired(exp):
                    continue
                ratio = difflib.SequenceMatcher(None, norm, stored_norm).ratio()
                if ratio >= self.threshold:
                    return resp
        return None

    def put(self, payload: Dict[str, Any], response: Dict[str, Any]) -> None:
        """写入缓存。"""
        if not self.enabled or not payload:
            return
        fp = self._fingerprint(payload)
        norm = self._normalize_payload(payload)
        with self._lock:
            self._store[fp] = (time.time() + self.ttl, response, norm)
            self._prune_locked()

    def clear(self) -> None:
        """清空缓存。"""
        with self._lock:
            self._store.clear()

    def stats(self) -> Dict[str, int]:
        """缓存状态统计。"""
        with self._lock:
            return {"entries": len(self._store), "capacity": self.max_entries}
