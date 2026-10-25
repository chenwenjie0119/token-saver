"""请求/响应文本压缩模块。

核心思路: 在不改变语义的前提下移除冗余字符，从而减少发送给模型的 token 数。
支持:
- 去除多余空白（连续空格 / 空行折叠）
- 去除代码块中的注释与空行
- 中英文 token 粗略估算
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# 常见编程语言单行注释标记
_SINGLE_LINE_COMMENT = re.compile(r"^\s*(#|//|--|')\s?.*$", re.MULTILINE)
# 折叠连续空行
_MULTI_BLANK_LINES = re.compile(r"\n\s*\n(\s*\n)+")
# 折叠连续空白字符
_MULTI_SPACES = re.compile(r"[ \t]{2,}")
# 折叠行尾空白
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)


@dataclass
class CompressOptions:
    """压缩选项。"""

    remove_blank_lines: bool = True          # 折叠连续空行
    remove_code_comments: bool = True        # 删除代码块内单行注释
    collapse_whitespace: bool = True         # 折叠连续空格
    min_text_length: int = 8                 # 低于该长度的文本不压缩（避免过度压缩短句）
    reserved_keywords: List[str] = field(default_factory=list)  # 保留原样的关键词


def estimate_tokens(text: str) -> int:
    """粗略估算文本 token 数。

    中文约 1 token/字，英文约 1 token/4 字符，按字符加权估算。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return int(cjk + other / 4) + 1


def compress_text(text: str, opts: Optional[CompressOptions] = None) -> str:
    """压缩单段文本，返回压缩后的文本。"""
    if not text or len(text) < (opts.min_text_length if opts else 8):
        return text
    opts = opts or CompressOptions()
    result = text

    if opts.remove_blank_lines:
        result = _MULTI_BLANK_LINES.sub("\n\n", result)

    if opts.remove_code_comments:
        result = _SINGLE_LINE_COMMENT.sub("", result)

    if opts.collapse_whitespace:
        result = _TRAILING_SPACE.sub("", result)
        result = _MULTI_SPACES.sub(" ", result)

    result = result.strip("\n")
    # 保留关键词原文
    if opts.reserved_keywords:
        placeholders = {}
        for i, kw in enumerate(opts.reserved_keywords):
            if kw and kw in result:
                ph = f"__KW{i}__"
                result = result.replace(kw, ph)
                placeholders[ph] = kw
        # 折叠后再还原
        result = _MULTI_BLANK_LINES.sub("\n\n", result)
        result = _TRAILING_SPACE.sub("", result)
        result = _MULTI_SPACES.sub(" ", result)
        for ph, kw in placeholders.items():
            result = result.replace(ph, kw)

    return result.strip()


def compress_message(
    message: Dict[str, Any], opts: Optional[CompressOptions] = None
) -> Dict[str, Any]:
    """压缩单条 chat message 的 content 字段。"""
    if not isinstance(message, dict):
        return message
    content = message.get("content")
    if isinstance(content, str):
        message = dict(message)
        message["content"] = compress_text(content, opts)
    elif isinstance(content, list):
        # 多模态 content（文本部分压缩，图片等保持不变）
        new_parts = []
        for part in content:
            if (
                isinstance(part, dict)
                and part.get("type") == "text"
                and isinstance(part.get("text"), str)
            ):
                part = dict(part)
                part["text"] = compress_text(part["text"], opts)
            new_parts.append(part)
        message = dict(message)
        message["content"] = new_parts
    return message


def compress_messages(
    messages: List[Dict[str, Any]], opts: Optional[CompressOptions] = None
) -> List[Dict[str, Any]]:
    """压缩整组 messages。"""
    return [compress_message(m, opts) for m in messages]


def count_messages_tokens(messages: List[Dict[str, Any]]) -> int:
    """统计整组 messages 的估算 token 数。"""
    total = 0
    for m in messages:
        total += estimate_tokens(str(m.get("role", "")))
        content = m.get("content", "")
        if isinstance(content, str):
            total += estimate_tokens(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    total += estimate_tokens(part["text"])
    return total
