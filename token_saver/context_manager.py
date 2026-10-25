"""上下文裁剪模块。

核心思路: 对话越长 token 消耗越大。通过"滑动窗口 + 旧消息摘要/压缩"在
保持核心信息的前提下大幅降低历史消息占用的 token 数。

策略（按顺序尝试）:
1. 滑动窗口: 只保留最近 N 条消息 + system 指令。
2. 历史折叠: 将窗口外的旧消息合并为一条摘要消息。
3. 可选 LLM 摘要: 调用模型对旧消息做语义摘要（需额外少量调用，默认关闭）。
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .compressor import compress_messages, estimate_tokens

# 标记来源，便于用户识别摘要内容
_SUMMARY_ROLE = "system"
_SUMMARY_PREFIX = "[历史消息摘要] "


def _default_fold(old_messages: List[Dict[str, Any]], max_chars: int = 600) -> str:
    """默认的历史折叠策略: 抽取关键内容拼接，超出长度直接截断。"""
    parts = []
    for m in old_messages:
        content = m.get("content", "")
        if isinstance(content, list):
            content = " ".join(
                p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
            )
        if not isinstance(content, str):
            continue
        content = content.strip()
        if not content:
            continue
        if len(content) > 120:
            content = content[:120] + "…"
        parts.append(f"{m.get('role', 'user')}: {content}")
    summary = " | ".join(parts)
    return summary[:max_chars]


def trim_context(
    messages: List[Dict[str, Any]],
    max_messages: int = 10,
    summary_after: int = 20,
    use_llm_summary: bool = False,
    llm_summarizer: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
    fold_fn: Optional[Callable[[List[Dict[str, Any]]], str]] = None,
) -> List[Dict[str, Any]]:
    """裁剪上下文。

    Args:
        messages: OpenAI 格式的 messages 列表。
        max_messages: 滑动窗口保留的最大消息条数（含 system 与摘要条）。
        summary_after: 消息总数超过该值时触发折叠。
        use_llm_summary: 是否调用模型生成语义摘要。
        llm_summarizer: 摘要函数，输入旧消息列表，输出摘要文本。
        fold_fn: 自定义折叠函数，输入旧消息列表，输出摘要文本。

    Returns:
        裁剪后的 messages 列表。
    """
    if not messages or len(messages) <= max_messages:
        return messages

    system_msgs = [m for m in messages if m.get("role") == "system"]
    rest = [m for m in messages if m.get("role") != "system"]

    if len(messages) <= summary_after:
        # 仅滑动窗口：保留最近的 max_messages 条非 system 消息
        kept = rest[-max_messages:]
        return system_msgs + kept

    # 超过 summary_after: 折叠窗口外旧消息
    overflow = rest[:-max_messages]
    kept = rest[-max_messages:]

    if use_llm_summary and llm_summarizer is not None:
        summary_text = llm_summarizer(overflow)
    else:
        summary_text = (fold_fn or _default_fold)(overflow)

    summary_msg: Dict[str, Any] = {
        "role": _SUMMARY_ROLE,
        "content": _SUMMARY_PREFIX + summary_text,
    }
    return system_msgs + [summary_msg] + kept


def report_savings(
    original: List[Dict[str, Any]], trimmed: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """统计裁剪前后 token 与条数变化，便于向用户汇报。"""
    before = estimate_tokens(str(original))
    after = estimate_tokens(str(trimmed))
    return {
        "original_count": len(original),
        "trimmed_count": len(trimmed),
        "original_tokens": before,
        "trimmed_tokens": after,
        "saved_tokens": max(0, before - after),
    }
