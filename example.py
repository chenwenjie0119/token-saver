"""示例脚本：演示 token-saver 的三种用法。

1) 客户端包装模式：直接使用 TokenSaverClient。
2) 代理服务器模式：先运行 python main.py，再把 base_url 指向本地代理。
3) 统计信息：查看压缩与缓存带来的节省。

运行前请先把 config.example.yaml 复制为 config.yaml 并填入 api_key，
或设置环境变量 OPENAI_API_KEY。
"""

from __future__ import annotations

import json
import os
import tempfile

from token_saver.cache import RequestCache
from token_saver.compressor import compress_text, estimate_tokens
from token_saver.config import load_config
from token_saver.context_manager import trim_context


def demo_compress() -> None:
    """演示文本压缩。"""
    print("=" * 60)
    print("1) 文本压缩")
    sample = """
    你好，   请用 Python 写一个快速排序。

    ```python
    # 注释行，压缩后会被移除
    def quick_sort(arr):
        # 另一条注释
        if len(arr) <= 1:
            return arr
        pivot = arr[0]
        return quick_sort([x for x in arr[1:] if x < pivot]) \\
               + [pivot] + \\
               quick_sort([x for x in arr[1:] if x >= pivot])
    ```

    谢谢，  请保持代码简洁。    
    """
    before = estimate_tokens(sample)
    after = compress_text(sample)
    print(f"压缩前估算 {before} tokens -> 压缩后估算 {estimate_tokens(after)} tokens")
    print(f"节省 {before - estimate_tokens(after)} tokens（{(before - estimate_tokens(after)) / before:.0%}）")
    print("压缩后内容预览:")
    print(after)
    print()


def demo_context() -> None:
    """演示上下文裁剪（滑动窗口 + 历史折叠）。"""
    print("=" * 60)
    print("2) 上下文裁剪")
    messages = [{"role": "system", "content": "你是助手"}]
    for i in range(1, 26):
        messages.append({"role": "user", "content": f"这是第 {i} 轮问题，包含一些详细背景资料和数字 {i * 37}。"})
        messages.append({"role": "assistant", "content": f"第 {i} 轮的回复内容。"})

    original_tokens = sum(estimate_tokens(str(m)) for m in messages)
    trimmed = trim_context(messages, max_messages=10, summary_after=20)
    trimmed_tokens = sum(estimate_tokens(str(m)) for m in trimmed)

    print(f"原始 {len(messages)} 条 -> 裁剪后 {len(trimmed)} 条")
    print(f"估算 tokens: {original_tokens} -> {trimmed_tokens}，节省 {original_tokens - trimmed_tokens}")
    print("裁剪后的首条消息（历史摘要）:")
    print(json.dumps(trimmed[0], ensure_ascii=False)[:200])
    print()


def demo_cache() -> None:
    """演示相似请求缓存。"""
    print("=" * 60)
    print("3) 关键词缓存 / 相似请求去重")
    cache = RequestCache(enabled=True, ttl_seconds=3600, similarity_threshold=0.9)
    payload1 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "介绍一下 token 节省的原理"}]}
    payload2 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "介绍一下 token 节省的原理"}]}
    payload3 = {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "介绍一下 token 节省的原理（追问细节）"}]}

    cache.put(payload1, "缓存中的响应内容")
    print("写入请求1后:")
    print(f"  请求2（完全相同）命中缓存: {cache.get(payload2) is not None}")
    print(f"  请求3（高度相似）命中缓存: {cache.get(payload3) is not None}")
    print()


def demo_client(config_path: str = "") -> None:
    """演示客户端包装模式（需要真实 API Key）。"""
    print("=" * 60)
    print("4) TokenSaverClient（需要 api_key，未配置则跳过）")
    config = load_config(config_path)
    if not config.provider.api_key:
        print("未检测到 api_key，跳过真实调用。可用以下方式配置:")
        print("  - 复制 config.example.yaml 为 config.yaml 并填写 provider.api_key")
        print("  - 设置环境变量 OPENAI_API_KEY")
        print()
        return

    from token_saver.client import TokenSaverClient

    client = TokenSaverClient.from_config(config)
    try:
        resp = client.chat(
            messages=[
                {"role": "user", "content": "用一句话介绍 Python 的 with 语句。"},
            ]
        )
        print(f"模型回复: {resp['content']}")
        print(f"缓存命中: {resp['cached']}")
        print(f"累计请求: {client.stats['requests']}，缓存命中: {client.stats['cache_hits']}，"
              f"预估节省 tokens: {client.stats['saved_tokens']}")
        # 再次调用相同请求，应命中缓存
        resp2 = client.chat(
            messages=[
                {"role": "user", "content": "用一句话介绍 Python 的 with 语句。"},
            ]
        )
        print(f"第二次相同请求缓存命中: {resp2['cached']}")
    except Exception as e:
        print(f"调用失败: {e}")
    finally:
        client.close()
    print()


if __name__ == "__main__":
    demo_compress()
    demo_context()
    demo_cache()
    demo_client()

    print("=" * 60)
    print("提示:")
    print("  - 代理模式: python main.py --config config.example.yaml，然后把 base_url 指向 http://127.0.0.1:8000/v1")
    print("  - 查看代理统计: 浏览器打开 http://127.0.0.1:8000/stats")
