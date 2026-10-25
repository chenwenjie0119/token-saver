"""token-saver: 调用 AI 模型 API 时节省 tokens 的工具集。

包含:
- compressor: 请求/响应文本压缩
- context_manager: 上下文裁剪（滑动窗口 + 摘要）
- cache: 相似请求缓存去重
- client: OpenAI 兼容客户端包装
- server: OpenAI 兼容代理服务器
"""

__version__ = "0.1.0"
