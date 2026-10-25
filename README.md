# token-saver

调用 AI 模型 API 时节省 tokens 的轻量工具集。支持 **请求/响应压缩**、**上下文裁剪**、**相似请求缓存**，并兼容 **OpenAI 兼容接口**（OpenAI / 通义千问 / DeepSeek / 智谱 / 本地 vLLM 等）。

## 功能特性

| 功能 | 说明 | 模块 |
| --- | --- | --- |
| 文本压缩 | 去除多余空白、空行、代码注释，不改变语义 | `token_saver/compressor.py` |
| 上下文裁剪 | 滑动窗口保留最近 N 条消息；超出部分折叠为历史摘要 | `token_saver/context_manager.py` |
| 关键词缓存 | 相同/相似请求自动去重，命中缓存直接返回，不再调用 API | `token_saver/cache.py` |
| 多模型 API | 任何 OpenAI 兼容接口，通过配置 `base_url` 切换 | `token_saver/client.py` / `server.py` |
| 两种接入方式 | ① 包装客户端直接调用；② 本地代理服务器，业务代码零改动 | 见下文用法 |

## 目录结构

```
token-saver/
├── README.md                  # 本文件
├── requirements.txt           # 依赖清单
├── config.example.yaml        # 示例配置（复制为 config.yaml 使用）
├── main.py                    # 入口：启动本地代理服务器
├── example.py                 # 示例脚本：演示三种节省策略与客户端用法
└── token_saver/
    ├── __init__.py
    ├── config.py              # 配置加载
    ├── compressor.py          # 请求/响应压缩
    ├── context_manager.py     # 上下文裁剪（滑动窗口 + 摘要）
    ├── cache.py               # 相似请求缓存
    ├── client.py              # OpenAI 兼容客户端包装
    └── server.py              # OpenAI 兼容代理服务器
```

## 安装依赖

```bash
pip install -r requirements.txt
```

依赖清单：

- `openai>=1.0.0` — 客户端模式底层调用
- `pyyaml>=6.0` — 配置解析
- `requests>=2.31.0` — 备用 HTTP 客户端（当前实现使用标准库，可选）

## 快速开始

### 方式一：客户端包装模式（推荐）

```python
from token_saver.client import TokenSaverClient
from token_saver.config import load_config

config = load_config("config.yaml")            # 或使用默认配置
client = TokenSaverClient.from_config(config)

resp = client.chat(
    messages=[
        {"role": "system", "content": "你是一个 Python 助手。"},
        {"role": "user", "content": "用一句话介绍 with 语句。"},
    ]
)
print(resp["content"])                          # 模型回复
print(resp["cached"])                           # 是否命中缓存
```

### 方式二：代理服务器模式（业务代码零改动）

```bash
cp config.example.yaml config.yaml   # 填入 provider.api_key
python main.py --config config.yaml
```

然后将任意 OpenAI 客户端的 `base_url` 指向本地代理：

```python
from openai import OpenAI

client = OpenAI(
    api_key="anything",                       # 本地代理不校验 key
    base_url="http://127.0.0.1:8000/v1",      # 指向 token-saver 代理
)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "你好"}],
)
print(resp.choices[0].message.content)
```

代理统计：浏览器访问 `http://127.0.0.1:8000/stats` 可查看请求数、缓存命中数与预估节省的 tokens。

### 运行示例脚本

```bash
python example.py
```

脚本会依次演示文本压缩、上下文裁剪、相似请求缓存，以及（在配置好 api_key 时）真实客户端调用。

## 配置说明

复制 `config.example.yaml` 为 `config.yaml`（或设置环境变量 `TOKEN_SAVER_CONFIG` 指向配置文件），按需修改：

```yaml
provider:
  base_url: "https://api.openai.com/v1"   # 兼容任意 OpenAI 风格服务
  api_key: "sk-xxx"
  model: "gpt-4o-mini"

compression:
  enabled: true
  remove_blank_lines: true
  remove_code_comments: true
  collapse_whitespace: true
  reserved_keywords: []                   # 需要保留原样的关键词（如代码标识符）

context:
  enabled: true
  max_messages: 10                        # 滑动窗口大小
  summary_after: 20                       # 超过该条数后对旧消息做折叠
  use_llm_summary: false                  # 是否调用模型生成语义摘要（默认关闭以省 token）

cache:
  enabled: true
  ttl_seconds: 3600                       # 缓存有效期
  similarity_threshold: 0.95              # 相似度阈值，命中则复用
  max_entries: 512

proxy:
  host: "127.0.0.1"
  port: 8000
```

## 节省原理说明

1. **压缩**：API 计费按 token 计算，而大量请求包含多余空白、空行、代码注释等"无效 token"。压缩模块在不改语义的前提下移除它们。
2. **上下文裁剪**：多轮对话历史线性增长是 token 消耗大头。滑动窗口只保留最近的 N 条消息，更早的历史折叠成一条摘要，核心信息不丢、token 大幅下降。
3. **缓存**：相同问题（或高度相似问题）无需重复付费。缓存模块对请求做规范化后按哈希精确匹配，并用相似度算法命中近似请求。

## 注意事项

- token 估算为近似值（中文约 1 token/字、英文约 1 token/4 字符），实际以模型分词器为准。
- 代理模式与客户端模式可同时启用，互不影响。

## 开源协议

本项目基于 [MIT License](LICENSE) 开源，版权所有 (c) 2026 Chen Wenjie (chenwenjie0119)。
