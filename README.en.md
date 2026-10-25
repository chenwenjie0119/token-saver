# token-saver

A lightweight toolkit to save tokens when calling AI model APIs. Supports **request/response compression**, **context trimming**, and **similar-request caching**, while staying compatible with **OpenAI-compatible APIs** (OpenAI / Qwen / DeepSeek / Zhipu GLM / local vLLM, etc.).

## Features

| Feature | Description | Module |
| --- | --- | --- |
| Text compression | Removes redundant whitespace, blank lines, and code comments without changing semantics | `token_saver/compressor.py` |
| Context trimming | Keeps the most recent N messages via a sliding window; folds older history into a summary | `token_saver/context_manager.py` |
| Request caching | Deduplicates identical/similar requests; cached responses are returned without calling the API | `token_saver/cache.py` |
| Multi-provider API | Any OpenAI-compatible endpoint, switched via the `base_url` config | `token_saver/client.py` / `server.py` |
| Two integration modes | ① wrapped client for direct calls; ② local proxy server with zero code changes | See below |

## Directory Structure

```
token-saver/
├── README.md                  # Chinese documentation
├── README.en.md               # English documentation (this file)
├── LICENSE                    # MIT License
├── requirements.txt           # Dependency list
├── config.example.yaml        # Example config (copy to config.yaml to use)
├── main.py                    # Entry point: starts the local proxy server
├── example.py                 # Example script: demonstrates all saving strategies
└── token_saver/
    ├── __init__.py
    ├── config.py              # Config loading
    ├── compressor.py          # Request/response compression
    ├── context_manager.py     # Context trimming (sliding window + summary)
    ├── cache.py               # Similar-request cache
    ├── client.py              # OpenAI-compatible client wrapper
    └── server.py              # OpenAI-compatible proxy server
```

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

- `openai>=1.0.0` — underlying client for the wrapper mode
- `pyyaml>=6.0` — config parsing
- `requests>=2.31.0` — fallback HTTP client (the current implementation uses the standard library; optional)

## Quick Start

### Mode 1: Wrapped Client (recommended)

```python
from token_saver.client import TokenSaverClient
from token_saver.config import load_config

config = load_config("config.yaml")            # or use the default config
client = TokenSaverClient.from_config(config)

resp = client.chat(
    messages=[
        {"role": "system", "content": "You are a Python assistant."},
        {"role": "user", "content": "Explain the `with` statement in one sentence."},
    ]
)
print(resp["content"])                          # model reply
print(resp["cached"])                           # whether it hit the cache
```

### Mode 2: Proxy Server (zero code changes)

```bash
cp config.example.yaml config.yaml   # fill in provider.api_key
python main.py --config config.yaml
```

Then point any OpenAI client's `base_url` to the local proxy:

```python
from openai import OpenAI

client = OpenAI(
    api_key="anything",                       # the local proxy does not validate keys
    base_url="http://127.0.0.1:8000/v1",      # points to the token-saver proxy
)
resp = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[{"role": "user", "content": "Hello!"}],
)
print(resp.choices[0].message.content)
```

Proxy statistics: open `http://127.0.0.1:8000/stats` in a browser to view request count, cache hits, and estimated token savings.

### Run the Example Script

```bash
python example.py
```

The script demonstrates text compression, context trimming, and similar-request caching step by step, plus a real client call (when `api_key` is configured).

## Configuration

Copy `config.example.yaml` to `config.yaml` (or set the environment variable `TOKEN_SAVER_CONFIG` to point to a config file), then adjust as needed:

```yaml
provider:
  base_url: "https://api.openai.com/v1"   # any OpenAI-compatible service
  api_key: "sk-xxx"
  model: "gpt-4o-mini"

compression:
  enabled: true
  remove_blank_lines: true
  remove_code_comments: true
  collapse_whitespace: true
  reserved_keywords: []                   # keywords to keep verbatim (e.g., code identifiers)

context:
  enabled: true
  max_messages: 10                        # sliding window size
  summary_after: 20                       # fold older messages after this count
  use_llm_summary: false                  # use the model to generate semantic summaries (off by default to save tokens)

cache:
  enabled: true
  ttl_seconds: 3600                       # cache TTL
  similarity_threshold: 0.95              # similarity threshold for cache reuse
  max_entries: 512

proxy:
  host: "127.0.0.1"
  port: 8000
```

## How It Saves Tokens

1. **Compression**: APIs charge per token, yet many requests contain "wasted tokens" such as extra whitespace, blank lines, and code comments. The compression module removes them without changing semantics.
2. **Context trimming**: Multi-turn conversation history grows linearly and is the biggest token consumer. The sliding window keeps only the most recent N messages; older history is folded into a single summary, preserving key information while drastically cutting tokens.
3. **Caching**: Identical (or highly similar) questions should not be paid for twice. The cache module normalizes requests, matches exact duplicates by hash, and reuses responses for similar requests via a similarity algorithm.

## Notes

- Token estimation is approximate (Chinese ~1 token/char, English ~1 token/4 chars); actual counts depend on the model tokenizer.
- The proxy mode and client mode can be enabled simultaneously.

## License

This project is open-sourced under the [MIT License](LICENSE). Copyright (c) 2026 Chen Wenjie (chenwenjie0119).
