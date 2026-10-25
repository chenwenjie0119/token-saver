"""token-saver 入口。

用法:
    python main.py                        # 使用 config.yaml（不存在则用默认配置）
    python main.py --config config.example.yaml
"""

from __future__ import annotations

import argparse
import sys

from token_saver.config import load_config
from token_saver.server import start_proxy


def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(description="token-saver OpenAI 兼容代理")
    parser.add_argument("--config", "-c", default="", help="配置文件路径（YAML）")
    parser.add_argument("--host", default=None, help="监听地址，默认取配置")
    parser.add_argument("--port", type=int, default=None, help="监听端口，默认取配置")
    args = parser.parse_args(argv)

    config = load_config(args.config)

    if not config.provider.api_key:
        print(
            "警告: 未配置 api_key。请设置配置文件中的 provider.api_key "
            "或环境变量 OPENAI_API_KEY，否则代理将无法转发请求。",
            file=sys.stderr,
        )

    try:
        start_proxy(config, host=args.host, port=args.port)
    except KeyboardInterrupt:
        print("\n已停止。")
    except OSError as e:
        print(f"启动失败: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
