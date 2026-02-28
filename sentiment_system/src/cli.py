from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import SentimentAnalysisSystem, SentimentSystemConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="情感分析系统 CLI")
    parser.add_argument(
        "command",
        choices=["preprocess", "train", "analyze", "visualize", "run-all", "predict"],
        help="要执行的命令",
    )
    parser.add_argument("--base-dir", default="sentiment_system", help="系统根目录")
    parser.add_argument("--texts", nargs="*", default=[], help="predict 命令下要预测的文本")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    system = SentimentAnalysisSystem(SentimentSystemConfig(base_dir=Path(args.base_dir)))

    if args.command == "preprocess":
        df = system.preprocess_data()
        print(f"预处理完成，共 {len(df)} 条")
    elif args.command == "train":
        metrics = system.train()
        print(json.dumps(metrics, ensure_ascii=False, indent=2))
    elif args.command == "analyze":
        analysis = system.analyze()
        print(json.dumps(analysis, ensure_ascii=False, indent=2))
    elif args.command == "visualize":
        system.visualize()
        print("可视化完成")
    elif args.command == "run-all":
        summary = system.run_all()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    elif args.command == "predict":
        if not args.texts:
            raise SystemExit("predict 命令需要通过 --texts 提供至少一条文本")
        result = system.predict(args.texts)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
