from __future__ import annotations

import argparse
import json
from pathlib import Path

from system import SentimentAnalysisSystem, SystemConfig
from system.config import AnalysisConfig, DataConfig, ExportConfig, PreprocessConfig, TrainingConfig


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="耳机评论情感分析系统")
    parser.add_argument("--data", required=True, help="评论数据集路径，支持 csv/xlsx/json")
    parser.add_argument("--id-column", default="content_id", help="评论ID列名")
    parser.add_argument("--text-column", default="content", help="评论文本列名")
    parser.add_argument("--label-column", default="sentiment_value", help="情感标签列名")
    parser.add_argument("--aspect-column", default="subject", help="属性列名")
    parser.add_argument("--sentiment-word-column", default="sentiment_word", help="情感词列名")
    parser.add_argument("--stopwords", default=None, help="停用词文件路径")
    parser.add_argument("--custom-dict", default=None, help="自定义词典路径")
    parser.add_argument("--output-dir", default="system/outputs", help="结果输出目录")
    parser.add_argument("--single-text", default=None, help="单条评论分析内容")
    return parser


def build_config(args: argparse.Namespace) -> SystemConfig:
    return SystemConfig(
        data=DataConfig(
            data_path=Path(args.data),
            id_column=args.id_column,
            text_column=args.text_column,
            label_column=args.label_column,
            aspect_column=args.aspect_column,
            sentiment_word_column=args.sentiment_word_column,
        ),
        preprocess=PreprocessConfig(
            stopwords_path=Path(args.stopwords) if args.stopwords else None,
            custom_dict_path=Path(args.custom_dict) if args.custom_dict else None,
        ),
        training=TrainingConfig(),
        analysis=AnalysisConfig(),
        export=ExportConfig(output_dir=Path(args.output_dir)),
    )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config = build_config(args)
    system = SentimentAnalysisSystem(config)
    summary = system.run()
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if args.single_text:
        single_result = system.analyze_single(args.single_text, model_name=summary["best_model"])
        print(json.dumps(single_result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
