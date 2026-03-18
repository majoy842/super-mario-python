"""
数据预处理模块（Python 3.11）。

职责：
- 读取原始 CSV（至少包含 text、label 两列）
- 做基础清洗（小写化、去特殊符号、压缩空白）
- 过滤空文本、转换标签类型
- 输出干净数据到 processed 目录

你可以在 PyCharm 中单独运行本文件来验证预处理是否正常。
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    """
    对单条文本做基础清洗。

    参数:
        text: 原始文本

    返回:
        清洗后的文本（保留中文、英文、数字和空格）
    """
    text = str(text).strip().lower()
    # 仅保留中文、英文、数字和空白字符，其余替换为空格
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s]", " ", text)
    # 连续空白压缩为一个空格
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """
    读取输入数据并完成预处理，返回处理后的 DataFrame。

    参数:
        input_path: 原始数据路径
        output_path: 处理后数据输出路径
    """
    df = pd.read_csv(input_path)

    # 校验关键字段
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    # 删除空值并复制，避免链式赋值问题
    df = df.dropna(subset=["text", "label"]).copy()

    # 文本清洗
    df["text"] = df["text"].apply(clean_text)

    # 过滤清洗后为空串的文本
    df = df[df["text"].str.len() > 0]

    # 标签转换为 int，便于模型训练
    df["label"] = df["label"].astype(int)

    # 保存结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    """命令行入口：用于单模块测试。"""
    parser = argparse.ArgumentParser(description="情感分析数据预处理")
    parser.add_argument("--input", default="sentiment_system/data/raw/reviews.csv")
    parser.add_argument("--output", default="sentiment_system/data/processed/reviews_clean.csv")
    args = parser.parse_args()

    df = preprocess(Path(args.input), Path(args.output))
    print(f"预处理完成，共 {len(df)} 条，输出至: {args.output}")


if __name__ == "__main__":
    main()
