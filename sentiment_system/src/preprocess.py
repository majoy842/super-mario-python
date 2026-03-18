"""预处理模块：清洗原始评论数据。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    """简单清洗：转小写、去特殊符号、去多余空格。"""
    text = str(text).lower().strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """读取 CSV，清洗后保存为新 CSV。"""
    df = pd.read_csv(input_path)

    if "text" not in df.columns or "label" not in df.columns:
        raise ValueError("输入文件必须包含 text 和 label 两列")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]
    df["label"] = df["label"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sentiment_system/data/raw/reviews.csv")
    parser.add_argument("--output", default="sentiment_system/data/processed/reviews_clean.csv")
    args = parser.parse_args()

    result = preprocess(Path(args.input), Path(args.output))
    print(f"预处理完成，样本数：{len(result)}")


if __name__ == "__main__":
    main()
