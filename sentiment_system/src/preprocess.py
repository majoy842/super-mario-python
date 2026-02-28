from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd


def clean_text(text: str) -> str:
    text = str(text).strip().lower()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required_cols = {"text", "label"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"缺少必要字段: {missing}")

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"].str.len() > 0]
    df["label"] = df["label"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="情感分析数据预处理")
    parser.add_argument("--input", default="sentiment_system/data/raw/reviews.csv")
    parser.add_argument("--output", default="sentiment_system/data/processed/reviews_clean.csv")
    args = parser.parse_args()

    df = preprocess(Path(args.input), Path(args.output))
    print(f"预处理完成，共 {len(df)} 条数据，输出至: {args.output}")


if __name__ == "__main__":
    main()
