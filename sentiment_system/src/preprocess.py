"""预处理模块：清洗评论数据（仅支持 content,sentiment_value 列）。"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]


def clean_text(text: str) -> str:
    """简单清洗：转小写、去特殊符号、去多余空格。"""
    text = str(text).lower().strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """只接受 content,sentiment_value，并转换为 text,label。"""
    if {"content", "sentiment_value"}.issubset(df.columns):
        out = df[["content", "sentiment_value"]].copy()
        out.columns = ["text", "label"]
        return out

    raise ValueError("输入文件必须包含 content,sentiment_value 两列")


def preprocess(input_path: Path, output_path: Path) -> pd.DataFrame:
    """读取 CSV，清洗后保存为新 CSV。"""
    raw_df = pd.read_csv(input_path)
    df = _normalize_columns(raw_df)

    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].apply(clean_text)
    df = df[df["text"] != ""]

    # 兼容 -1/0/1 三分类标注
    df["label"] = pd.to_numeric(df["label"], errors="coerce")
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE_DIR / "data/raw/reviews.csv"))
    parser.add_argument("--output", default=str(BASE_DIR / "data/processed/reviews_clean.csv"))
    args = parser.parse_args()

    result = preprocess(Path(args.input), Path(args.output))
    print(f"预处理完成，样本数：{len(result)}，标签集合：{sorted(result['label'].unique().tolist())}")


if __name__ == "__main__":
    main()
