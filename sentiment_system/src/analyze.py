"""分析模块：用训练好的模型做预测并统计关键词。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jieba
import joblib
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
STOPWORDS = {"的", "了", "很", "也", "就", "都", "和", "太", "不"}


def tokenize(text: str) -> list[str]:
    """中文分词并去掉常见停用词。"""
    words = []
    for w in jieba.cut(str(text)):
        w = w.strip()
        if w and w not in STOPWORDS and len(w) > 1:
            words.append(w)
    return words


def run_analysis(data_path: Path, model_path: Path, output_path: Path) -> dict:
    """预测情感并输出统计结果。"""
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    df["pred"] = model.predict(df["text"])
    df["positive_prob"] = model.predict_proba(df["text"])[:, 1]

    pos_words: list[str] = []
    neg_words: list[str] = []

    for _, row in df.iterrows():
        words = tokenize(row["text"])
        if int(row["pred"]) == 1:
            pos_words.extend(words)
        else:
            neg_words.extend(words)

    result = {
        "sample_count": int(len(df)),
        "pred_positive_ratio": float((df["pred"] == 1).mean()),
        "top_positive_words": Counter(pos_words).most_common(10),
        "top_negative_words": Counter(neg_words).most_common(10),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.parent / "predictions.csv", index=False)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE_DIR / "data/processed/reviews_clean.csv"))
    parser.add_argument("--model", default=str(BASE_DIR / "models/sentiment_model.joblib"))
    parser.add_argument("--output", default=str(BASE_DIR / "outputs/analysis.json"))
    args = parser.parse_args()

    result = run_analysis(Path(args.data), Path(args.model), Path(args.output))
    print("分析完成：")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
