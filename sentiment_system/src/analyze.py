from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jieba
import joblib
import pandas as pd


STOPWORDS = {"的", "了", "很", "也", "就", "都", "和", "太", "不", "有", "在", "是", "我", "还", "让"}


def tokenize(text: str) -> list[str]:
    return [w.strip() for w in jieba.cut(text) if w.strip() and w not in STOPWORDS and len(w.strip()) > 1]


def run_analysis(data_path: Path, model_path: Path, output_path: Path) -> dict:
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    preds = model.predict(df["text"])
    probs = model.predict_proba(df["text"])[:, 1]
    df["pred"] = preds
    df["positive_prob"] = probs

    positive_tokens: list[str] = []
    negative_tokens: list[str] = []

    for _, row in df.iterrows():
        words = tokenize(row["text"])
        if row["pred"] == 1:
            positive_tokens.extend(words)
        else:
            negative_tokens.extend(words)

    result = {
        "sample_count": len(df),
        "pred_positive_ratio": float((df["pred"] == 1).mean()),
        "top_positive_words": Counter(positive_tokens).most_common(20),
        "top_negative_words": Counter(negative_tokens).most_common(20),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.parent / "predictions.csv", index=False)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="情感分析挖掘")
    parser.add_argument("--data", default="sentiment_system/data/processed/reviews_clean.csv")
    parser.add_argument("--model", default="sentiment_system/models/sentiment_model.joblib")
    parser.add_argument("--output", default="sentiment_system/outputs/analysis.json")
    args = parser.parse_args()

    result = run_analysis(Path(args.data), Path(args.model), Path(args.output))
    print("分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
