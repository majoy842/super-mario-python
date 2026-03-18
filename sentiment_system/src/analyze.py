"""分析模块：用最佳模型做预测并统计关键词。"""

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
    words = []
    for w in jieba.cut(str(text)):
        w = w.strip()
        if w and w not in STOPWORDS and len(w) > 1:
            words.append(w)
    return words


def _positive_prob(model, texts: pd.Series) -> list[float]:
    """优先使用predict_proba提取正向(1)概率；不支持时回退为0/1标签概率。"""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(texts)
        classes = list(model.classes_)
        if 1 in classes:
            return probs[:, classes.index(1)].tolist()
    preds = model.predict(texts)
    return [1.0 if int(x) == 1 else 0.0 for x in preds]


def run_analysis(data_path: Path, model_path: Path, output_path: Path) -> dict:
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    df["pred"] = model.predict(df["text"])
    df["positive_prob"] = _positive_prob(model, df["text"])

    pos_words: list[str] = []
    neg_words: list[str] = []

    for _, row in df.iterrows():
        words = tokenize(row["text"])
        if int(row["pred"]) == 1:
            pos_words.extend(words)
        elif int(row["pred"]) == -1:
            neg_words.extend(words)

    class_dist = {str(int(k)): int(v) for k, v in Counter(df["pred"]).items()}

    result = {
        "sample_count": int(len(df)),
        "pred_positive_ratio": float((df["pred"] == 1).mean()),
        "pred_distribution": class_dist,
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
