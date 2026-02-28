from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_model(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    df = pd.read_csv(data_path)
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="训练情感分析模型")
    parser.add_argument("--data", default="sentiment_system/data/processed/reviews_clean.csv")
    parser.add_argument("--model", default="sentiment_system/models/sentiment_model.joblib")
    parser.add_argument("--metrics", default="sentiment_system/outputs/metrics.json")
    args = parser.parse_args()

    metrics = train_model(Path(args.data), Path(args.model), Path(args.metrics))
    print(f"训练完成，准确率: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
