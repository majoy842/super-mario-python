"""训练模块：训练一个简单的情感分类模型。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


def train_model(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    """读取数据并训练模型，返回准确率。"""
    df = pd.read_csv(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42
    )

    model = Pipeline(
        [
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )
    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    metrics = {"accuracy": float(accuracy_score(y_test, y_pred))}

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="sentiment_system/data/processed/reviews_clean.csv")
    parser.add_argument("--model", default="sentiment_system/models/sentiment_model.joblib")
    parser.add_argument("--metrics", default="sentiment_system/outputs/metrics.json")
    args = parser.parse_args()

    metrics = train_model(Path(args.data), Path(args.model), Path(args.metrics))
    print(f"训练完成，准确率：{metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
