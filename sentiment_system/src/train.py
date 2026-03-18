"""训练模块：训练多种模型并做性能对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]


def _build_models() -> dict[str, Pipeline]:
    """准备多个可对比模型。"""
    return {
        "logistic_regression": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", LogisticRegression(max_iter=1200)),
        ]),
        "multinomial_nb": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", MultinomialNB()),
        ]),
        "sgd_log_loss": Pipeline([
            ("tfidf", TfidfVectorizer()),
            ("clf", SGDClassifier(loss="log_loss", max_iter=1000, random_state=42)),
        ]),
    }


def train_model(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    """读取数据，训练多个模型，保存最佳模型。"""
    df = pd.read_csv(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    results: list[dict] = []
    best_name = ""
    best_model: Pipeline | None = None
    best_f1 = -1.0

    for name, model in _build_models().items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)

        acc = float(accuracy_score(y_test, y_pred))
        f1 = float(f1_score(y_test, y_pred, average="macro"))

        results.append({"model": name, "accuracy": acc, "macro_f1": f1})

        if f1 > best_f1:
            best_f1 = f1
            best_name = name
            best_model = model

    assert best_model is not None

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    metrics = {
        "best_model": best_name,
        "results": sorted(results, key=lambda x: x["macro_f1"], reverse=True),
    }

    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(BASE_DIR / "data/processed/reviews_clean.csv"))
    parser.add_argument("--model", default=str(BASE_DIR / "models/sentiment_model.joblib"))
    parser.add_argument("--metrics", default=str(BASE_DIR / "outputs/metrics.json"))
    args = parser.parse_args()

    metrics = train_model(Path(args.data), Path(args.model), Path(args.metrics))
    top = metrics["results"][0]
    print(f"训练完成，最佳模型：{metrics['best_model']}，macro_f1={top['macro_f1']:.4f}")


if __name__ == "__main__":
    main()
