"""训练模块：训练多种模型并做性能对比。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, f1_score, log_loss, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parents[1]
EPOCHS = 8


def _train_normal_model(model: Pipeline, x_train, x_test, y_train, y_test) -> tuple[Pipeline, dict]:
    """训练普通模型并返回评估结果。"""
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    result = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, average="macro")),
    }
    return model, result


def _train_sgd_with_history(x_train, x_test, y_train, y_test) -> tuple[Pipeline, dict, dict]:
    """训练 SGD 模型，并记录每一轮的 loss/accuracy 曲线。"""
    vectorizer = TfidfVectorizer()
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    clf = SGDClassifier(loss="log_loss", max_iter=1, tol=None, random_state=42)
    classes = sorted(pd.Series(y_train).unique().tolist())

    loss_history: list[float] = []
    accuracy_history: list[float] = []

    for _ in range(EPOCHS):
        clf.partial_fit(x_train_vec, y_train, classes=classes)
        y_pred = clf.predict(x_test_vec)
        probs = clf.predict_proba(x_test_vec)

        accuracy_history.append(float(accuracy_score(y_test, y_pred)))
        loss_history.append(float(log_loss(y_test, probs, labels=classes)))

    result = {
        "accuracy": float(accuracy_score(y_test, clf.predict(x_test_vec))),
        "precision": float(precision_score(y_test, clf.predict(x_test_vec), average="macro", zero_division=0)),
        "recall": float(recall_score(y_test, clf.predict(x_test_vec), average="macro", zero_division=0)),
        "f1_score": float(f1_score(y_test, clf.predict(x_test_vec), average="macro")),
    }

    history = {
        "epochs": list(range(1, EPOCHS + 1)),
        "loss": loss_history,
        "accuracy": accuracy_history,
    }

    model = Pipeline([
        ("tfidf", vectorizer),
        ("clf", clf),
    ])
    return model, result, history


def train_model(data_path: Path, model_path: Path, metrics_path: Path) -> dict:
    """读取数据，训练多个模型，保存最佳模型。"""
    df = pd.read_csv(data_path)

    x_train, x_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.25, random_state=42, stratify=df["label"]
    )

    candidates: list[tuple[str, object]] = [
        (
            "logistic_regression",
            Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", LogisticRegression(max_iter=1200)),
            ]),
        ),
        (
            "multinomial_nb",
            Pipeline([
                ("tfidf", TfidfVectorizer()),
                ("clf", MultinomialNB()),
            ]),
        ),
    ]

    results: list[dict] = []
    best_name = ""
    best_model = None
    best_f1 = -1.0
    training_history: dict | None = None

    for name, model in candidates:
        trained_model, scores = _train_normal_model(model, x_train, x_test, y_train, y_test)
        results.append({"model": name, **scores})

        if scores["f1_score"] > best_f1:
            best_f1 = scores["f1_score"]
            best_name = name
            best_model = trained_model

    sgd_model, sgd_scores, sgd_history = _train_sgd_with_history(x_train, x_test, y_train, y_test)
    results.append({"model": "sgd_log_loss", **sgd_scores})

    if sgd_scores["f1_score"] > best_f1:
        best_f1 = sgd_scores["f1_score"]
        best_name = "sgd_log_loss"
        best_model = sgd_model

    training_history = {
        "model": "sgd_log_loss",
        **sgd_history,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, model_path)

    metrics = {
        "best_model": best_name,
        "results": sorted(results, key=lambda x: x["f1_score"], reverse=True),
        "training_history": training_history,
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
    print(
        f"训练完成，最佳模型：{metrics['best_model']}，"
        f"accuracy={top['accuracy']:.4f}，precision={top['precision']:.4f}，"
        f"recall={top['recall']:.4f}，f1_score={top['f1_score']:.4f}"
    )


if __name__ == "__main__":
    main()
