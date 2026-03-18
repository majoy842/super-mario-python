"""
模型训练模块（Python 3.11）。

职责：
- 读取清洗后的数据
- 划分训练集 / 测试集
- 使用 TF-IDF + LogisticRegression 训练
- 输出模型文件与评估指标（JSON）

你可以在 PyCharm 中单独运行本文件调试训练过程。
"""

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
    """
    训练情感分类模型并输出指标。

    参数:
        data_path: 预处理后的 CSV 路径
        model_path: 模型保存路径
        metrics_path: 指标 JSON 保存路径

    返回:
        包含 accuracy 与详细分类报告的字典
    """
    df = pd.read_csv(data_path)

    # 划分训练/测试集（按标签分层抽样，保持类别比例）
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["label"],
        test_size=0.25,
        random_state=42,
        stratify=df["label"],
    )

    # 建立 Pipeline，便于统一管理特征与模型
    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1)),
            ("clf", LogisticRegression(max_iter=1000)),
        ]
    )

    # 模型训练
    pipeline.fit(X_train, y_train)

    # 测试集评估
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "report": classification_report(y_test, y_pred, output_dict=True),
    }

    # 保存模型和指标
    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    return metrics


def main() -> None:
    """命令行入口：用于单模块测试。"""
    parser = argparse.ArgumentParser(description="训练情感分析模型")
    parser.add_argument("--data", default="sentiment_system/data/processed/reviews_clean.csv")
    parser.add_argument("--model", default="sentiment_system/models/sentiment_model.joblib")
    parser.add_argument("--metrics", default="sentiment_system/outputs/metrics.json")
    args = parser.parse_args()

    metrics = train_model(Path(args.data), Path(args.model), Path(args.metrics))
    print(f"训练完成，准确率: {metrics['accuracy']:.4f}")


if __name__ == "__main__":
    main()
