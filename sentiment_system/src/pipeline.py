from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class SentimentSystemConfig:
    base_dir: Path = Path("sentiment_system")

    @property
    def raw_data_path(self) -> Path:
        return self.base_dir / "data/raw/reviews.csv"

    @property
    def clean_data_path(self) -> Path:
        return self.base_dir / "data/processed/reviews_clean.csv"

    @property
    def model_path(self) -> Path:
        return self.base_dir / "models/sentiment_model.joblib"

    @property
    def metrics_path(self) -> Path:
        return self.base_dir / "outputs/metrics.json"

    @property
    def analysis_path(self) -> Path:
        return self.base_dir / "outputs/analysis.json"

    @property
    def outputs_dir(self) -> Path:
        return self.base_dir / "outputs"


class SentimentAnalysisSystem:
    """可复用的情感分析系统封装。"""

    def __init__(self, config: SentimentSystemConfig | None = None):
        self.config = config or SentimentSystemConfig()

    def preprocess_data(self, input_path: Path | None = None, output_path: Path | None = None):
        from .preprocess import preprocess

        return preprocess(input_path or self.config.raw_data_path, output_path or self.config.clean_data_path)

    def train(self, data_path: Path | None = None) -> dict[str, Any]:
        from .train import train_model

        return train_model(
            data_path or self.config.clean_data_path,
            self.config.model_path,
            self.config.metrics_path,
        )

    def analyze(self, data_path: Path | None = None) -> dict[str, Any]:
        from .analyze import run_analysis

        return run_analysis(
            data_path or self.config.clean_data_path,
            self.config.model_path,
            self.config.analysis_path,
        )

    def visualize(self) -> None:
        from .visualize import visualize

        visualize(self.config.metrics_path, self.config.analysis_path, self.config.outputs_dir)

    def predict(self, texts: list[str]) -> list[dict[str, Any]]:
        import joblib
        import pandas as pd

        model = joblib.load(self.config.model_path)
        text_series = pd.Series(texts)
        labels = model.predict(text_series)
        probs = model.predict_proba(text_series)[:, 1]
        return [
            {
                "text": t,
                "label": int(label),
                "positive_prob": float(prob),
            }
            for t, label, prob in zip(texts, labels, probs)
        ]

    def run_all(self) -> dict[str, Any]:
        self.preprocess_data()
        metrics = self.train()
        analysis = self.analyze()
        self.visualize()
        summary = {
            "accuracy": metrics["accuracy"],
            "pred_positive_ratio": analysis["pred_positive_ratio"],
            "metrics_path": str(self.config.metrics_path),
            "analysis_path": str(self.config.analysis_path),
            "outputs_dir": str(self.config.outputs_dir),
        }
        with (self.config.outputs_dir / "run_summary.json").open("w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        return summary
