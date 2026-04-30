from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class TrainingArtifacts:
    model_name: str
    model: Any
    metadata: dict = field(default_factory=dict)


@dataclass
class ModelPrediction:
    label: str
    confidence: float
    probabilities: dict[str, float]


class BaseSentimentModel:
    model_name = "base"

    def train(self, train_df, text_column: str, label_column: str, config):
        raise NotImplementedError

    def predict(self, texts: list[str]) -> list[ModelPrediction]:
        raise NotImplementedError

    def save(self, save_dir: str):
        raise NotImplementedError

    def load(self, save_dir: str):
        raise NotImplementedError

    def is_ready(self) -> bool:
        return False
