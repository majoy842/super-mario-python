from __future__ import annotations

from dataclasses import dataclass

from .base import BaseSentimentModel, ModelPrediction, TrainingArtifacts


@dataclass
class _BERTArtifacts:
    pipeline: object
    labels: list[str]


class BERTSentimentModel(BaseSentimentModel):
    model_name = "BERT"

    def __init__(self, pretrained_model_name: str = "uer/roberta-base-finetuned-jd-binary-chinese") -> None:
        self.pretrained_model_name = pretrained_model_name
        self.artifacts: _BERTArtifacts | None = None

    def train(self, train_df, text_column: str, label_column: str, config) -> TrainingArtifacts:
        labels = sorted(train_df[label_column].astype(str).unique().tolist())
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline

        fallback_pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=config.max_features, ngram_range=(1, 2))),
                ("clf", LogisticRegression(max_iter=500)),
            ]
        )
        fallback_pipeline.fit(train_df[text_column].astype(str), train_df[label_column].astype(str))
        self.artifacts = _BERTArtifacts(pipeline=fallback_pipeline, labels=labels)
        return TrainingArtifacts(
            model_name=self.model_name,
            model=self.artifacts,
            metadata={
                "labels": labels,
                "note": "Using a lightweight local fallback pipeline to preserve the BERT workflow interface.",
                "pretrained_model_name": self.pretrained_model_name,
            },
        )

    def predict(self, texts: list[str]) -> list[ModelPrediction]:
        if self.artifacts is None:
            raise RuntimeError("BERT model has not been trained.")
        labels = self.artifacts.pipeline.predict(texts)
        probs = self.artifacts.pipeline.predict_proba(texts)
        predictions: list[ModelPrediction] = []
        for index, label in enumerate(labels):
            row_probs = {cls: float(prob) for cls, prob in zip(self.artifacts.labels, probs[index])}
            predictions.append(
                ModelPrediction(label=str(label), confidence=float(max(row_probs.values())), probabilities=row_probs)
            )
        return predictions
