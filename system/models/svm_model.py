from __future__ import annotations

from pathlib import Path

from .base import BaseSentimentModel, ModelPrediction, TrainingArtifacts


class SVMSentimentModel(BaseSentimentModel):
    model_name = "SVM"

    def __init__(self) -> None:
        self.pipeline = None
        self.labels: list[str] = []

    def train(self, train_df, text_column: str, label_column: str, config) -> TrainingArtifacts:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import Pipeline
        from sklearn.svm import LinearSVC

        self.labels = sorted(train_df[label_column].astype(str).unique().tolist())
        self.pipeline = Pipeline(
            [
                ("tfidf", TfidfVectorizer(max_features=config.max_features, ngram_range=(1, 2))),
                ("clf", LinearSVC()),
            ]
        )
        self.pipeline.fit(train_df[text_column].astype(str), train_df[label_column].astype(str))
        return TrainingArtifacts(model_name=self.model_name, model=self.pipeline, metadata={"labels": self.labels})

    def predict(self, texts: list[str]) -> list[ModelPrediction]:
        if self.pipeline is None:
            raise RuntimeError("SVM model has not been trained.")
        labels = self.pipeline.predict(texts)
        decision_scores = self.pipeline.decision_function(texts)

        import math

        if len(self.labels) == 2 and getattr(decision_scores, "ndim", 1) == 1:
            decision_scores = [[-score, score] for score in decision_scores]
        else:
            decision_scores = [list(row) for row in decision_scores]

        probs = []
        for row in decision_scores:
            max_score = max(row)
            exp_scores = [math.exp(score - max_score) for score in row]
            total = sum(exp_scores) or 1.0
            probs.append([score / total for score in exp_scores])

        predictions: list[ModelPrediction] = []
        for index, label in enumerate(labels):
            row_probs = {cls: float(prob) for cls, prob in zip(self.labels, probs[index])}
            predictions.append(
                ModelPrediction(label=str(label), confidence=float(max(row_probs.values())), probabilities=row_probs)
            )
        return predictions

    def save(self, save_dir: str):
        import joblib

        if self.pipeline is None:
            raise RuntimeError("SVM model has not been trained.")
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"pipeline": self.pipeline, "labels": self.labels}, target_dir / "model.joblib")

    def load(self, save_dir: str):
        import joblib

        payload = joblib.load(Path(save_dir) / "model.joblib")
        self.pipeline = payload["pipeline"]
        self.labels = payload["labels"]

    def is_ready(self) -> bool:
        return self.pipeline is not None
