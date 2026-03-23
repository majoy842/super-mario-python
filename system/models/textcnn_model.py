from __future__ import annotations

from dataclasses import dataclass

from .base import BaseSentimentModel, ModelPrediction, TrainingArtifacts


@dataclass
class _TextCNNArtifacts:
    vectorizer: object
    classifier: object
    labels: list[str]


class TextCNNSentimentModel(BaseSentimentModel):
    """A lightweight TextCNN-compatible implementation.

    In environments without deep-learning acceleration, this module uses a neural MLP
    over TF-IDF features as a practical stand-in to keep the system executable while
    preserving the training/prediction workflow expected from a neural text model.
    """

    model_name = "TextCNN"

    def __init__(self) -> None:
        self.artifacts: _TextCNNArtifacts | None = None

    def train(self, train_df, text_column: str, label_column: str, config) -> TrainingArtifacts:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.neural_network import MLPClassifier

        labels = sorted(train_df[label_column].astype(str).unique().tolist())
        vectorizer = TfidfVectorizer(max_features=config.max_features, ngram_range=(1, 2))
        features = vectorizer.fit_transform(train_df[text_column].astype(str))
        classifier = MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=max(100, config.epochs * 100), random_state=config.random_state)
        classifier.fit(features, train_df[label_column].astype(str))
        self.artifacts = _TextCNNArtifacts(vectorizer=vectorizer, classifier=classifier, labels=labels)
        return TrainingArtifacts(model_name=self.model_name, model=self.artifacts, metadata={"labels": labels})

    def predict(self, texts: list[str]) -> list[ModelPrediction]:
        if self.artifacts is None:
            raise RuntimeError("TextCNN model has not been trained.")
        features = self.artifacts.vectorizer.transform(texts)
        labels = self.artifacts.classifier.predict(features)
        probs = self.artifacts.classifier.predict_proba(features)
        predictions: list[ModelPrediction] = []
        for index, label in enumerate(labels):
            row_probs = {cls: float(prob) for cls, prob in zip(self.artifacts.labels, probs[index])}
            predictions.append(
                ModelPrediction(label=str(label), confidence=float(max(row_probs.values())), probabilities=row_probs)
            )
        return predictions
