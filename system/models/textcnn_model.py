from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BaseSentimentModel, ModelPrediction, TrainingArtifacts


@dataclass
class _TextCNNArtifacts:
    model: object
    vocab: dict[str, int]
    labels: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    max_length: int
    device: str


class _TextCNNClassifier:
    @staticmethod
    def build(vocab_size, embedding_dim, num_classes, num_filters, kernel_sizes, dropout):
        import torch
        import torch.nn as nn

        class TextCNNClassifier(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
                self.convs = nn.ModuleList([nn.Conv2d(1, num_filters, (kernel_size, embedding_dim)) for kernel_size in kernel_sizes])
                self.dropout = nn.Dropout(dropout)
                self.fc = nn.Linear(num_filters * len(kernel_sizes), num_classes)

            def forward(self, input_ids):
                embedded = self.embedding(input_ids).unsqueeze(1)
                conv_outputs = [torch.relu(conv(embedded)).squeeze(3) for conv in self.convs]
                pooled_outputs = [torch.max(output, dim=2).values for output in conv_outputs]
                features = torch.cat(pooled_outputs, dim=1)
                return self.fc(self.dropout(features))

        return TextCNNClassifier()


class TextCNNSentimentModel(BaseSentimentModel):
    model_name = "TextCNN"

    def __init__(self) -> None:
        self.artifacts: _TextCNNArtifacts | None = None
        self._arch_config = {
            "embedding_dim": 128,
            "num_filters": 128,
            "kernel_sizes": [3, 4, 5],
            "dropout": 0.5,
        }

    def _resolve_device(self, config):
        import torch

        if getattr(config, "device", "auto") == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(config.device)

    def _build_vocab(self, texts: list[str], max_vocab_size: int) -> dict[str, int]:
        from collections import Counter

        counter = Counter()
        for text in texts:
            counter.update(str(text).split())
        vocab = {"<PAD>": 0, "<UNK>": 1}
        for token, _ in counter.most_common(max_vocab_size - 2):
            vocab[token] = len(vocab)
        return vocab

    def _encode_texts(self, texts: list[str], vocab: dict[str, int], max_length: int):
        encoded = []
        for text in texts:
            token_ids = [vocab.get(token, vocab["<UNK>"]) for token in str(text).split()[:max_length]]
            if len(token_ids) < max_length:
                token_ids.extend([vocab["<PAD>"]] * (max_length - len(token_ids)))
            encoded.append(token_ids)
        return encoded

    def train(self, train_df, text_column: str, label_column: str, config) -> TrainingArtifacts:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

        labels = sorted(train_df[label_column].astype(str).unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        vocab = self._build_vocab(train_df[text_column].astype(str).tolist(), config.textcnn_vocab_size)
        encoded_texts = self._encode_texts(train_df[text_column].astype(str).tolist(), vocab, config.max_length)
        encoded_labels = [label_to_id[label] for label in train_df[label_column].astype(str).tolist()]

        device = self._resolve_device(config)
        model = _TextCNNClassifier.build(
            vocab_size=len(vocab),
            embedding_dim=config.textcnn_embedding_dim,
            num_classes=len(labels),
            num_filters=config.textcnn_num_filters,
            kernel_sizes=config.textcnn_kernel_sizes,
            dropout=config.textcnn_dropout,
        ).to(device)
        self._arch_config = {
            "embedding_dim": config.textcnn_embedding_dim,
            "num_filters": config.textcnn_num_filters,
            "kernel_sizes": list(config.textcnn_kernel_sizes),
            "dropout": config.textcnn_dropout,
        }

        dataset = TensorDataset(torch.tensor(encoded_texts, dtype=torch.long), torch.tensor(encoded_labels, dtype=torch.long))
        sample_weights = None
        class_weights = None
        if getattr(config, "use_class_weight", True):
            from collections import Counter

            counts = Counter(encoded_labels)
            class_weights = torch.tensor([len(encoded_labels) / (len(labels) * counts[idx]) for idx in range(len(labels))], dtype=torch.float)
            sample_weights = torch.tensor([class_weights[label].item() for label in encoded_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True) if sample_weights is not None else None
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            pin_memory=(device.type == "cuda"),
            num_workers=config.num_workers,
        )
        optimizer = torch.optim.Adam(model.parameters(), lr=max(config.learning_rate, 1e-3))
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and getattr(config, "use_amp", True)))

        model.train()
        for _ in range(config.epochs):
            for input_ids, labels_tensor in loader:
                input_ids = input_ids.to(device, non_blocking=(device.type == "cuda"))
                labels_tensor = labels_tensor.to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda" and getattr(config, "use_amp", True))):
                    logits = model(input_ids)
                    loss = criterion(logits, labels_tensor)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

        self.artifacts = _TextCNNArtifacts(
            model=model,
            vocab=vocab,
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            max_length=config.max_length,
            device=str(device),
        )
        return TrainingArtifacts(model_name=self.model_name, model=self.artifacts, metadata={"labels": labels, "device": str(device)})

    def predict(self, texts: list[str]) -> list[ModelPrediction]:
        import torch

        if self.artifacts is None:
            raise RuntimeError("TextCNN model has not been trained.")
        device = torch.device(self.artifacts.device)
        encoded_texts = self._encode_texts(texts, self.artifacts.vocab, self.artifacts.max_length)
        input_tensor = torch.tensor(encoded_texts, dtype=torch.long).to(device)
        self.artifacts.model.eval()
        with torch.no_grad():
            logits = self.artifacts.model(input_tensor)
            probabilities = torch.softmax(logits, dim=1).cpu().tolist()

        predictions: list[ModelPrediction] = []
        for row_probs in probabilities:
            label_idx = max(range(len(row_probs)), key=lambda idx: row_probs[idx])
            label = self.artifacts.id_to_label[label_idx]
            probability_map = {self.artifacts.id_to_label[idx]: float(prob) for idx, prob in enumerate(row_probs)}
            predictions.append(ModelPrediction(label=label, confidence=float(max(row_probs)), probabilities=probability_map))
        return predictions

    def save(self, save_dir: str):
        import torch

        if self.artifacts is None:
            raise RuntimeError("TextCNN model has not been trained.")
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self.artifacts.model.state_dict(),
                "vocab": self.artifacts.vocab,
                "labels": self.artifacts.labels,
                "label_to_id": self.artifacts.label_to_id,
                "id_to_label": self.artifacts.id_to_label,
                "max_length": self.artifacts.max_length,
                "device": self.artifacts.device,
                "arch_config": self._arch_config,
            },
            target_dir / "model.pt",
        )

    def load(self, save_dir: str):
        import torch

        payload = torch.load(Path(save_dir) / "model.pt", map_location="cpu")
        labels = payload["labels"]
        arch = payload.get("arch_config", self._arch_config)
        model = _TextCNNClassifier.build(
            vocab_size=len(payload["vocab"]),
            embedding_dim=arch["embedding_dim"],
            num_classes=len(labels),
            num_filters=arch["num_filters"],
            kernel_sizes=arch["kernel_sizes"],
            dropout=arch["dropout"],
        )
        model.load_state_dict(payload["state_dict"])
        model.eval()
        self.artifacts = _TextCNNArtifacts(
            model=model,
            vocab=payload["vocab"],
            labels=labels,
            label_to_id=payload["label_to_id"],
            id_to_label=payload["id_to_label"],
            max_length=payload["max_length"],
            device="cpu",
        )

    def is_ready(self) -> bool:
        return self.artifacts is not None
