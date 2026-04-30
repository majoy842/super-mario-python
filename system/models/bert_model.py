from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .base import BaseSentimentModel, ModelPrediction, TrainingArtifacts


@dataclass
class _BERTArtifacts:
    model: object
    tokenizer: object
    labels: list[str]
    label_to_id: dict[str, int]
    id_to_label: dict[int, str]
    max_length: int
    device: str
    pretrained_model_name: str


class EncodedReviewDataset:
    def __init__(self, input_ids, attention_masks, labels):
        self.input_ids = list(input_ids)
        self.attention_masks = list(attention_masks)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_masks[idx],
            "labels": self.labels[idx],
        }


class BertBatchCollator:
    def __init__(self, tokenizer, device_type: str) -> None:
        from transformers import DataCollatorWithPadding

        self._collator = DataCollatorWithPadding(
            tokenizer=tokenizer,
            pad_to_multiple_of=8 if device_type == "cuda" else None,
        )

    def __call__(self, batch):
        import torch

        labels_tensor = torch.tensor([item["labels"] for item in batch], dtype=torch.long)
        features = [{"input_ids": item["input_ids"], "attention_mask": item["attention_mask"]} for item in batch]
        padded = self._collator(features)
        padded["labels"] = labels_tensor
        return padded


class BERTSentimentModel(BaseSentimentModel):
    model_name = "BERT"

    def _log_progress(self, message: str) -> None:
        print(f"[BERT] {message}", flush=True)

    def __init__(self, pretrained_model_name: str = "bert-base-chinese") -> None:
        self.pretrained_model_name = pretrained_model_name
        self.artifacts: _BERTArtifacts | None = None

    def _resolve_device(self, config):
        import torch

        requested_device = str(getattr(config, "device", "auto")).lower()
        require_gpu = bool(getattr(config, "require_gpu", False))
        cuda_available = torch.cuda.is_available()

        if requested_device in {"cuda", "gpu"}:
            if not cuda_available:
                raise RuntimeError("TrainingConfig.device 设置为 GPU，但当前环境未检测到可用 CUDA 设备。")
            return torch.device("cuda")

        if requested_device == "cpu":
            if require_gpu:
                raise RuntimeError("TrainingConfig.require_gpu=True 时不能强制使用 CPU。")
            return torch.device("cpu")

        if require_gpu and not cuda_available:
            raise RuntimeError("TrainingConfig.require_gpu=True，但当前环境未检测到可用 CUDA 设备。")

        return torch.device("cuda" if cuda_available else "cpu")

    def train(self, train_df, text_column: str, label_column: str, config) -> TrainingArtifacts:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, WeightedRandomSampler
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        labels = sorted(train_df[label_column].astype(str).unique().tolist())
        label_to_id = {label: idx for idx, label in enumerate(labels)}
        id_to_label = {idx: label for label, idx in label_to_id.items()}
        pretrained_model_name = getattr(config, "bert_model_name", self.pretrained_model_name)
        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_model_name,
            num_labels=len(labels),
            id2label=id_to_label,
            label2id=label_to_id,
        )

        encoded_labels = [label_to_id[label] for label in train_df[label_column].astype(str).tolist()]
        train_texts = train_df[text_column].astype(str).tolist()
        tokenized = tokenizer(train_texts, truncation=True, max_length=config.max_length, padding=False)
        dataset = EncodedReviewDataset(
            input_ids=tokenized["input_ids"],
            attention_masks=tokenized["attention_mask"],
            labels=encoded_labels,
        )

        sample_weights = None
        class_weights = None
        if getattr(config, "use_class_weight", True):
            from collections import Counter

            counts = Counter(encoded_labels)
            class_weights = torch.tensor(
                [len(encoded_labels) / (len(labels) * counts[idx]) for idx in range(len(labels))],
                dtype=torch.float,
            )
            sample_weights = torch.tensor([class_weights[label].item() for label in encoded_labels], dtype=torch.float)
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True) if sample_weights is not None else None

        device = self._resolve_device(config)
        if device.type == "cuda":
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            self._log_progress(f"using CUDA device: {gpu_name}")
            enable_tf32 = bool(getattr(config, "bert_tf32", False))
            torch.backends.cuda.matmul.allow_tf32 = enable_tf32
            torch.backends.cudnn.allow_tf32 = enable_tf32
            torch.backends.cudnn.benchmark = True
            if enable_tf32:
                torch.set_float32_matmul_precision("high")
        collate_fn = BertBatchCollator(tokenizer=tokenizer, device_type=device.type)
        model.to(device)
        worker_count = config.num_workers if config.num_workers > 0 else (2 if device.type == "cuda" else 0)
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            collate_fn=collate_fn,
            pin_memory=(device.type == "cuda"),
            num_workers=worker_count,
            persistent_workers=worker_count > 0,
        )
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=float(getattr(config, "weight_decay", 0.01)),
        )
        total_batches = len(loader)
        total_steps = max(1, total_batches * int(config.epochs))
        warmup_steps = int(total_steps * float(getattr(config, "warmup_ratio", 0.1)))
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device) if class_weights is not None else None)
        use_amp = (device.type == "cuda" and getattr(config, "use_amp", True))
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

        model.train()
        self._log_progress(
            f"training start: device={device}, epochs={config.epochs}, batch_size={config.batch_size}, batches_per_epoch={total_batches}, warmup_steps={warmup_steps}"
        )
        if device.type != "cuda":
            self._log_progress("warning: BERT is running on CPU; check CUDA/PyTorch installation or set training.device='cuda'.")
        for epoch in range(config.epochs):
            epoch_loss = 0.0
            for batch_index, batch in enumerate(loader, start=1):
                input_ids = batch["input_ids"].to(device, non_blocking=(device.type == "cuda"))
                attention_mask = batch["attention_mask"].to(device, non_blocking=(device.type == "cuda"))
                labels_tensor = batch["labels"].to(device, non_blocking=(device.type == "cuda"))
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=use_amp):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = criterion(outputs.logits, labels_tensor)
                scaler.scale(loss).backward()
                max_grad_norm = float(getattr(config, "max_grad_norm", 1.0))
                if max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                epoch_loss += float(loss.detach().item())
                if batch_index == 1 or batch_index == total_batches or batch_index % max(1, total_batches // 5) == 0:
                    self._log_progress(
                        f"epoch {epoch + 1}/{config.epochs} - batch {batch_index}/{total_batches} - loss={loss.detach().item():.4f}"
                    )
            self._log_progress(
                f"epoch {epoch + 1}/{config.epochs} completed - avg_loss={epoch_loss / max(1, total_batches):.4f}"
            )

        self.artifacts = _BERTArtifacts(
            model=model,
            tokenizer=tokenizer,
            labels=labels,
            label_to_id=label_to_id,
            id_to_label=id_to_label,
            max_length=config.max_length,
            device=str(device),
            pretrained_model_name=pretrained_model_name,
        )
        return TrainingArtifacts(
            model_name=self.model_name,
            model=self.artifacts,
            metadata={"labels": labels, "pretrained_model_name": pretrained_model_name, "device": str(device)},
        )

    def predict(self, texts: list[str], batch_size: int = 8) -> list[ModelPrediction]:
        import torch
        from torch.utils.data import DataLoader

        if self.artifacts is None:
            raise RuntimeError("BERT model has not been trained.")
        tokenizer = self.artifacts.tokenizer
        model = self.artifacts.model
        device = torch.device(self.artifacts.device)
        tokenized = tokenizer(texts, truncation=True, max_length=self.artifacts.max_length, padding=False)
        dataset = EncodedReviewDataset(
            input_ids=tokenized["input_ids"],
            attention_masks=tokenized["attention_mask"],
            labels=[0] * len(texts),
        )
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=BertBatchCollator(tokenizer=tokenizer, device_type=device.type),
            pin_memory=(device.type == "cuda"),
            num_workers=0,
        )
        model.eval()
        probabilities: list[list[float]] = []
        self._log_progress(
            f"prediction start: device={device}, samples={len(texts)}, batch_size={batch_size}, batches={len(loader)}"
        )
        with torch.no_grad():
            for batch_index, batch in enumerate(loader, start=1):
                batch = {key: value.to(device, non_blocking=(device.type == "cuda")) for key, value in batch.items() if key != "labels"}
                outputs = model(**batch)
                probabilities.extend(torch.softmax(outputs.logits, dim=1).cpu().tolist())
                if batch_index == 1 or batch_index == len(loader) or batch_index % max(1, len(loader) // 5) == 0:
                    self._log_progress(f"prediction progress: batch {batch_index}/{len(loader)}")

        predictions: list[ModelPrediction] = []
        for row_probs in probabilities:
            label_idx = max(range(len(row_probs)), key=lambda idx: row_probs[idx])
            label = self.artifacts.id_to_label[label_idx]
            probability_map = {self.artifacts.id_to_label[idx]: float(prob) for idx, prob in enumerate(row_probs)}
            predictions.append(ModelPrediction(label=label, confidence=float(max(row_probs)), probabilities=probability_map))
        return predictions

    def save(self, save_dir: str):
        import json

        if self.artifacts is None:
            raise RuntimeError("BERT model has not been trained.")
        target_dir = Path(save_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts.model.save_pretrained(target_dir / "model")
        self.artifacts.tokenizer.save_pretrained(target_dir / "tokenizer")
        metadata = {
            "labels": self.artifacts.labels,
            "label_to_id": self.artifacts.label_to_id,
            "id_to_label": self.artifacts.id_to_label,
            "max_length": self.artifacts.max_length,
            "device": self.artifacts.device,
            "pretrained_model_name": self.artifacts.pretrained_model_name,
        }
        (target_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    def load(self, save_dir: str):
        import json
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

        target_dir = Path(save_dir)
        metadata = json.loads((target_dir / "metadata.json").read_text(encoding="utf-8"))
        tokenizer = AutoTokenizer.from_pretrained(target_dir / "tokenizer")
        model = AutoModelForSequenceClassification.from_pretrained(target_dir / "model")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        self.artifacts = _BERTArtifacts(
            model=model,
            tokenizer=tokenizer,
            labels=metadata["labels"],
            label_to_id={k: int(v) for k, v in metadata["label_to_id"].items()},
            id_to_label={int(k): v for k, v in metadata["id_to_label"].items()},
            max_length=int(metadata["max_length"]),
            device=str(device),
            pretrained_model_name=metadata.get("pretrained_model_name", self.pretrained_model_name),
        )

    def is_ready(self) -> bool:
        return self.artifacts is not None
