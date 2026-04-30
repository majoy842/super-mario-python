from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path

from system import SentimentAnalysisSystem, SystemConfig
from system.config import DataConfig, ExportConfig


@lru_cache(maxsize=4)
def get_system(
    data_path: str = "system/data/reviews.csv",
    output_dir: str = "system/outputs/app_run",
    artifacts_dir: str = "system/outputs/artifacts",
    text_column: str = "content",
    label_column: str = "sentiment_value",
    aspect_column: str = "subject",
) -> SentimentAnalysisSystem:
    config = SystemConfig(
        data=DataConfig(
            data_path=Path(data_path),
            text_column=text_column,
            label_column=label_column,
            aspect_column=aspect_column,
        ),
        export=ExportConfig(output_dir=Path(output_dir), artifacts_dir=Path(artifacts_dir)),
    )
    return SentimentAnalysisSystem(config)


def warmup_inference(system: SentimentAnalysisSystem) -> str:
    status = system.load_available_models()
    ready_models = [name for name, ready in status.items() if ready]
    if not ready_models:
        return "SVM"
    return "BERT" if status.get("BERT") else ready_models[0]


def get_model_status(system: SentimentAnalysisSystem) -> dict[str, bool]:
    status = system.load_available_models()
    return {name: bool(status.get(name, False)) for name in ["BERT", "TextCNN", "SVM"]}


def force_retrain_models(system: SentimentAnalysisSystem) -> dict:
    _, processed_df, _ = system.load_and_preprocess()
    comparison_df = system.force_retrain_models(processed_df=processed_df)
    return {
        "retrained": True,
        "best_model": str(comparison_df.iloc[0]["model"]) if len(comparison_df) else None,
        "comparison": comparison_df.to_dict(orient="records"),
    }


def predict_single(system: SentimentAnalysisSystem, text: str, model_name: str) -> dict:
    if not system.model_status().get(model_name, False):
        raise RuntimeError(f"当前 {model_name} 模型未加载，请先训练并加载权重。")
    return system.analyze_single(text=text, model_name=model_name, with_aspect=False)


def predict_batch(system: SentimentAnalysisSystem, comments: list[str], model_name: str) -> dict:
    if not system.model_status().get(model_name, False):
        raise RuntimeError(f"当前 {model_name} 模型未加载，请先训练并加载权重。")
    routed = system.route_by_volume(
        comments=comments,
        model_name=model_name,
        with_aspect=False,
        with_aspect_distribution=False,
        with_fine_grained=False,
        with_pain_mining=False,
    )
    predictions = routed.get("predictions", [])
    label_counts = Counter(str(item.get("predicted_label", "")) for item in predictions)
    distribution_fig = Path(system.config.export.output_dir) / "label_distribution.png"
    return {
        "mode": routed.get("mode", "empty"),
        "predictions": predictions,
        "label_counts": dict(label_counts),
        "distribution_fig": str(distribution_fig) if distribution_fig.exists() else None,
    }
