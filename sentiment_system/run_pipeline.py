"""总流程：预处理 -> 训练 -> 分析 -> 可视化。"""

from __future__ import annotations

from pathlib import Path

from src.analyze import run_analysis
from src.preprocess import preprocess
from src.train import train_model
from src.visualize import visualize


def main() -> None:
    base = Path("sentiment_system")

    raw_path = base / "data/raw/reviews.csv"
    clean_path = base / "data/processed/reviews_clean.csv"
    model_path = base / "models/sentiment_model.joblib"
    metrics_path = base / "outputs/metrics.json"
    analysis_path = base / "outputs/analysis.json"
    output_dir = base / "outputs"

    preprocess(raw_path, clean_path)
    metrics = train_model(clean_path, model_path, metrics_path)
    analysis = run_analysis(clean_path, model_path, analysis_path)
    visualize(metrics_path, analysis_path, output_dir)

    print(f"完成！准确率={metrics['accuracy']:.4f}，正向占比={analysis['pred_positive_ratio']:.2%}")


if __name__ == "__main__":
    main()
