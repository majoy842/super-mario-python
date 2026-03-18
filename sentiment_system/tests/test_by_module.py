"""简单分步测试脚本。"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_system.src.analyze import run_analysis
from sentiment_system.src.preprocess import preprocess
from sentiment_system.src.train import train_model
from sentiment_system.src.visualize import visualize

BASE = Path("sentiment_system")


def run_test() -> None:
    raw_path = BASE / "data/raw/reviews.csv"
    clean_path = BASE / "data/processed/reviews_clean.csv"
    model_path = BASE / "models/sentiment_model.joblib"
    metrics_path = BASE / "outputs/metrics.json"
    analysis_path = BASE / "outputs/analysis.json"
    out_dir = BASE / "outputs"

    df = preprocess(raw_path, clean_path)
    assert len(df) > 0

    metrics = train_model(clean_path, model_path, metrics_path)
    assert "accuracy" in metrics

    result = run_analysis(clean_path, model_path, analysis_path)
    assert "pred_positive_ratio" in result

    visualize(metrics_path, analysis_path, out_dir)
    assert (out_dir / "summary_bar.png").exists()

    print("测试通过")


if __name__ == "__main__":
    run_test()
