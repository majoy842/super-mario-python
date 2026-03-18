"""总流程：预处理 -> 训练 -> 分析 -> 可视化。

说明：
- 本文件已做 PyCharm 友好处理。
- 你可以在 PyCharm 中直接右键运行本文件。
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_system.src.analyze import run_analysis
from sentiment_system.src.preprocess import preprocess
from sentiment_system.src.train import train_model
from sentiment_system.src.visualize import visualize


BASE_DIR = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(BASE_DIR / "data/raw/reviews.csv"))
    args = parser.parse_args()

    raw_path = Path(args.input)
    clean_path = BASE_DIR / "data/processed/reviews_clean.csv"
    model_path = BASE_DIR / "models/sentiment_model.joblib"
    metrics_path = BASE_DIR / "outputs/metrics.json"
    analysis_path = BASE_DIR / "outputs/analysis.json"
    output_dir = BASE_DIR / "outputs"

    preprocess(raw_path, clean_path)
    metrics = train_model(clean_path, model_path, metrics_path)
    analysis = run_analysis(clean_path, model_path, analysis_path)
    visualize(metrics_path, analysis_path, output_dir)

    top = metrics['results'][0]
    print(f"完成！最佳模型={metrics['best_model']}，macro_f1={top['macro_f1']:.4f}，正向占比={analysis['pred_positive_ratio']:.2%}")


if __name__ == "__main__":
    main()
