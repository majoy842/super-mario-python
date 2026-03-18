"""总流程：预处理 -> 训练 -> 分析 -> 可视化。

说明：
- 本文件已做 PyCharm 友好处理。
- 你可以在 PyCharm 中直接右键运行本文件。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 兼容 PyCharm 直接运行脚本：把仓库根目录加到 sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sentiment_system.src.analyze import run_analysis
from sentiment_system.src.preprocess import preprocess
from sentiment_system.src.train import train_model
from sentiment_system.src.visualize import visualize


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
