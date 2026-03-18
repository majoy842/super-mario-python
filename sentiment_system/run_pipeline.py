"""
情感分析系统总入口（适配 Python 3.11）。

这个脚本强调“按步骤直连调用”，不做额外封装，方便你在 PyCharm 中：
1. 直接右键运行本文件；
2. 在断点模式下逐步调试每个阶段；
3. 清楚看到每个模块的输入和输出路径。
"""

from __future__ import annotations

from pathlib import Path

from src.analyze import run_analysis
from src.preprocess import preprocess
from src.train import train_model
from src.visualize import visualize


def main() -> None:
    """按顺序执行：预处理 -> 训练 -> 分析 -> 可视化。"""
    base = Path("sentiment_system")

    # -------------------------
    # 1) 路径定义（统一在这里改）
    # -------------------------
    raw_data = base / "data/raw/reviews.csv"
    clean_data = base / "data/processed/reviews_clean.csv"
    model_path = base / "models/sentiment_model.joblib"
    metrics_path = base / "outputs/metrics.json"
    analysis_path = base / "outputs/analysis.json"
    outputs_dir = base / "outputs"

    # -------------------------
    # 2) 执行各阶段
    # -------------------------
    preprocess(raw_data, clean_data)
    metrics = train_model(clean_data, model_path, metrics_path)
    analysis = run_analysis(clean_data, model_path, analysis_path)
    visualize(metrics_path, analysis_path, outputs_dir)

    # -------------------------
    # 3) 简单结果摘要
    # -------------------------
    print(
        f"流程完成：准确率={metrics['accuracy']:.4f}，"
        f"预测正向占比={analysis['pred_positive_ratio']:.2%}"
    )


if __name__ == "__main__":
    main()
