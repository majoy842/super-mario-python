"""
按模块分块测试脚本（适配 PyCharm 直接运行）。

说明：
- 这是“集成化分步骤测试”，不是 pytest 单元测试。
- 你可以在 PyCharm 中逐段打断点查看每一步输出文件。
- 运行前请确认已安装 requirements.txt 中的依赖。
"""

from __future__ import annotations

import sys
from pathlib import Path

# 兼容“直接运行当前文件”的场景：把仓库根目录加入 sys.path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentiment_system.src.analyze import run_analysis
from sentiment_system.src.preprocess import preprocess
from sentiment_system.src.train import train_model
from sentiment_system.src.visualize import visualize


BASE = Path("sentiment_system")
RAW_DATA = BASE / "data/raw/reviews.csv"
CLEAN_DATA = BASE / "data/processed/reviews_clean.csv"
MODEL_PATH = BASE / "models/sentiment_model.joblib"
METRICS_PATH = BASE / "outputs/metrics.json"
ANALYSIS_PATH = BASE / "outputs/analysis.json"
OUTPUT_DIR = BASE / "outputs"


def test_step_1_preprocess() -> None:
    """测试模块1：数据预处理。"""
    df = preprocess(RAW_DATA, CLEAN_DATA)
    assert len(df) > 0, "预处理后数据不应为空"
    assert CLEAN_DATA.exists(), "预处理输出文件不存在"


def test_step_2_train() -> None:
    """测试模块2：模型训练。"""
    metrics = train_model(CLEAN_DATA, MODEL_PATH, METRICS_PATH)
    assert "accuracy" in metrics, "训练结果缺少 accuracy"
    assert MODEL_PATH.exists(), "模型文件未生成"
    assert METRICS_PATH.exists(), "指标文件未生成"


def test_step_3_analyze() -> None:
    """测试模块3：分析挖掘。"""
    result = run_analysis(CLEAN_DATA, MODEL_PATH, ANALYSIS_PATH)
    assert "pred_positive_ratio" in result, "分析结果缺少正向占比"
    assert ANALYSIS_PATH.exists(), "分析结果 JSON 未生成"
    assert (OUTPUT_DIR / "predictions.csv").exists(), "预测明细未生成"


def test_step_4_visualize() -> None:
    """测试模块4：可视化。"""
    visualize(METRICS_PATH, ANALYSIS_PATH, OUTPUT_DIR)
    assert (OUTPUT_DIR / "summary_bar.png").exists(), "摘要图未生成"
    assert (OUTPUT_DIR / "top_words.png").exists(), "高频词图未生成"


if __name__ == "__main__":
    # 便于在 PyCharm 中“运行当前文件”完成全流程分步测试
    test_step_1_preprocess()
    test_step_2_train()
    test_step_3_analyze()
    test_step_4_visualize()
    print("分模块测试全部通过。")
