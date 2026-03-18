"""
可视化模块（Python 3.11）。

职责：
- 读取训练指标和分析结果
- 生成摘要柱状图（准确率、预测正向占比）
- 生成正负面高频词对比图
- 生成正面词云

你可以在 PyCharm 中单独运行本文件测试图表生成。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

# 使用无界面后端，便于在服务器或 CI 环境生成图片
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud

# 优先尝试可显示中文的字体路径
FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _pick_font() -> str | None:
    """从候选字体中选出可用字体，避免词云中文乱码。"""
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    return None


def visualize(metrics_path: Path, analysis_path: Path, out_dir: Path) -> None:
    """根据指标与分析结果生成可视化图表。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with analysis_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    accuracy = metrics["accuracy"]
    ratio = analysis["pred_positive_ratio"]

    # 图1：总体摘要（准确率 vs 正向占比）
    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Positive Ratio"], [accuracy, ratio], color=["#4caf50", "#2196f3"])
    plt.ylim(0, 1)
    plt.title("Model Summary")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()

    # 图2：正负面高频词对比
    top_pos = pd.DataFrame(analysis["top_positive_words"][:10], columns=["word", "count"])
    top_neg = pd.DataFrame(analysis["top_negative_words"][:10], columns=["word", "count"])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.barh(top_pos["word"], top_pos["count"], color="#66bb6a")
    plt.gca().invert_yaxis()
    plt.title("Top Positive Words")

    plt.subplot(1, 2, 2)
    plt.barh(top_neg["word"], top_neg["count"], color="#ef5350")
    plt.gca().invert_yaxis()
    plt.title("Top Negative Words")

    plt.tight_layout()
    plt.savefig(out_dir / "top_words.png", dpi=150)
    plt.close()

    # 图3：正面词云
    font_path = _pick_font()
    pos_freq = dict(analysis["top_positive_words"])
    if pos_freq:
        wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white")
        wc.generate_from_frequencies(pos_freq)
        wc.to_file(str(out_dir / "positive_wordcloud.png"))


def main() -> None:
    """命令行入口：用于单模块测试。"""
    parser = argparse.ArgumentParser(description="生成情感分析可视化图表")
    parser.add_argument("--metrics", default="sentiment_system/outputs/metrics.json")
    parser.add_argument("--analysis", default="sentiment_system/outputs/analysis.json")
    parser.add_argument("--out", default="sentiment_system/outputs")
    args = parser.parse_args()

    visualize(Path(args.metrics), Path(args.analysis), Path(args.out))
    print(f"可视化图表已生成到: {args.out}")


if __name__ == "__main__":
    main()
