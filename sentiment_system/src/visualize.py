"""可视化模块：把结果画成图片。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

BASE_DIR = Path(__file__).resolve().parents[1]


def visualize(metrics_path: Path, analysis_path: Path, out_dir: Path) -> None:
    """生成 3 张图：摘要柱状图、关键词柱状图、正向词云。"""
    out_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with analysis_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    plt.figure(figsize=(6, 4))
    plt.bar(["accuracy", "positive_ratio"], [metrics["accuracy"], analysis["pred_positive_ratio"]])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=120)
    plt.close()

    pos = dict(analysis["top_positive_words"])
    neg = dict(analysis["top_negative_words"])

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.barh(list(pos.keys()), list(pos.values()))
    plt.title("positive")

    plt.subplot(1, 2, 2)
    plt.barh(list(neg.keys()), list(neg.values()))
    plt.title("negative")

    plt.tight_layout()
    plt.savefig(out_dir / "top_words.png", dpi=120)
    plt.close()

    if pos:
        wc = WordCloud(width=800, height=400, background_color="white")
        wc.generate_from_frequencies(pos)
        wc.to_file(str(out_dir / "positive_wordcloud.png"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", default=str(BASE_DIR / "outputs/metrics.json"))
    parser.add_argument("--analysis", default=str(BASE_DIR / "outputs/analysis.json"))
    parser.add_argument("--out", default=str(BASE_DIR / "outputs"))
    args = parser.parse_args()

    visualize(Path(args.metrics), Path(args.analysis), Path(args.out))
    print("可视化完成")


if __name__ == "__main__":
    main()
