from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from wordcloud import WordCloud


FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _pick_font() -> str | None:
    for f in FONT_CANDIDATES:
        if Path(f).exists():
            return f
    return None


def visualize(metrics_path: Path, analysis_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    with metrics_path.open("r", encoding="utf-8") as f:
        metrics = json.load(f)
    with analysis_path.open("r", encoding="utf-8") as f:
        analysis = json.load(f)

    accuracy = metrics["accuracy"]
    ratio = analysis["pred_positive_ratio"]

    plt.figure(figsize=(6, 4))
    plt.bar(["Accuracy", "Positive Ratio"], [accuracy, ratio], color=["#4caf50", "#2196f3"])
    plt.ylim(0, 1)
    plt.title("Model Summary")
    plt.tight_layout()
    plt.savefig(out_dir / "summary_bar.png", dpi=150)
    plt.close()

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

    font_path = _pick_font()
    pos_freq = dict(analysis["top_positive_words"])
    if pos_freq:
        wc = WordCloud(font_path=font_path, width=800, height=400, background_color="white")
        wc.generate_from_frequencies(pos_freq)
        wc.to_file(str(out_dir / "positive_wordcloud.png"))


def main() -> None:
    parser = argparse.ArgumentParser(description="生成情感分析可视化图表")
    parser.add_argument("--metrics", default="sentiment_system/outputs/metrics.json")
    parser.add_argument("--analysis", default="sentiment_system/outputs/analysis.json")
    parser.add_argument("--out", default="sentiment_system/outputs")
    args = parser.parse_args()

    visualize(Path(args.metrics), Path(args.analysis), Path(args.out))
    print(f"可视化图表已生成到: {args.out}")


if __name__ == "__main__":
    main()
