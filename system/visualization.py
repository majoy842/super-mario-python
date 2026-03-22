from __future__ import annotations

from pathlib import Path


class ResultVisualizer:
    def __init__(self, output_dir: Path, dpi: int = 200) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi

    def plot_label_distribution(self, df, label_column: str, filename: str = "label_distribution.png") -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(8, 5))
        sns.countplot(data=df, x=label_column, order=df[label_column].value_counts().index)
        plt.title("情感标签分布")
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def plot_model_comparison(self, comparison_df, filename: str = "model_comparison.png") -> Path:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 6))
        metrics = [column for column in comparison_df.columns if column != "model"]
        for metric in metrics:
            plt.plot(comparison_df["model"], comparison_df[metric], marker="o", label=metric)
        plt.title("模型效果对比")
        plt.ylim(0, 1)
        plt.legend()
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path

    def plot_aspect_distribution(self, aspect_distribution_df, filename: str = "aspect_distribution.png") -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        plt.figure(figsize=(10, 6))
        sns.barplot(data=aspect_distribution_df, x="aspect_detected", y="count", hue=aspect_distribution_df.columns[1])
        plt.title("细粒度属性情感分布")
        plt.xticks(rotation=30)
        plt.tight_layout()
        path = self.output_dir / filename
        plt.savefig(path, dpi=self.dpi)
        plt.close()
        return path
