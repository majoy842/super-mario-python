from __future__ import annotations

from pathlib import Path


class ResultVisualizer:
    def __init__(self, output_dir: Path, dpi: int = 200) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dpi = dpi
        self.font_family = None
        self.font_path = None
        self.font_properties = None
        self._configure_chinese_font()

    def _configure_chinese_font(self) -> None:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            return

        import matplotlib
        from matplotlib import font_manager

        candidate_files = [
            Path("system/assets/fonts/SimHei.ttf"),
            Path("system/assets/fonts/msyh.ttc"),
            Path("system/assets/fonts/Microsoft YaHei.ttf"),
            Path("system/assets/fonts/NotoSansCJK-Regular.ttc"),
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/msyhbd.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
        ]
        candidate_names = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "Source Han Sans SC",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Arial Unicode MS",
        ]

        selected_family = None
        selected_path = None
        for font_path in candidate_files:
            if font_path.exists():
                font_manager.fontManager.addfont(str(font_path))
                selected_path = str(font_path)
                selected_family = font_manager.FontProperties(fname=str(font_path)).get_name()
                break

        if selected_family is None:
            available = {font.name for font in font_manager.fontManager.ttflist}
            for font_name in candidate_names:
                if font_name in available:
                    selected_family = font_name
                    break

        if selected_family:
            matplotlib.rcParams["font.sans-serif"] = [selected_family, "DejaVu Sans"]
            matplotlib.rcParams["font.family"] = "sans-serif"
            if selected_path:
                self.font_properties = font_manager.FontProperties(fname=selected_path)
            else:
                self.font_properties = font_manager.FontProperties(family=selected_family)
        matplotlib.rcParams["axes.unicode_minus"] = False
        self.font_family = selected_family
        self.font_path = selected_path

    def _apply_axis_font(self, ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None, rotate_x: int = 0) -> None:
        if title is not None:
            ax.set_title(title, fontproperties=self.font_properties)
        if xlabel is not None:
            ax.set_xlabel(xlabel, fontproperties=self.font_properties)
        if ylabel is not None:
            ax.set_ylabel(ylabel, fontproperties=self.font_properties)
        for label in ax.get_xticklabels():
            if self.font_properties is not None:
                label.set_fontproperties(self.font_properties)
            if rotate_x:
                label.set_rotation(rotate_x)
        for label in ax.get_yticklabels():
            if self.font_properties is not None:
                label.set_fontproperties(self.font_properties)
        legend = ax.get_legend()
        if legend is not None:
            for text in legend.get_texts():
                if self.font_properties is not None:
                    text.set_fontproperties(self.font_properties)
            if legend.get_title() is not None and self.font_properties is not None:
                legend.get_title().set_fontproperties(self.font_properties)

    def plot_label_distribution(self, df, label_column: str, filename: str = "label_distribution.png") -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.countplot(data=df, x=label_column, order=df[label_column].value_counts().index, ax=ax)
        self._apply_axis_font(ax, title="情感标签分布", xlabel=label_column, ylabel="count")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_model_comparison(self, comparison_df, filename: str = "model_comparison.png") -> Path:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        metrics = [column for column in comparison_df.columns if column != "model"]
        for metric in metrics:
            ax.plot(comparison_df["model"], comparison_df[metric], marker="o", label=metric)
        ax.set_ylim(0, 1)
        ax.legend(prop=self.font_properties)
        self._apply_axis_font(ax, title="模型效果对比", xlabel="model", ylabel="score")
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_aspect_distribution(self, aspect_distribution_df, filename: str = "aspect_distribution.png") -> Path:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(data=aspect_distribution_df, x="aspect_detected", y="count", hue=aspect_distribution_df.columns[1], ax=ax)
        self._apply_axis_font(ax, title="细粒度属性情感分布", xlabel="aspect_detected", ylabel="count", rotate_x=30)
        fig.tight_layout()
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path

    def plot_wordcloud(self, term_frequencies: dict[str, int], filename: str = "pain_point_wordcloud.png") -> Path | None:
        import importlib.util

        if not term_frequencies:
            return None
        if importlib.util.find_spec("wordcloud") is None or importlib.util.find_spec("matplotlib") is None:
            return None

        from wordcloud import WordCloud
        import matplotlib.pyplot as plt

        if self.font_path is None and self.font_family is None:
            return None

        wordcloud = WordCloud(
            width=1200,
            height=800,
            background_color="white",
            font_path=self.font_path,
            collocations=False,
        ).generate_from_frequencies(term_frequencies)

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(wordcloud, interpolation="bilinear")
        ax.axis("off")
        self._apply_axis_font(ax, title="痛点高频词词云")
        path = self.output_dir / filename
        fig.savefig(path, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        return path
