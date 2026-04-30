from __future__ import annotations

from pathlib import Path


class ResultVisualizer:
    def __init__(self, output_dir: Path, chart_dpi: int = 120, wordcloud_font_path: Path | None = None) -> None:
        self.output_dir = Path(output_dir)
        self.chart_dpi = chart_dpi
        self.wordcloud_font_path = Path(wordcloud_font_path) if wordcloud_font_path else None
        self.plot_font_family: str | None = None
        self.plot_font_path: str | None = None
        self.plot_font_properties = None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._configure_plot_font()

    def _configure_plot_font(self) -> None:
        import importlib.util

        if importlib.util.find_spec("matplotlib") is None:
            return

        import matplotlib
        from matplotlib import font_manager

        candidate_paths = [
            self.wordcloud_font_path,
            Path("C:/Windows/Fonts/msyh.ttc"),
            Path("C:/Windows/Fonts/simhei.ttf"),
            Path("C:/Windows/Fonts/simsun.ttc"),
            Path("/System/Library/Fonts/PingFang.ttc"),
            Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"),
            Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
        ]
        candidate_names = [
            "Microsoft YaHei",
            "SimHei",
            "SimSun",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "PingFang SC",
        ]

        selected_path = None
        selected_family = None
        for path in candidate_paths:
            if path and Path(path).exists():
                selected_path = str(path)
                font_manager.fontManager.addfont(selected_path)
                selected_family = font_manager.FontProperties(fname=selected_path).get_name()
                break

        if selected_family is None:
            available = {font.name for font in font_manager.fontManager.ttflist}
            for family in candidate_names:
                if family in available:
                    selected_family = family
                    break

        matplotlib.rcParams["font.sans-serif"] = (
            [selected_family, "DejaVu Sans"]
            if selected_family
            else candidate_names + ["DejaVu Sans"]
        )
        matplotlib.rcParams["font.family"] = "sans-serif"
        if selected_family:
            matplotlib.rcParams["font.family"] = "sans-serif"
            self.plot_font_properties = (
                font_manager.FontProperties(fname=selected_path)
                if selected_path
                else font_manager.FontProperties(family=selected_family)
            )

        matplotlib.rcParams["axes.unicode_minus"] = False
        self.plot_font_family = selected_family
        self.plot_font_path = selected_path

    def _resolve_wordcloud_font(self) -> str | None:
        candidate_paths = []
        if self.wordcloud_font_path:
            candidate_paths.append(self.wordcloud_font_path)
        if self.plot_font_path:
            candidate_paths.append(Path(self.plot_font_path))

        module_dir = Path(__file__).resolve().parent
        candidate_paths.extend(
            [
                module_dir / "assets" / "fonts" / "msyh.ttc",
                module_dir / "assets" / "fonts" / "simhei.ttf",
                module_dir / "assets" / "fonts" / "simsun.ttc",
                module_dir / "assets" / "fonts" / "NotoSansCJK-Regular.ttc",
                Path("C:/Windows/Fonts/msyh.ttc"),
                Path("C:/Windows/Fonts/simhei.ttf"),
                Path("C:/Windows/Fonts/simsun.ttc"),
                Path("/System/Library/Fonts/PingFang.ttc"),
                Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
                Path("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"),
            ]
        )
        for path in candidate_paths:
            if path and path.exists():
                return str(path)
        return None

    def _safe_plot(self, plotter, filename: str):
        try:
            import matplotlib.pyplot as plt

            plotter(plt)
            out = self.output_dir / filename
            plt.tight_layout()
            plt.savefig(out, dpi=self.chart_dpi)
            plt.close()
            return out
        except Exception:
            return None

    def plot_model_comparison(self, comparison_df):
        def _plot(plt):
            metrics = [
                column
                for column in comparison_df.columns
                if column not in {"model", "status"}
            ]
            ax = comparison_df.set_index("model")[metrics].plot(
                kind="line",
                marker="o",
                linewidth=2,
                rot=0,
            )
            ax.set_ylim(0, 1.02)
            ax.legend(loc="lower left", prop=self.plot_font_properties)
            ax.set_ylabel("得分", fontproperties=self.plot_font_properties)
            ax.set_xlabel("模型", fontproperties=self.plot_font_properties)
            ax.set_title("模型性能对比", fontproperties=self.plot_font_properties)

        return self._safe_plot(_plot, "model_comparison.png")

    def plot_label_distribution(self, df, label_column: str):
        def _plot(plt):
            counts = df[label_column].value_counts()
            counts.plot(kind="bar")
            plt.title("Label Distribution")

        return self._safe_plot(_plot, "label_distribution.png")

    def plot_aspect_distribution(self, distribution_df):
        def _plot(plt):
            pivot = distribution_df.pivot_table(index="aspect_detected", columns=distribution_df.columns[1], values="count", fill_value=0)
            pivot.plot(kind="bar", stacked=True)
            plt.title("Aspect Distribution")

        return self._safe_plot(_plot, "aspect_distribution.png")

    def plot_wordcloud(self, word_freq: dict[str, int], filename: str, max_words: int = 100, min_font_size: int = 8):
        if not word_freq:
            return None

        def _plot(plt):
            from wordcloud import WordCloud

            wordcloud_kwargs = dict(
                width=1400,
                height=900,
                background_color="white",
                max_words=max_words,
                min_font_size=min_font_size,
                collocations=False,
            )
            font_path = self._resolve_wordcloud_font()
            if font_path:
                wordcloud_kwargs["font_path"] = font_path
            wc = WordCloud(**wordcloud_kwargs)
            wc.generate_from_frequencies(word_freq)
            plt.imshow(wc, interpolation="bilinear")
            plt.axis("off")

        return self._safe_plot(_plot, filename)
