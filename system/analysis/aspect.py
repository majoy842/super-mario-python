from __future__ import annotations


class FineGrainedAnalyzer:
    def __init__(
        self,
        aspect_keywords: dict[str, list[str]],
        focus_aspects: list[str] | None = None,
        include_other_in_focus_analysis: bool = False,
    ) -> None:
        self.aspect_keywords = aspect_keywords
        self.focus_aspects = focus_aspects or []
        self.include_other_in_focus_analysis = include_other_in_focus_analysis

    def infer_aspect(self, text: str) -> str:
        for aspect, keywords in self.aspect_keywords.items():
            if any(keyword in text for keyword in keywords):
                return aspect
        return "其他"

    def analyze(self, df, text_column: str, label_column: str, aspect_column: str | None = None):
        working_df = df.copy()
        if aspect_column and aspect_column in working_df.columns:
            working_df["aspect_detected"] = working_df[aspect_column].fillna("其他").astype(str)
        else:
            working_df["aspect_detected"] = working_df[text_column].astype(str).map(self.infer_aspect)

        distribution = (
            working_df.groupby(["aspect_detected", label_column])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
        )
        summary = (
            distribution.pivot_table(index="aspect_detected", columns=label_column, values="count", fill_value=0)
            .reset_index()
            .sort_values("aspect_detected")
        )

        focus_df = working_df.copy()
        if self.focus_aspects:
            focus_df = focus_df[focus_df["aspect_detected"].isin(self.focus_aspects)].copy()
        if self.include_other_in_focus_analysis:
            focus_df = working_df.copy()

        focus_distribution = (
            focus_df.groupby(["aspect_detected", label_column])
            .size()
            .reset_index(name="count")
            .sort_values("count", ascending=False)
            if not focus_df.empty
            else distribution.iloc[0:0].copy()
        )
        return working_df, distribution, summary, focus_distribution


DEFAULT_ASPECT_KEYWORDS = {
    "价格": ["贵", "便宜", "性价比", "价格"],
    "服务": ["服务", "客服", "态度", "响应"],
    "质量": ["质量", "做工", "耐用", "材质"],
    "物流": ["物流", "快递", "送达", "配送"],
}
