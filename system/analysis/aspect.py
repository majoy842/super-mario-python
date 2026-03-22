from __future__ import annotations


class FineGrainedAnalyzer:
    def __init__(self, aspect_keywords: dict[str, list[str]]) -> None:
        self.aspect_keywords = aspect_keywords

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
            working_df.groupby(["aspect_detected", label_column]).size().reset_index(name="count").sort_values("count", ascending=False)
        )
        summary = (
            distribution.pivot_table(index="aspect_detected", columns=label_column, values="count", fill_value=0)
            .reset_index()
            .sort_values("aspect_detected")
        )
        return working_df, distribution, summary
