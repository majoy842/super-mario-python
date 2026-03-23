from __future__ import annotations


class PainPointMiner:
    def __init__(
        self,
        negative_labels: list[str],
        neutral_labels: list[str],
        trigger_terms: list[str],
        forum_noise_terms: list[str],
        synonym_map: dict[str, str],
        top_k: int = 15,
        cluster_count: int = 3,
    ) -> None:
        self.negative_labels = {str(label).lower() for label in negative_labels}
        self.neutral_labels = {str(label).lower() for label in neutral_labels}
        self.trigger_terms = sorted(set(trigger_terms), key=len, reverse=True)
        self.forum_noise_terms = {item.lower() for item in forum_noise_terms}
        self.synonym_map = synonym_map
        self.top_k = top_k
        self.cluster_count = cluster_count

    def _is_negative(self, label: str) -> bool:
        return str(label).lower() in self.negative_labels

    def _is_neutral(self, label: str) -> bool:
        return str(label).lower() in self.neutral_labels

    def _normalize_text(self, text: str) -> str:
        content = str(text)
        for source, target in self.synonym_map.items():
            content = content.replace(source, target)
        return content

    def _match_triggers(self, text: str) -> list[str]:
        normalized_text = self._normalize_text(str(text).lower())
        return [term for term in self.trigger_terms if term.lower() in normalized_text]

    def _noise_hits(self, text: str) -> list[str]:
        normalized_text = str(text).lower()
        return [term for term in self.forum_noise_terms if term in normalized_text]

    def _build_candidate_pool(self, df, text_column: str, label_column: str):
        working_df = df.copy()
        working_df["pain_point_text"] = working_df[text_column].astype(str).map(self._normalize_text)
        working_df["trigger_terms"] = working_df["pain_point_text"].map(self._match_triggers)
        working_df["forum_noise_hits"] = working_df["pain_point_text"].map(self._noise_hits)
        working_df["pain_point_score"] = working_df["trigger_terms"].map(len)
        working_df["candidate_reason"] = ""

        negative_mask = working_df[label_column].map(self._is_negative)
        neutral_trigger_mask = working_df[label_column].map(self._is_neutral) & (working_df["pain_point_score"] > 0)
        trigger_mask = working_df["pain_point_score"] > 0
        low_noise_mask = working_df["forum_noise_hits"].map(len) <= 1
        candidate_mask = (negative_mask | neutral_trigger_mask | trigger_mask) & low_noise_mask

        working_df.loc[negative_mask, "candidate_reason"] = "negative_label"
        working_df.loc[neutral_trigger_mask, "candidate_reason"] = "neutral_with_trigger"
        working_df.loc[trigger_mask & ~negative_mask & ~neutral_trigger_mask, "candidate_reason"] = "trigger_match"
        return working_df[candidate_mask].copy().reset_index(drop=True)

    def _extract_keywords(self, candidate_df, normalized_text_column: str):
        from collections import Counter

        counter = Counter()
        for text in candidate_df[normalized_text_column].astype(str):
            counter.update([token for token in text.split() if token])
        return counter.most_common(self.top_k)

    def _cluster_within_aspect(self, aspect_df, text_column: str, normalized_text_column: str):
        import pandas as pd
        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.pipeline import FeatureUnion

        if aspect_df.empty:
            return aspect_df, []

        text_data = aspect_df[normalized_text_column].fillna("").astype(str)
        sample_count = len(aspect_df)
        if sample_count == 1:
            aspect_df = aspect_df.copy()
            aspect_df["cluster_id"] = 0
            summary = [{"cluster_id": 0, "sample_comments": aspect_df.iloc[0][text_column], "size": 1}]
            return aspect_df, summary

        cluster_count = min(self.cluster_count, max(1, sample_count // 2), sample_count)
        vectorizer = FeatureUnion(
            [
                ("word", TfidfVectorizer(max_features=500, ngram_range=(1, 2))),
                ("char", TfidfVectorizer(max_features=500, analyzer="char_wb", ngram_range=(2, 4))),
            ]
        )
        features = vectorizer.fit_transform(text_data)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        aspect_df = aspect_df.copy()
        aspect_df["cluster_id"] = model.fit_predict(features)
        summary_df = (
            aspect_df.groupby("cluster_id")
            .agg(size=(text_column, "count"), sample_comments=(text_column, lambda items: " | ".join(items.head(3).tolist())))
            .reset_index()
            .sort_values("size", ascending=False)
        )
        return aspect_df, summary_df.to_dict(orient="records")

    def mine(self, df, text_column: str, label_column: str, aspect_column: str, normalized_text_column: str = "normalized_text"):
        import pandas as pd

        candidate_df = self._build_candidate_pool(df, text_column=text_column, label_column=label_column)
        if candidate_df.empty:
            return candidate_df, [], []

        candidate_df[aspect_column] = candidate_df[aspect_column].fillna("其他").astype(str)
        high_freq_terms = self._extract_keywords(candidate_df, normalized_text_column)

        clustered_frames = []
        aspect_cluster_summary = []
        for aspect_name, aspect_df in candidate_df.groupby(aspect_column):
            clustered_df, cluster_summary = self._cluster_within_aspect(
                aspect_df,
                text_column=text_column,
                normalized_text_column=normalized_text_column,
            )
            clustered_frames.append(clustered_df)
            aspect_cluster_summary.append(
                {
                    "aspect": aspect_name,
                    "candidate_count": int(len(aspect_df)),
                    "cluster_summary": cluster_summary,
                }
            )

        combined_df = pd.concat(clustered_frames, ignore_index=True) if clustered_frames else candidate_df.copy()
        return combined_df.reset_index(drop=True), high_freq_terms, aspect_cluster_summary
