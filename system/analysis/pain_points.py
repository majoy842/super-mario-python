from __future__ import annotations


class PainPointMiner:
    def __init__(self, negative_labels: list[str], top_k: int = 15, cluster_count: int = 3) -> None:
        self.negative_labels = {str(label).lower() for label in negative_labels}
        self.top_k = top_k
        self.cluster_count = cluster_count

    def _is_negative(self, label: str) -> bool:
        return str(label).lower() in self.negative_labels

    def mine(self, df, text_column: str, label_column: str, normalized_text_column: str = "normalized_text"):
        from collections import Counter

        from sklearn.cluster import KMeans
        from sklearn.feature_extraction.text import TfidfVectorizer

        negative_df = df[df[label_column].map(self._is_negative)].copy()
        if negative_df.empty:
            return negative_df, [], []

        all_tokens = " ".join(negative_df.get(normalized_text_column, negative_df[text_column]).astype(str).tolist()).split()
        high_freq_terms = Counter(all_tokens).most_common(self.top_k)

        text_data = negative_df.get(normalized_text_column, negative_df[text_column]).astype(str)
        if len(negative_df) < self.cluster_count:
            cluster_count = max(1, len(negative_df))
        else:
            cluster_count = self.cluster_count
        vectorizer = TfidfVectorizer(max_features=500)
        features = vectorizer.fit_transform(text_data)
        model = KMeans(n_clusters=cluster_count, random_state=42, n_init=10)
        negative_df["cluster_id"] = model.fit_predict(features)
        cluster_summary = (
            negative_df.groupby("cluster_id")[text_column]
            .apply(lambda items: " | ".join(items.head(3).tolist()))
            .reset_index(name="sample_comments")
            .to_dict(orient="records")
        )
        return negative_df.reset_index(drop=True), high_freq_terms, cluster_summary
