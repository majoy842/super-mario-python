from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable


class TextPreprocessor:
    def __init__(
        self,
        stopwords_path: Path | None = None,
        custom_dict_path: Path | None = None,
        lowercase: bool = True,
        remove_digits: bool = False,
        min_token_length: int = 1,
    ) -> None:
        self.lowercase = lowercase
        self.remove_digits = remove_digits
        self.min_token_length = min_token_length
        self.stopwords = self._load_stopwords(stopwords_path)
        self.custom_dict_path = custom_dict_path

    @staticmethod
    def _load_stopwords(stopwords_path: Path | None) -> set[str]:
        if stopwords_path is None or not Path(stopwords_path).exists():
            return set()
        return {line.strip() for line in Path(stopwords_path).read_text(encoding="utf-8").splitlines() if line.strip()}

    def clean_text(self, text: str) -> str:
        content = str(text).strip()
        if self.lowercase:
            content = content.lower()
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"[^\w\u4e00-\u9fff\s]", " ", content)
        if self.remove_digits:
            content = re.sub(r"\d+", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        return content

    def tokenize(self, text: str) -> list[str]:
        import jieba

        if self.custom_dict_path and Path(self.custom_dict_path).exists():
            jieba.load_userdict(str(self.custom_dict_path))
        tokens = [token.strip() for token in jieba.lcut(text) if token.strip()]
        return [token for token in tokens if token not in self.stopwords and len(token) >= self.min_token_length]

    def normalize(self, text: str) -> str:
        cleaned = self.clean_text(text)
        tokens = self.tokenize(cleaned)
        return " ".join(tokens)

    def transform_dataframe(self, df, text_column: str, remove_duplicates: bool = True):
        working_df = df.copy()
        working_df[text_column] = working_df[text_column].fillna("").astype(str)
        if remove_duplicates:
            working_df = working_df.drop_duplicates(subset=[text_column])
        working_df["clean_text"] = working_df[text_column].map(self.clean_text)
        working_df["tokens"] = working_df["clean_text"].map(self.tokenize)
        working_df["normalized_text"] = working_df["tokens"].map(lambda items: " ".join(items))
        return working_df.reset_index(drop=True)

    def extract_keywords(self, token_series: Iterable[list[str]], top_k: int = 20) -> list[tuple[str, int]]:
        from collections import Counter

        counter = Counter()
        for tokens in token_series:
            counter.update(tokens)
        return counter.most_common(top_k)
