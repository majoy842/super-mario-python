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
        min_text_length: int = 4,
        max_text_length: int = 512,
        drop_symbol_only: bool = True,
        drop_meaningless_english: bool = True,
        noise_phrases: list[str] | None = None,
        synonym_map: dict[str, str] | None = None,
    ) -> None:
        self.lowercase = lowercase
        self.remove_digits = remove_digits
        self.min_token_length = min_token_length
        self.min_text_length = min_text_length
        self.max_text_length = max_text_length
        self.drop_symbol_only = drop_symbol_only
        self.drop_meaningless_english = drop_meaningless_english
        self.noise_phrases = {item.strip().lower() for item in (noise_phrases or []) if item.strip()}
        self.synonym_map = synonym_map or {}
        self.stopwords = self._load_stopwords(stopwords_path)
        self.custom_dict_path = custom_dict_path

    @staticmethod
    def _load_stopwords(stopwords_path: Path | None) -> set[str]:
        if stopwords_path is None or not Path(stopwords_path).exists():
            return set()
        return {line.strip() for line in Path(stopwords_path).read_text(encoding="utf-8").splitlines() if line.strip()}

    @staticmethod
    def _contains_chinese(text: str) -> bool:
        return bool(re.search(r"[\u4e00-\u9fff]", text))

    def apply_synonym_map(self, text: str) -> str:
        normalized_text = text
        for source, target in self.synonym_map.items():
            normalized_text = normalized_text.replace(source, target)
        return normalized_text

    def is_symbol_only(self, text: str) -> bool:
        stripped = re.sub(r"\s+", "", str(text))
        if not stripped:
            return True
        return not bool(re.search(r"[A-Za-z0-9\u4e00-\u9fff]", stripped))

    def is_meaningless_english(self, text: str) -> bool:
        english_only = re.sub(r"[^A-Za-z\s]", "", text).strip()
        if not english_only:
            return False
        words = [item for item in english_only.split() if item]
        return len(words) > 0 and not self._contains_chinese(text) and len(words) <= 2

    def clean_text(self, text: str) -> str:
        content = str(text).strip()
        if self.lowercase:
            content = content.lower()
        content = self.apply_synonym_map(content)
        content = re.sub(r"\s+", " ", content)
        content = re.sub(r"[\t\r\n]+", " ", content)
        if self.remove_digits:
            content = re.sub(r"\d+", " ", content)
        content = re.sub(r"[^\x00-\x7F\u4e00-\u9fffA-Za-z0-9\s]", " ", content)
        content = re.sub(r"\s+", " ", content).strip()
        return content[: self.max_text_length]

    def should_drop(self, text: str) -> bool:
        cleaned = str(text).strip().lower()
        if not cleaned:
            return True
        if self.drop_symbol_only and self.is_symbol_only(cleaned):
            return True
        if len(cleaned) < self.min_text_length:
            return True
        if cleaned in self.noise_phrases:
            return True
        if self.drop_meaningless_english and self.is_meaningless_english(cleaned):
            return True
        return False

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

    @staticmethod
    def normalize_label(value):
        mapping = {"-1": "negative", -1: "negative", "0": "neutral", 0: "neutral", "1": "positive", 1: "positive"}
        return mapping.get(value, mapping.get(str(value), str(value)))

    def transform_dataframe(self, df, text_column: str, label_column: str, aspect_column: str, remove_duplicates: bool = True):
        working_df = df.copy()
        working_df[text_column] = working_df[text_column].fillna("").astype(str)
        if aspect_column in working_df.columns:
            working_df[aspect_column] = working_df[aspect_column].fillna("其他").astype(str)
        else:
            working_df[aspect_column] = "其他"
        if label_column in working_df.columns:
            working_df[label_column] = working_df[label_column].map(self.normalize_label)
        working_df["raw_text_length"] = working_df[text_column].map(lambda item: len(str(item).strip()))
        if remove_duplicates:
            working_df = working_df.drop_duplicates(subset=[text_column])
        working_df["clean_text"] = working_df[text_column].map(self.clean_text)
        working_df["should_drop"] = working_df["clean_text"].map(self.should_drop)
        working_df = working_df[~working_df["should_drop"]].copy()
        working_df["text_length"] = working_df["clean_text"].map(len)
        working_df["tokens"] = working_df["clean_text"].map(self.tokenize)
        working_df["normalized_text"] = working_df["tokens"].map(lambda items: " ".join(items)[: self.max_text_length])
        working_df = working_df[working_df["normalized_text"].str.len() >= self.min_text_length].copy()
        return working_df.reset_index(drop=True)

    def extract_keywords(self, token_series: Iterable[list[str]], top_k: int = 20) -> list[tuple[str, int]]:
        from collections import Counter

        counter = Counter()
        for tokens in token_series:
            counter.update(tokens)
        return counter.most_common(top_k)
