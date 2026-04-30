from __future__ import annotations

import re


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
        stopwords: set[str] | None = None,
        min_token_length: int = 2,
        allowed_pos_prefixes: tuple[str, ...] = ("n", "v", "a"),
        scene_terms: list[str] | None = None,
        preserve_negation_words: list[str] | None = None,
        preserve_degree_words: list[str] | None = None,
        preserve_emotion_words: list[str] | None = None,
        pain_aspect_terms: list[str] | None = None,
        pain_negative_descriptors: list[str] | None = None,
        weak_terms: list[str] | None = None,
        topic_terms: list[str] | None = None,
        phrase_normalization_map: dict[str, str] | None = None,
        pain_aspect_mapping: dict[str, list[str]] | None = None,
        incomplete_phrase_terms: list[str] | None = None,
        bare_negative_terms: list[str] | None = None,
        positive_neutral_terms: list[str] | None = None,
        standard_label_whitelist: list[str] | None = None,
    ) -> None:
        self.negative_labels = {str(label).lower() for label in negative_labels}
        self.neutral_labels = {str(label).lower() for label in neutral_labels}
        self.trigger_terms = sorted(set(trigger_terms), key=len, reverse=True)
        self.forum_noise_terms = {item.lower() for item in forum_noise_terms}
        self.synonym_map = synonym_map
        self.top_k = top_k
        self.cluster_count = cluster_count
        self.stopwords = {item.strip() for item in (stopwords or set()) if item.strip()}
        self.scene_terms = {item.strip() for item in (scene_terms or []) if item.strip()}
        self.min_token_length = min_token_length
        self.allowed_pos_prefixes = allowed_pos_prefixes
        self.preserve_negation_words = {item.strip() for item in (preserve_negation_words or []) if item.strip()}
        self.preserve_degree_words = {item.strip() for item in (preserve_degree_words or []) if item.strip()}
        self.preserve_emotion_words = {item.strip() for item in (preserve_emotion_words or []) if item.strip()}
        self.protected_words = self.preserve_negation_words | self.preserve_degree_words | self.preserve_emotion_words
        self.pain_aspect_terms = {item.strip() for item in (pain_aspect_terms or []) if item.strip()}
        self.pain_negative_descriptors = {item.strip() for item in (pain_negative_descriptors or []) if item.strip()}
        self.weak_terms = {item.strip() for item in (weak_terms or []) if item.strip()}
        self.topic_terms = {item.strip() for item in (topic_terms or []) if item.strip()}
        self.phrase_normalization_map = {k.strip(): v.strip() for k, v in (phrase_normalization_map or {}).items() if k.strip() and v.strip()}
        self.pain_aspect_mapping = {name: [k.strip() for k in keywords if k.strip()] for name, keywords in (pain_aspect_mapping or {}).items()}
        self.incomplete_phrase_terms = {item.strip() for item in (incomplete_phrase_terms or []) if item.strip()}
        self.bare_negative_terms = {item.strip() for item in (bare_negative_terms or []) if item.strip()}
        self.positive_neutral_terms = {item.strip() for item in (positive_neutral_terms or []) if item.strip()}
        self.standard_label_whitelist = {item.strip() for item in (standard_label_whitelist or []) if item.strip()}
        self.standard_label_whitelist_compact = {item.replace(" ", "") for item in self.standard_label_whitelist}
        self.single_char_keep = {item for item in self.preserve_emotion_words if len(item) == 1}
        self.invalid_phrase_terms = {
            "不错",
            "还行",
            "可以",
            "挺好",
            "还好",
            "意思",
            "实在",
            "比较",
            "确实",
            "感觉",
            "觉得",
            "没兴趣",
            "提升很大",
            "差点",
            "可能贵",
        }
        self.complaint_patterns = [
            re.compile(pattern)
            for pattern in [
                r"(不稳|不行|不好|不舒服|不识别|识别不了|延迟|卡顿|断连|掉线|掉驱动|发热|偏高|刺耳|闷|糊|噪声|底噪)",
                r"(不会|不能|无法|没法|没办法).{0,4}(连接|驱动|识别|使用)",
            ]
        ]
        self.contrast_markers = ["但是", "不过", "然而", "可惜", "就是", "只是", "但", "唯一不足是"]
        self.reverse_positive_terms = {"不贵", "好听", "满意", "不错", "可以", "还行"}

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

    @staticmethod
    def _split_clauses(text: str) -> list[str]:
        parts = re.split(r"[，,。；;！!？?、\n]+", str(text))
        return [part.strip() for part in parts if part.strip()]

    def _extract_negative_clauses(self, text: str) -> list[str]:
        content = self._normalize_text(text)
        clauses = self._split_clauses(content)
        selected: list[str] = []

        for marker in self.contrast_markers:
            if marker in content:
                tails = content.split(marker)[1:]
                selected.extend(self._split_clauses(" ".join(tails)))
                break

        if selected:
            return selected

        for clause in clauses:
            trigger_hit = len(self._match_triggers(clause)) > 0
            complaint_hit = any(pattern.search(clause) for pattern in self.complaint_patterns)
            descriptor_hit = any(term in clause for term in self.pain_negative_descriptors)
            if trigger_hit or complaint_hit or descriptor_hit:
                selected.append(clause)
        return selected or clauses

    def _token_filter(self, token: str) -> bool:
        token = token.strip()
        if not token:
            return False
        if token in self.protected_words:
            return True
        if token in self.stopwords:
            return False
        if len(token) < self.min_token_length and token not in self.single_char_keep:
            return False
        return True

    def extract_filtered_tokens(self, text: str) -> list[str]:
        import jieba.posseg as pseg

        tokens: list[str] = []
        for pair in pseg.cut(self._normalize_text(text)):
            word = pair.word.strip()
            if not self._token_filter(word):
                continue
            if self.allowed_pos_prefixes and not pair.flag.startswith(self.allowed_pos_prefixes):
                continue
            tokens.append(word)
        return tokens

    def _build_candidate_pool(self, df, text_column: str, label_column: str):
        working_df = df.copy()
        working_df["pain_point_text"] = working_df[text_column].astype(str).map(self._normalize_text)
        working_df["trigger_terms"] = working_df["pain_point_text"].map(self._match_triggers)
        working_df["forum_noise_hits"] = working_df["pain_point_text"].map(self._noise_hits)
        working_df["pain_point_score"] = working_df["trigger_terms"].map(len)
        working_df["complaint_pattern_hits"] = working_df["pain_point_text"].map(
            lambda text: [pattern.pattern for pattern in self.complaint_patterns if pattern.search(str(text))]
        )
        working_df["pain_point_clauses"] = working_df["pain_point_text"].map(self._extract_negative_clauses)
        working_df["pain_point_clause_text"] = working_df["pain_point_clauses"].map(lambda clauses: "；".join(clauses))
        working_df["candidate_reason"] = ""

        negative_mask = working_df[label_column].map(self._is_negative)
        complaint_mask = working_df["complaint_pattern_hits"].map(len) > 0
        neutral_trigger_mask = working_df[label_column].map(self._is_neutral) & (
            (working_df["pain_point_score"] > 0) | complaint_mask
        )
        trigger_mask = working_df["pain_point_score"] > 0
        explicit_complaint_mask = complaint_mask & ~negative_mask
        low_noise_mask = working_df["forum_noise_hits"].map(len) <= 1
        candidate_mask = (negative_mask | neutral_trigger_mask | trigger_mask | explicit_complaint_mask) & low_noise_mask

        working_df.loc[negative_mask, "candidate_reason"] = "negative_label"
        working_df.loc[neutral_trigger_mask, "candidate_reason"] = "neutral_with_trigger"
        working_df.loc[trigger_mask & ~negative_mask & ~neutral_trigger_mask, "candidate_reason"] = "trigger_match"
        working_df.loc[explicit_complaint_mask & ~negative_mask & ~neutral_trigger_mask, "candidate_reason"] = "complaint_pattern"
        return working_df[candidate_mask].copy().reset_index(drop=True)

    def _extract_pain_phrases(self, tokens: list[str]) -> list[str]:
        phrases: list[str] = []
        for idx in range(len(tokens) - 1):
            phrase = f"{tokens[idx]} {tokens[idx + 1]}"
            if self._is_valid_phrase(phrase):
                phrases.append(phrase)
        for idx in range(len(tokens) - 2):
            phrase = f"{tokens[idx]} {tokens[idx + 1]} {tokens[idx + 2]}"
            if self._is_valid_phrase(phrase):
                phrases.append(phrase)

        for idx, token in enumerate(tokens):
            if token not in self.pain_aspect_terms:
                continue
            for offset in range(1, 4):
                if idx + offset >= len(tokens):
                    break
                descriptor = tokens[idx + offset]
                if descriptor in self.pain_negative_descriptors or descriptor in self.protected_words:
                    phrase = f"{token} {descriptor}"
                    if self._is_valid_phrase(phrase):
                        phrases.append(phrase)
        return [self._normalize_phrase(phrase) for phrase in phrases if phrase.strip()]

    def _normalize_phrase(self, phrase: str) -> str:
        normalized = phrase.strip()
        for source, target in self.phrase_normalization_map.items():
            normalized = normalized.replace(source, target)
        compact = normalized.replace(" ", "")
        if any(term in compact for term in self.reverse_positive_terms):
            return ""
        if "驱动" in compact and ("不足" in compact or "不动" in compact or "不了" in compact):
            return "驱动不足"
        if "价格" in compact and ("贵" in compact or "高" in compact or "溢价" in compact):
            return "价格偏高"
        if ("连接" in compact or "蓝牙" in compact) and "延迟" in compact:
            return "连接延迟"
        if ("连接" in compact or "蓝牙" in compact) and ("不稳" in compact or "断连" in compact or "卡顿" in compact):
            return "连接不稳"
        if ("噪声" in compact or "底噪" in compact or "杂音" in compact) and ("明显" in compact or "大" in compact or "有" in compact):
            return "噪声明显"
        if ("佩戴" in compact or "压耳" in compact or "夹头" in compact) and ("不适" in compact or "难受" in compact or "差" in compact):
            return "佩戴不适"
        if "做工" in compact and ("差" in compact or "粗糙" in compact):
            return "做工较差"
        if ("解析" in compact or "音质" in compact) and ("差" in compact or "一般" in compact):
            return "解析较差"
        return normalized

    def _is_negative_phrase(self, phrase: str) -> bool:
        joined = phrase.replace(" ", "")
        if any(term in joined for term in self.reverse_positive_terms):
            return False
        if joined in self.positive_neutral_terms:
            return False
        if any(term in joined for term in self.pain_negative_descriptors):
            return True
        if any(term in joined for term in self.protected_words):
            return True
        if any(term in joined for term in self.trigger_terms):
            return True
        return False

    def _is_complete_phrase(self, phrase: str) -> bool:
        parts = [part.strip() for part in phrase.split() if part.strip()]
        if len(parts) < 2:
            return False
        if any(part in self.incomplete_phrase_terms for part in parts):
            return False
        joined = "".join(parts)
        if joined in self.positive_neutral_terms:
            return False
        return True

    def _is_valid_phrase(self, phrase: str) -> bool:
        compact = phrase.replace(" ", "").strip()
        if not compact:
            return False
        if any(term in compact for term in self.reverse_positive_terms):
            return False
        if any(term in compact for term in self.invalid_phrase_terms):
            return False
        if compact in self.positive_neutral_terms:
            return False
        has_domain_term = any(term in compact for term in self.pain_aspect_terms)
        has_negative_term = any(term in compact for term in self.pain_negative_descriptors) or any(
            term in compact for term in self.trigger_terms
        )
        return has_domain_term and has_negative_term

    def _is_allowed_label(self, phrase: str) -> bool:
        if not phrase:
            return False
        if not self.standard_label_whitelist:
            return True
        return phrase.replace(" ", "") in self.standard_label_whitelist_compact

    def _extract_keywords(self, candidate_df, normalized_text_column: str):
        from collections import Counter

        phrase_counter = Counter()
        unigram_counter = Counter()
        source_column = "pain_point_clause_text" if "pain_point_clause_text" in candidate_df.columns else normalized_text_column
        for text in candidate_df[source_column].astype(str):
            tokens = self.extract_filtered_tokens(text)
            unigram_counter.update([token for token in tokens if token not in self.scene_terms and token not in self.weak_terms])
            phrase_counter.update(
                [
                    phrase
                    for phrase in self._extract_pain_phrases(tokens)
                    if phrase
                    and phrase not in self.weak_terms
                    and self._is_negative_phrase(phrase)
                    and self._is_complete_phrase(phrase)
                    and self._is_allowed_label(phrase)
                ]
            )
            unigram_counter.update(
                [
                    token
                    for token in tokens
                    if token not in self.scene_terms
                    and token not in self.weak_terms
                    and token not in self.bare_negative_terms
                    and token not in self.positive_neutral_terms
                ]
            )
        merged = phrase_counter + unigram_counter
        return merged.most_common(self.top_k)

    def build_wordcloud_frequencies(self, candidate_df, normalized_text_column: str = "normalized_text") -> dict[str, int]:
        from collections import Counter

        if candidate_df.empty:
            return {}
        counter = Counter()
        source_column = "pain_point_clause_text" if "pain_point_clause_text" in candidate_df.columns else normalized_text_column
        for text in candidate_df[source_column].astype(str):
            tokens = self.extract_filtered_tokens(str(text))
            phrases = self._extract_pain_phrases(tokens)
            counter.update(
                [
                    phrase
                    for phrase in phrases
                    if phrase.strip()
                    and phrase not in self.weak_terms
                    and self._is_negative_phrase(phrase)
                    and self._is_complete_phrase(phrase)
                    and self._is_allowed_label(phrase)
                    and not all(part in self.topic_terms for part in phrase.split())
                ]
            )
        # fallback to filtered unigrams only when phrase extraction is too sparse
        if not counter:
            for text in candidate_df[source_column].astype(str):
                tokens = self.extract_filtered_tokens(str(text))
                counter.update(
                    [
                        token
                        for token in tokens
                        if token not in self.scene_terms
                        and token not in self.weak_terms
                        and token not in self.bare_negative_terms
                        and token not in self.positive_neutral_terms
                    ]
                )
        return dict(counter)

    def build_pain_point_top_table(self, phrase_freq: dict[str, int], top_n: int = 20) -> list[dict[str, str | int]]:
        sorted_items = sorted(phrase_freq.items(), key=lambda item: item[1], reverse=True)[:top_n]
        rows = []
        for phrase, freq in sorted_items:
            aspect_bucket = "其他问题"
            compact_phrase = phrase.replace(" ", "")
            for bucket, keywords in self.pain_aspect_mapping.items():
                if any(keyword in compact_phrase for keyword in keywords):
                    aspect_bucket = bucket
                    break
            rows.append({"pain_point_phrase": phrase, "count": int(freq), "aspect_bucket": aspect_bucket})
        return rows

    @staticmethod
    def build_phrase_table(phrase_freq: dict[str, int]) -> list[dict[str, str | int]]:
        return [{"pain_point_phrase": phrase, "count": int(freq)} for phrase, freq in sorted(phrase_freq.items(), key=lambda item: item[1], reverse=True)]

    @staticmethod
    def build_aspect_distribution(top_rows: list[dict[str, str | int]]) -> list[dict[str, str | int | float]]:
        from collections import Counter

        counter = Counter(str(item.get("aspect_bucket", "其他问题")) for item in top_rows)
        total = sum(counter.values()) or 1
        return [
            {"aspect_bucket": aspect, "count": int(count), "ratio": round(count / total, 4)}
            for aspect, count in counter.most_common()
        ]

    def _cluster_within_aspect(self, aspect_df, text_column: str, normalized_text_column: str):
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

        if normalized_text_column not in candidate_df.columns:
            candidate_df[normalized_text_column] = candidate_df[text_column].astype(str).map(self._normalize_text)

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
