from __future__ import annotations

from collections import Counter
from dataclasses import asdict
import math
from pathlib import Path

from .analysis.aspect import FineGrainedAnalyzer
from .analysis.evaluator import ModelEvaluator
from .analysis.pain_points import PainPointMiner
from .config import SystemConfig
from .core.dataset import DatasetLoader
from .core.preprocessing import TextPreprocessor
from .exporters import ResultExporter
from .models import BERTSentimentModel, SVMSentimentModel, TextCNNSentimentModel
from .visualization import ResultVisualizer


class SentimentAnalysisSystem:
    def __init__(self, config: SystemConfig) -> None:
        self.config = config
        self.dataset_loader = DatasetLoader(config.data.data_path, config.data.encoding)
        self.preprocessor = TextPreprocessor(
            stopwords_path=config.preprocess.stopwords_path,
            custom_dict_path=config.preprocess.custom_dict_path,
            lowercase=config.preprocess.lowercase,
            remove_digits=config.preprocess.remove_digits,
            min_token_length=config.preprocess.min_token_length,
            min_text_length=config.preprocess.min_text_length,
            max_text_length=config.preprocess.max_text_length,
            drop_symbol_only=config.preprocess.drop_symbol_only,
            drop_meaningless_english=config.preprocess.drop_meaningless_english,
            noise_phrases=config.preprocess.noise_phrases,
            synonym_map=config.preprocess.synonym_map,
        )
        self.evaluator = ModelEvaluator()
        self.aspect_analyzer = FineGrainedAnalyzer(
            config.analysis.aspect_keywords,
            focus_aspects=config.analysis.focus_aspects,
            include_other_in_focus_analysis=config.analysis.include_other_in_focus_analysis,
        )
        wordcloud_stopwords = set(self.preprocessor.stopwords) | set(config.analysis.extra_stopwords)
        self.pain_point_miner = PainPointMiner(
            negative_labels=config.analysis.negative_labels,
            neutral_labels=config.analysis.neutral_labels,
            trigger_terms=config.analysis.pain_point_trigger_terms,
            forum_noise_terms=config.analysis.forum_noise_terms,
            synonym_map=config.analysis.pain_point_synonyms,
            top_k=config.analysis.pain_point_top_k,
            cluster_count=config.analysis.cluster_count,
            stopwords=wordcloud_stopwords,
            min_token_length=max(2, config.preprocess.min_token_length),
            scene_terms=config.analysis.scene_terms,
            preserve_negation_words=config.analysis.preserve_negation_words,
            preserve_degree_words=config.analysis.preserve_degree_words,
            preserve_emotion_words=config.analysis.preserve_emotion_words,
            pain_aspect_terms=config.analysis.pain_aspect_terms,
            pain_negative_descriptors=config.analysis.pain_negative_descriptors,
            weak_terms=config.analysis.weak_terms,
            topic_terms=config.analysis.topic_terms,
            phrase_normalization_map=config.analysis.phrase_normalization_map,
            pain_aspect_mapping=config.analysis.pain_aspect_mapping,
            incomplete_phrase_terms=config.analysis.incomplete_phrase_terms,
            bare_negative_terms=config.analysis.bare_negative_terms,
            positive_neutral_terms=config.analysis.positive_neutral_terms,
            standard_label_whitelist=config.analysis.pain_point_label_whitelist,
        )
        self.exporter = ResultExporter(config.export.output_dir)
        self.visualizer = ResultVisualizer(
            config.export.output_dir,
            config.export.chart_dpi,
            wordcloud_font_path=config.export.wordcloud_font_path,
        )
        self.models = {
            "SVM": SVMSentimentModel(),
            "TextCNN": TextCNNSentimentModel(),
            "BERT": BERTSentimentModel(),
        }
        self.artifacts_dir = Path(self.config.export.artifacts_dir)
        self._comparison_cache = None
        self._best_model_name: str | None = None

    @staticmethod
    def _text_column_for_model(model_name: str) -> str:
        return "clean_text" if model_name.upper() == "BERT" else "normalized_text"

    def _build_term_freq(self, df, label_column: str, target_labels: set[str], normalized_text_column: str = "normalized_text") -> dict[str, int]:
        counter = Counter()
        if df.empty:
            return {}
        for _, row in df.iterrows():
            if str(row[label_column]).lower() not in target_labels:
                continue
            tokens = self.pain_point_miner.extract_filtered_tokens(str(row.get(normalized_text_column, "")))
            counter.update(tokens)
            counter.update([f"{tokens[idx]} {tokens[idx + 1]}" for idx in range(len(tokens) - 1)])
        return dict(counter)

    def load_and_preprocess(self):
        df = self.dataset_loader.load()
        summary = self.dataset_loader.summarize(
            df,
            text_column=self.config.data.text_column,
            label_column=self.config.data.label_column,
            aspect_column=self.config.data.aspect_column,
        )
        processed_df = self.preprocessor.transform_dataframe(
            df,
            text_column=self.config.data.text_column,
            label_column=self.config.data.label_column,
            aspect_column=self.config.data.aspect_column,
            remove_duplicates=self.config.preprocess.remove_duplicates,
        )
        return df, processed_df, summary

    def train_and_compare(self, processed_df):
        import pandas as pd
        from sklearn.model_selection import train_test_split

        required_models = {name.upper() for name in getattr(self.config.training, "required_models", ["BERT"])}
        allow_skip_models = bool(getattr(self.config.training, "allow_skip_models", False))
        train_df, test_df = train_test_split(
            processed_df,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            stratify=processed_df[self.config.data.label_column].astype(str),
        )

        comparison_rows = []
        prediction_frames = {}
        failed_required_models: list[str] = []
        for model_name, model in self.models.items():
            try:
                text_col = self._text_column_for_model(model_name)
                model.train(train_df, text_col, self.config.data.label_column, self.config.training)
                predictions = model.predict(test_df[text_col].astype(str).tolist())
                pred_labels = [item.label for item in predictions]
                metrics = self.evaluator.evaluate(test_df[self.config.data.label_column].astype(str).tolist(), pred_labels)
                save_ok, save_msg = self._save_model_if_possible(model_name, model)
                comparison_rows.append(
                    {
                        "model": model_name,
                        "accuracy": metrics["accuracy"],
                        "macro_precision": metrics["macro_precision"],
                        "macro_f1": metrics["macro_f1"],
                        "macro_recall": metrics["macro_recall"],
                        "weighted_precision": metrics["weighted_precision"],
                        "weighted_recall": metrics["weighted_recall"],
                        "weighted_f1": metrics["weighted_f1"],
                        "negative_recall": metrics["negative_recall"],
                        "status": "ok" if save_ok else f"trained_but_save_failed: {save_msg}",
                    }
                )
                prediction_frame = test_df.copy()
                prediction_frame[f"{model_name}_prediction"] = pred_labels
                prediction_frame[f"{model_name}_confidence"] = [item.confidence for item in predictions]
                prediction_frames[model_name] = prediction_frame
                self.exporter.to_json(metrics, f"{model_name.lower()}_metrics.json")
            except Exception as exc:
                failure_reason = f"{type(exc).__name__}: {exc}"
                comparison_rows.append(
                    {
                        "model": model_name,
                        "accuracy": 0.0,
                        "macro_precision": 0.0,
                        "macro_f1": 0.0,
                        "macro_recall": 0.0,
                        "weighted_precision": 0.0,
                        "weighted_recall": 0.0,
                        "weighted_f1": 0.0,
                        "negative_recall": 0.0,
                        "status": f"skipped: {type(exc).__name__}",
                    }
                )
                self.exporter.to_json(
                    {"status": "skipped", "reason": failure_reason},
                    f"{model_name.lower()}_metrics.json",
                )
                if model_name.upper() in required_models:
                    failed_required_models.append(f"{model_name}: {failure_reason}")

        if failed_required_models and not allow_skip_models:
            error_message = "Required models failed to train/load -> " + " | ".join(failed_required_models)
            raise RuntimeError(error_message)

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            ["macro_f1", "negative_recall", "weighted_f1", "accuracy"],
            ascending=False,
        ).reset_index(drop=True)
        self.exporter.to_csv(comparison_df, "model_comparison.csv")
        self.visualizer.plot_model_comparison(comparison_df)
        self._comparison_cache = comparison_df.copy()
        self._best_model_name = str(comparison_df.iloc[0]["model"]) if len(comparison_df) else None
        return comparison_df, prediction_frames

    def _save_model_if_possible(self, model_name: str, model) -> tuple[bool, str | None]:
        target_dir = self.artifacts_dir / model_name.lower()
        try:
            model.save(str(target_dir))
            if not target_dir.exists():
                raise RuntimeError(f"{model_name} save finished but target dir not found: {target_dir}")
            return True, None
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"

    def load_available_models(self) -> dict[str, bool]:
        status: dict[str, bool] = {}
        for model_name, model in self.models.items():
            model_dir = self.artifacts_dir / model_name.lower()
            if not model_dir.exists():
                status[model_name] = model.is_ready()
                continue
            try:
                model.load(str(model_dir))
            except Exception:
                status[model_name] = model.is_ready()
                continue
            status[model_name] = model.is_ready()
        return status

    def model_status(self) -> dict[str, bool]:
        return {name: model.is_ready() for name, model in self.models.items()}

    def prepare_models_for_inference(self, processed_df=None, require_all_models: bool = False, force_retrain: bool = False):
        import pandas as pd

        if force_retrain:
            self._comparison_cache = None
            self._best_model_name = None
            if processed_df is None:
                _, processed_df, _ = self.load_and_preprocess()
            comparison_df, _ = self.train_and_compare(processed_df)
            return comparison_df

        loaded_status = self.load_available_models()
        ready_models = [name for name, ready in loaded_status.items() if ready]
        missing_models = [name for name, ready in loaded_status.items() if not ready]
        if ready_models and (not require_all_models or not missing_models):
            if self._best_model_name is None or self._best_model_name not in ready_models:
                self._best_model_name = "BERT" if loaded_status.get("BERT") else ready_models[0]
            if self._comparison_cache is None:
                self._comparison_cache = pd.DataFrame(
                    [
                        {
                            "model": name,
                            "accuracy": 0.0,
                            "macro_precision": 0.0,
                            "macro_f1": 0.0,
                            "macro_recall": 0.0,
                            "weighted_precision": 0.0,
                            "weighted_recall": 0.0,
                            "weighted_f1": 0.0,
                            "negative_recall": 0.0,
                            "status": "loaded" if loaded_status.get(name) else "missing",
                        }
                        for name in self.models
                    ]
                )
            return self._comparison_cache

        if self._comparison_cache is not None and self._best_model_name:
            return self._comparison_cache

        if processed_df is None:
            _, processed_df, _ = self.load_and_preprocess()
        comparison_df, _ = self.train_and_compare(processed_df)
        return comparison_df

    def force_retrain_models(self, processed_df=None):
        return self.prepare_models_for_inference(processed_df=processed_df, require_all_models=True, force_retrain=True)

    def _resolve_inference_model(self, model_name: str | None = None) -> str:
        target = model_name or self._best_model_name
        if target is None:
            raise RuntimeError(
                "No trained model is ready for inference. Run prepare_models_for_inference() first "
                "to keep training offline and inference online."
            )
        if target not in self.models:
            raise ValueError(f"Unsupported model_name: {target}")
        return target

    def analyze_single(self, text: str, model_name: str | None = None, with_aspect: bool = True) -> dict:
        model_name = self._resolve_inference_model(model_name)
        clean_text = self.preprocessor.clean_text(text)
        normalized_text = self.preprocessor.normalize(text)
        inference_text = clean_text if self._text_column_for_model(model_name) == "clean_text" else normalized_text
        prediction = self.models[model_name].predict([inference_text])[0]
        aspect = self.aspect_analyzer.infer_aspect(text) if with_aspect else None
        return {
            "original_text": text,
            "clean_text": clean_text,
            "normalized_text": normalized_text,
            "predicted_label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "model_used": model_name,
            "inference_text_column": self._text_column_for_model(model_name),
            "aspect": aspect,
            "explanation": f"模型判断为{prediction.label}，置信度{prediction.confidence:.3f}",
        }

    def batch_analyze(self, processed_df, model_name: str | None = None):
        model_name = self._resolve_inference_model(model_name)
        text_col = self._text_column_for_model(model_name)
        result_df = processed_df.copy()
        predictions = self.models[model_name].predict(result_df[text_col].astype(str).tolist())
        result_df["predicted_label"] = [item.label for item in predictions]
        result_df["prediction_confidence"] = [item.confidence for item in predictions]
        result_df["prediction_text_column"] = text_col
        self.visualizer.plot_label_distribution(result_df, "predicted_label")
        self.exporter.to_csv(result_df, f"{model_name.lower()}_batch_predictions.csv")
        return result_df

    def route_by_volume(
        self,
        comments: list[str],
        *,
        with_aspect: bool = True,
        with_aspect_distribution: bool = False,
        with_fine_grained: bool = True,
        with_pain_mining: bool = True,
        model_name: str | None = None,
    ) -> dict:
        import pandas as pd

        if not comments:
            raise ValueError("comments must not be empty")

        if len(comments) == 1:
            return {
                "mode": "single",
                "result": self.analyze_single(comments[0], model_name=model_name, with_aspect=with_aspect),
            }

        df = pd.DataFrame(
            {
                self.config.data.text_column: comments,
                self.config.data.aspect_column: ["其他"] * len(comments),
            }
        )
        df[self.config.data.label_column] = "neutral"
        processed_df = self.preprocessor.transform_dataframe(
            df,
            text_column=self.config.data.text_column,
            label_column=self.config.data.label_column,
            aspect_column=self.config.data.aspect_column,
            remove_duplicates=False,
        )
        prediction_df = self.batch_analyze(processed_df, model_name=model_name)
        result = {
            "mode": "basic_batch" if len(comments) < 10 else "full_batch",
            "predictions": prediction_df.to_dict(orient="records"),
        }

        if len(comments) < 10:
            if with_aspect_distribution:
                _, distribution_df, summary_df, focus_df = self.run_fine_grained_analysis(prediction_df)
                result["aspect_distribution"] = distribution_df.to_dict(orient="records")
                result["aspect_summary"] = summary_df.to_dict(orient="records")
                result["focus_aspect_distribution"] = focus_df.to_dict(orient="records")
            return result

        if with_fine_grained:
            _, distribution_df, summary_df, focus_df = self.run_fine_grained_analysis(prediction_df)
            result["aspect_distribution"] = distribution_df.to_dict(orient="records")
            result["aspect_summary"] = summary_df.to_dict(orient="records")
            result["focus_aspect_distribution"] = focus_df.to_dict(orient="records")
        if with_pain_mining:
            candidate_df, high_freq_terms, cluster_summary, wordcloud_path = self.run_pain_point_mining(prediction_df)
            result["pain_point"] = {
                "candidate_count": int(len(candidate_df)),
                "high_frequency_terms": high_freq_terms,
                "cluster_summary": cluster_summary,
                "wordcloud_path": str(wordcloud_path) if wordcloud_path else None,
            }
        return result

    def run_fine_grained_analysis(self, prediction_df):
        analyzed_df, distribution_df, summary_df, focus_distribution_df = self.aspect_analyzer.analyze(
            prediction_df,
            text_column=self.config.data.text_column,
            label_column="predicted_label",
            aspect_column=self.config.data.aspect_column,
        )
        self.exporter.to_csv(distribution_df, "aspect_distribution.csv")
        self.exporter.to_csv(summary_df, "aspect_summary.csv")
        self.exporter.to_csv(focus_distribution_df, "focus_aspect_distribution.csv")
        self.visualizer.plot_aspect_distribution(distribution_df)
        return analyzed_df, distribution_df, summary_df, focus_distribution_df

    def run_pain_point_mining(self, prediction_df):
        import pandas as pd

        if len(prediction_df) < self.config.analysis.pain_point_min_samples:
            self.exporter.to_json(
                {
                    "status": "skipped",
                    "reason": f"当前样本量不足（{len(prediction_df)}<{self.config.analysis.pain_point_min_samples}），不建议进行痛点挖掘",
                    "pain_point_top_table": [],
                    "high_frequency_terms": [],
                    "aspect_cluster_summary": [],
                },
                "pain_point_summary.json",
            )
            return prediction_df.iloc[0:0].copy(), [], [], None

        candidate_df, high_freq_terms, cluster_summary = self.pain_point_miner.mine(
            prediction_df,
            text_column=self.config.data.text_column,
            label_column="predicted_label",
            aspect_column=self.config.data.aspect_column,
        )
        if len(candidate_df) > 0:
            self.exporter.to_csv(candidate_df, "pain_point_candidates.csv")
            clause_columns = [
                column
                for column in [
                    self.config.data.text_column,
                    self.config.data.aspect_column,
                    "predicted_label",
                    "pain_point_clause_text",
                    "candidate_reason",
                    "trigger_terms",
                ]
                if column in candidate_df.columns
            ]
            if clause_columns:
                self.exporter.to_csv(candidate_df[clause_columns], "pain_point_clean_clauses.csv")

        phrase_word_freq = self.pain_point_miner.build_wordcloud_frequencies(candidate_df, normalized_text_column="normalized_text")
        phrase_table_rows = self.pain_point_miner.build_phrase_table(phrase_word_freq)
        if phrase_table_rows:
            self.exporter.to_csv(pd.DataFrame(phrase_table_rows), "pain_point_phrase_table.csv")
        pain_point_top_rows = self.pain_point_miner.build_pain_point_top_table(phrase_word_freq, top_n=20)
        aspect_distribution_rows = self.pain_point_miner.build_aspect_distribution(pain_point_top_rows)
        if pain_point_top_rows:
            self.exporter.to_csv(pd.DataFrame(pain_point_top_rows), "pain_point_top_table.csv")
            self.exporter.to_csv(pd.DataFrame(pain_point_top_rows), "pain_point_normalized_table.csv")
        if aspect_distribution_rows:
            self.exporter.to_csv(pd.DataFrame(aspect_distribution_rows), "pain_point_aspect_distribution.csv")
        combined_wordcloud_path = self.visualizer.plot_wordcloud(
            phrase_word_freq,
            "pain_point_phrase_wordcloud.png",
            max_words=150,
            min_font_size=8,
        )
        self.exporter.to_json({"cluster_summary": cluster_summary}, "pain_point_cluster_summary.json")

        self.exporter.to_json(
            {
                "high_frequency_terms": high_freq_terms,
                "aspect_cluster_summary": cluster_summary,
                "wordcloud_path": str(combined_wordcloud_path) if combined_wordcloud_path else None,
                "pain_phrase_term_freq": dict(sorted(phrase_word_freq.items(), key=lambda item: item[1], reverse=True)[:100]),
                "pain_point_top_table": pain_point_top_rows,
                "pain_point_aspect_distribution": aspect_distribution_rows,
            },
            "pain_point_summary.json",
        )
        return candidate_df, high_freq_terms, cluster_summary, combined_wordcloud_path

    def _estimate_training_time(self, processed_rows: int) -> dict:
        train_rows = max(1, int(processed_rows * (1 - self.config.training.test_size)))
        batch_size = max(1, int(self.config.training.batch_size))
        steps_per_epoch = max(1, math.ceil(train_rows / batch_size))
        epochs = max(1, int(self.config.training.epochs))
        total_steps = steps_per_epoch * epochs

        device_name = str(self.config.training.device)
        if device_name == "auto":
            try:
                import torch

                device_name = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device_name = "cpu"
        is_gpu = str(device_name).lower().startswith("cuda")

        max_length_factor = max(0.5, float(self.config.training.max_length) / 128.0)
        bert_step_seconds = 0.35 if is_gpu else 1.2
        textcnn_step_seconds = 0.06 if is_gpu else 0.12

        estimates_seconds = {
            "SVM": max(2.0, 0.0025 * train_rows + 1.5),
            "TextCNN": max(10.0, total_steps * textcnn_step_seconds),
            "BERT": max(25.0, total_steps * bert_step_seconds * max_length_factor),
        }

        estimate_rows = []
        for model_name, seconds in estimates_seconds.items():
            estimate_rows.append(
                {
                    "model": model_name,
                    "estimated_seconds": round(seconds, 1),
                    "estimated_minutes": round(seconds / 60.0, 2),
                    "assumption": f"rows={train_rows}, epochs={epochs}, batch_size={batch_size}, device={device_name}",
                }
            )

        return {
            "train_rows": train_rows,
            "steps_per_epoch": steps_per_epoch,
            "total_steps": total_steps,
            "device_assumption": device_name,
            "models": estimate_rows,
            "total_estimated_minutes": round(sum(estimates_seconds.values()) / 60.0, 2),
        }

    def run(self, force_retrain: bool = False):
        raw_df, processed_df, dataset_summary = self.load_and_preprocess()
        training_time_estimate = self._estimate_training_time(len(processed_df))
        comparison_df = self.prepare_models_for_inference(
            processed_df,
            require_all_models=True,
            force_retrain=force_retrain,
        )
        best_model = self._resolve_inference_model()
        batch_results = self.batch_analyze(processed_df, model_name=best_model)
        _, aspect_distribution, aspect_summary, focus_aspect_distribution = self.run_fine_grained_analysis(batch_results)
        pain_point_df, high_freq_terms, cluster_summary, wordcloud_path = self.run_pain_point_mining(batch_results)
        self.exporter.to_json(
            {
                "dataset_summary": asdict(dataset_summary),
                "preprocess_rules": {
                    "min_text_length": self.config.preprocess.min_text_length,
                    "max_text_length": self.config.preprocess.max_text_length,
                    "drop_symbol_only": self.config.preprocess.drop_symbol_only,
                    "drop_meaningless_english": self.config.preprocess.drop_meaningless_english,
                    "noise_phrases": self.config.preprocess.noise_phrases,
                    "stopwords_count": len(self.preprocessor.stopwords),
                    "wordcloud_min_token_length": self.pain_point_miner.min_token_length,
                    "wordcloud_allowed_pos_prefixes": list(self.pain_point_miner.allowed_pos_prefixes),
                },
                "pain_point_strategy": {
                    "candidate_pool": "negative + neutral_with_trigger + trigger_match + complaint_pattern",
                    "aspect_first_clustering": True,
                    "trigger_terms": self.config.analysis.pain_point_trigger_terms,
                    "wordcloud_unit": "negative-complaint phrases (bi-gram/tri-gram + aspect pairs)",
                },
                "best_model": best_model,
                "training_time_estimate": training_time_estimate,
                "pain_point_candidate_count": int(len(pain_point_df)),
                "high_frequency_terms": high_freq_terms,
                "cluster_summary": cluster_summary,
                "wordcloud_path": str(wordcloud_path) if wordcloud_path else None,
            },
            "run_summary.json",
        )
        return {
            "raw_rows": len(raw_df),
            "processed_rows": len(processed_df),
            "best_model": best_model,
            "training_time_estimate": training_time_estimate,
            "comparison": comparison_df.to_dict(orient="records"),
            "aspect_distribution_rows": len(aspect_distribution),
            "focus_aspect_distribution_rows": len(focus_aspect_distribution),
            "aspect_summary_rows": len(aspect_summary),
            "pain_point_candidate_count": len(pain_point_df),
            "output_dir": str(Path(self.config.export.output_dir).resolve()),
        }
