from __future__ import annotations

from dataclasses import asdict
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
        self.pain_point_miner = PainPointMiner(
            negative_labels=config.analysis.negative_labels,
            neutral_labels=config.analysis.neutral_labels,
            trigger_terms=config.analysis.pain_point_trigger_terms,
            forum_noise_terms=config.analysis.forum_noise_terms,
            synonym_map=config.analysis.pain_point_synonyms,
            top_k=config.analysis.pain_point_top_k,
            cluster_count=config.analysis.cluster_count,
        )
        self.exporter = ResultExporter(config.export.output_dir)
        self.visualizer = ResultVisualizer(config.export.output_dir, config.export.chart_dpi)
        self.models = {
            "SVM": SVMSentimentModel(),
            "TextCNN": TextCNNSentimentModel(),
            "BERT": BERTSentimentModel(),
        }

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

        train_df, test_df = train_test_split(
            processed_df,
            test_size=self.config.training.test_size,
            random_state=self.config.training.random_state,
            stratify=processed_df[self.config.data.label_column].astype(str),
        )

        comparison_rows = []
        prediction_frames = {}
        for model_name, model in self.models.items():
            model.train(train_df, "normalized_text", self.config.data.label_column, self.config.training)
            predictions = model.predict(test_df["normalized_text"].astype(str).tolist())
            pred_labels = [item.label for item in predictions]
            metrics = self.evaluator.evaluate(test_df[self.config.data.label_column].astype(str).tolist(), pred_labels)
            comparison_rows.append(
                {
                    "model": model_name,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "macro_recall": metrics["macro_recall"],
                    "weighted_f1": metrics["weighted_f1"],
                    "negative_recall": metrics["negative_recall"],
                }
            )
            prediction_frame = test_df.copy()
            prediction_frame[f"{model_name}_prediction"] = pred_labels
            prediction_frame[f"{model_name}_confidence"] = [item.confidence for item in predictions]
            prediction_frames[model_name] = prediction_frame
            self.exporter.to_json(metrics, f"{model_name.lower()}_metrics.json")

        comparison_df = pd.DataFrame(comparison_rows).sort_values(
            ["macro_f1", "negative_recall", "accuracy"], ascending=False
        ).reset_index(drop=True)
        self.exporter.to_csv(comparison_df, "model_comparison.csv")
        self.visualizer.plot_model_comparison(comparison_df)
        return comparison_df, prediction_frames

    def analyze_single(self, text: str, model_name: str = "BERT") -> dict:
        normalized_text = self.preprocessor.normalize(text)
        prediction = self.models[model_name].predict([normalized_text])[0]
        aspect = self.aspect_analyzer.infer_aspect(text)
        return {
            "original_text": text,
            "normalized_text": normalized_text,
            "predicted_label": prediction.label,
            "confidence": prediction.confidence,
            "probabilities": prediction.probabilities,
            "aspect": aspect,
        }

    def batch_analyze(self, processed_df, model_name: str = "BERT"):
        result_df = processed_df.copy()
        predictions = self.models[model_name].predict(result_df["normalized_text"].astype(str).tolist())
        result_df["predicted_label"] = [item.label for item in predictions]
        result_df["prediction_confidence"] = [item.confidence for item in predictions]
        self.visualizer.plot_label_distribution(result_df, "predicted_label")
        self.exporter.to_csv(result_df, f"{model_name.lower()}_batch_predictions.csv")
        return result_df

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
        candidate_df, high_freq_terms, cluster_summary = self.pain_point_miner.mine(
            prediction_df,
            text_column=self.config.data.text_column,
            label_column="predicted_label",
            aspect_column=self.config.data.aspect_column,
        )
        if len(candidate_df) > 0:
            self.exporter.to_csv(candidate_df, "pain_point_candidates.csv")
        self.exporter.to_json(
            {
                "high_frequency_terms": high_freq_terms,
                "aspect_cluster_summary": cluster_summary,
            },
            "pain_point_summary.json",
        )
        return candidate_df, high_freq_terms, cluster_summary

    def run(self):
        raw_df, processed_df, dataset_summary = self.load_and_preprocess()
        comparison_df, _ = self.train_and_compare(processed_df)
        best_model = comparison_df.iloc[0]["model"]
        batch_results = self.batch_analyze(processed_df, model_name=best_model)
        _, aspect_distribution, aspect_summary, focus_aspect_distribution = self.run_fine_grained_analysis(batch_results)
        pain_point_df, high_freq_terms, cluster_summary = self.run_pain_point_mining(batch_results)
        self.exporter.to_json(
            {
                "dataset_summary": asdict(dataset_summary),
                "preprocess_rules": {
                    "min_text_length": self.config.preprocess.min_text_length,
                    "max_text_length": self.config.preprocess.max_text_length,
                    "drop_symbol_only": self.config.preprocess.drop_symbol_only,
                    "drop_meaningless_english": self.config.preprocess.drop_meaningless_english,
                    "noise_phrases": self.config.preprocess.noise_phrases,
                },
                "pain_point_strategy": {
                    "candidate_pool": "negative + neutral_with_trigger + trigger_match",
                    "aspect_first_clustering": True,
                    "trigger_terms": self.config.analysis.pain_point_trigger_terms,
                },
                "best_model": best_model,
                "pain_point_candidate_count": int(len(pain_point_df)),
                "high_frequency_terms": high_freq_terms,
                "cluster_summary": cluster_summary,
            },
            "run_summary.json",
        )
        return {
            "raw_rows": len(raw_df),
            "processed_rows": len(processed_df),
            "best_model": best_model,
            "comparison": comparison_df.to_dict(orient="records"),
            "aspect_distribution_rows": len(aspect_distribution),
            "focus_aspect_distribution_rows": len(focus_aspect_distribution),
            "aspect_summary_rows": len(aspect_summary),
            "pain_point_candidate_count": len(pain_point_df),
            "output_dir": str(Path(self.config.export.output_dir).resolve()),
        }
