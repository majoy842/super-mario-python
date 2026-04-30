from __future__ import annotations

from pathlib import Path

from system.config import DataConfig, ExportConfig, SystemConfig
from system.pipeline import SentimentAnalysisSystem


def main() -> None:
    config = SystemConfig(
        data=DataConfig(
            data_path=Path("system/data/reviews.csv"),
            text_column="content",
            label_column="sentiment_value",
            aspect_column="subject",
            id_column="content_id",
            sentiment_word_column="sentiment_word",
        ),
        export=ExportConfig(output_dir=Path("system/outputs/build_run"), artifacts_dir=Path("system/outputs/artifacts")),
    )
    system = SentimentAnalysisSystem(config)
    _, processed_df, _ = system.load_and_preprocess()
    comparison_df = system.prepare_models_for_inference(processed_df, require_all_models=True, force_retrain=True)
    print(comparison_df)
    print("artifacts dir:", system.artifacts_dir.resolve())


if __name__ == "__main__":
    main()
