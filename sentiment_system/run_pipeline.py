from __future__ import annotations

from src.pipeline import SentimentAnalysisSystem


def main() -> None:
    summary = SentimentAnalysisSystem().run_all()
    print(
        f"流程完成。准确率: {summary['accuracy']:.4f}，"
        f"正向占比: {summary['pred_positive_ratio']:.2%}"
    )


if __name__ == "__main__":
    main()
