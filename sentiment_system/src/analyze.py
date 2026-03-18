"""
分析挖掘模块（Python 3.11）。

职责：
- 加载训练好的模型
- 对全量文本做预测并计算正向概率
- 统计正负文本中的高频词
- 输出预测明细和分析结果 JSON

你可以在 PyCharm 中单独运行本文件验证分析效果。
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import jieba
import joblib
import pandas as pd

# 轻量停用词（可按项目逐步扩展）
STOPWORDS = {"的", "了", "很", "也", "就", "都", "和", "太", "不", "有", "在", "是", "我", "还", "让"}


def tokenize(text: str) -> list[str]:
    """中文分词并过滤停用词、单字碎片。"""
    return [
        w.strip()
        for w in jieba.cut(text)
        if w.strip() and w not in STOPWORDS and len(w.strip()) > 1
    ]


def run_analysis(data_path: Path, model_path: Path, output_path: Path) -> dict:
    """
    对数据集执行预测分析并输出结构化结果。

    返回字段:
        sample_count: 样本总数
        pred_positive_ratio: 预测为正向的比例
        top_positive_words: 正向文本高频词
        top_negative_words: 负向文本高频词
    """
    df = pd.read_csv(data_path)
    model = joblib.load(model_path)

    # 模型预测标签与正向概率
    preds = model.predict(df["text"])
    probs = model.predict_proba(df["text"])[:, 1]
    df["pred"] = preds
    df["positive_prob"] = probs

    # 分别收集预测正/负样本中的词
    positive_tokens: list[str] = []
    negative_tokens: list[str] = []

    for _, row in df.iterrows():
        words = tokenize(row["text"])
        if row["pred"] == 1:
            positive_tokens.extend(words)
        else:
            negative_tokens.extend(words)

    result = {
        "sample_count": len(df),
        "pred_positive_ratio": float((df["pred"] == 1).mean()),
        "top_positive_words": Counter(positive_tokens).most_common(20),
        "top_negative_words": Counter(negative_tokens).most_common(20),
    }

    # 输出预测明细 + 汇总结果
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path.parent / "predictions.csv", index=False)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    return result


def main() -> None:
    """命令行入口：用于单模块测试。"""
    parser = argparse.ArgumentParser(description="情感分析挖掘")
    parser.add_argument("--data", default="sentiment_system/data/processed/reviews_clean.csv")
    parser.add_argument("--model", default="sentiment_system/models/sentiment_model.joblib")
    parser.add_argument("--output", default="sentiment_system/outputs/analysis.json")
    args = parser.parse_args()

    result = run_analysis(Path(args.data), Path(args.model), Path(args.output))
    print("分析完成:")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
