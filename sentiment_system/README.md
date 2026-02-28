# 情感分析系统（Sentiment Analysis System）

该模块提供从原始文本到可视化结果的完整流程，包含：

1. 数据预处理
2. 模型训练（TF-IDF + 逻辑回归）
3. 分析挖掘（情感占比、关键词统计）
4. 可视化（指标柱状图、关键词图、词云）
5. 面向代码调用的系统封装（`SentimentAnalysisSystem`）

## 目录结构

```text
sentiment_system/
├── data/
│   ├── raw/reviews.csv
│   └── processed/reviews_clean.csv
├── models/sentiment_model.joblib
├── outputs/
│   ├── metrics.json
│   ├── analysis.json
│   ├── predictions.csv
│   ├── run_summary.json
│   ├── summary_bar.png
│   ├── top_words.png
│   └── positive_wordcloud.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── analyze.py
│   ├── visualize.py
│   ├── pipeline.py
│   └── cli.py
└── run_pipeline.py
```

## 快速开始

```bash
pip install -r requirements.txt
python sentiment_system/run_pipeline.py
```

## CLI 用法

```bash
python -m sentiment_system.src.cli run-all
python -m sentiment_system.src.cli preprocess
python -m sentiment_system.src.cli train
python -m sentiment_system.src.cli analyze
python -m sentiment_system.src.cli visualize
python -m sentiment_system.src.cli predict --texts "服务很好下次再来" "质量太差很失望"
```

## 代码封装调用

```python
from sentiment_system.src import SentimentAnalysisSystem

system = SentimentAnalysisSystem()
summary = system.run_all()
print(summary)

preds = system.predict(["服务很好下次再来", "质量太差很失望"])
print(preds)
```

## 输入数据格式

CSV 至少包含两列：

- `text`: 评论文本
- `label`: 标签（1=正向，0=负向）

## 输出说明

- `metrics.json`: 模型准确率和分类指标
- `analysis.json`: 正向占比及高频词统计
- `predictions.csv`: 每条样本的预测标签与概率
- `run_summary.json`: 整体流水线摘要
- `summary_bar.png`: 准确率与正向占比可视化
- `top_words.png`: 正负面高频词可视化
- `positive_wordcloud.png`: 正面词云
