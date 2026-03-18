# 情感分析系统（Sentiment Analysis System）

> 面向 **Python 3.11**，适合在 **PyCharm** 里按模块调试与测试。

该模块提供完整流程：

1. 数据预处理
2. 模型训练（TF-IDF + 逻辑回归）
3. 分析挖掘（情感占比、关键词统计）
4. 可视化（柱状图、关键词图、词云）

---

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
│   ├── summary_bar.png
│   ├── top_words.png
│   └── positive_wordcloud.png
├── src/
│   ├── preprocess.py
│   ├── train.py
│   ├── analyze.py
│   └── visualize.py
├── tests/
│   ├── test_by_module.py
│   └── TEST_PLAN.md
└── run_pipeline.py
```

---

## 安装依赖

```bash
pip install -r requirements.txt
```

---

## 在 PyCharm 中推荐的测试方式

### 方式 A：一键跑完整流程

```bash
python sentiment_system/run_pipeline.py
```

### 方式 B：按模块分块测试（推荐）

```bash
python sentiment_system/src/preprocess.py
python sentiment_system/src/train.py
python sentiment_system/src/analyze.py
python sentiment_system/src/visualize.py
```

### 方式 C：运行分模块测试脚本

```bash
python sentiment_system/tests/test_by_module.py
```

---

## 输入数据格式

CSV 至少包含两列：

- `text`: 评论文本
- `label`: 标签（1=正向，0=负向）

---

## 输出说明

- `metrics.json`: 模型准确率和分类指标
- `analysis.json`: 正向占比及高频词统计
- `predictions.csv`: 每条样本的预测标签与概率
- `summary_bar.png`: 准确率与正向占比可视化
- `top_words.png`: 正负面高频词可视化
- `positive_wordcloud.png`: 正面词云

---

## 完整测试方案

请查看：

- `sentiment_system/tests/TEST_PLAN.md`

其中包含：分模块测试、全流程回归、异常与边界测试、PyCharm 调试建议。
