# 情感分析系统（简化版）

这是一个适合本科生学习的情感分析小项目，支持 Python 3.11。

## 功能
1. 数据预处理（统一使用 content,sentiment_value）
2. 多模型训练与对比（LR / NB / SGD）
3. 情感分析（支持标签 -1/0/1）
4. 可视化（柱状图 + 词云 + Loss/Accuracy 曲线）

## 数据格式要求
必须使用以下列名：
- `content`：评论文本
- `sentiment_value`：情感标签

标签支持：`1`（正向）、`0`（中性）、`-1`（负向）。

## 安装
```bash
pip install -r requirements.txt
```

## PyCharm 测试（推荐）
1. 打开项目根目录 `super-mario-python`
2. 解释器选择 Python 3.11 虚拟环境
3. 直接右键运行：
   - `sentiment_system/run_pipeline.py`（一键全流程）
   - 或者逐个运行 `sentiment_system/src/` 下 4 个模块

> 所有脚本都使用“基于文件位置”的路径，PyCharm 工作目录可任意。

默认原始数据文件：`sentiment_system/data/raw/earphone_sentiment.csv`

## 运行方式
### 方式1：一键运行
```bash
python sentiment_system/run_pipeline.py
```

### 方式2：分步骤运行
```bash
python sentiment_system/src/preprocess.py
python sentiment_system/src/train.py
python sentiment_system/src/analyze.py
python sentiment_system/src/visualize.py
```

### 方式3：运行测试脚本
```bash
python sentiment_system/tests/test_by_module.py
```

## 输出文件
会生成在 `sentiment_system/outputs/`：
- metrics.json（包含 Accuracy、Precision、Recall、F1-score 和最佳模型）
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png
- training_curves.png（Loss 曲线 + Accuracy 曲线）
- positive_wordcloud.png

版本更新日志见：`sentiment_system/CHANGELOG.md`


## 模型评估指标
- Accuracy（准确率）
- Precision（精确率）
- Recall（召回率）
- F1-score（F1值）

以上指标都会写入 `metrics.json`，并在训练完成与总流程完成时输出到控制台。
