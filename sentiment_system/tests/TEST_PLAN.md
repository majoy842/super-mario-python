# 简单测试方案（PyCharm版）

## 环境
- Python 3.11
- 安装依赖：`pip install -r requirements.txt`

## 0. PyCharm 配置
1. 打开仓库根目录 `super-mario-python`
2. 设置解释器为 Python 3.11
3. Working directory 任意（脚本内部已做绝对路径定位）

---

默认测试数据：`sentiment_system/data/raw/earphone_sentiment.csv`

## 1. 分模块测试
在 PyCharm 中按顺序运行：
- `sentiment_system/src/preprocess.py`
- `sentiment_system/src/train.py`
- `sentiment_system/src/analyze.py`
- `sentiment_system/src/visualize.py`

检查 `sentiment_system/outputs/` 是否有：
- metrics.json（包含 Accuracy、Precision、Recall、F1-score 等模型对比指标）
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png
- training_curves.png（Loss 曲线与 Accuracy 曲线）

## 2. 自动分步测试
```bash
python sentiment_system/tests/test_by_module.py
```
控制台出现“测试通过”表示核心流程正常。

## 3. 一键全流程
```bash
python sentiment_system/run_pipeline.py
```
控制台出现“完成！”并且输出图片存在即可。
