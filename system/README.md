# 情感分析系统代码说明

该目录按照“数据导入 → 文本预处理 → 情感分类 → 模型对比 → 细粒度分析 → 痛点挖掘 → 结果展示与导出”的设计思路组织实现。

## 目录结构

- `config.py`：系统配置。
- `core/`：数据导入与文本预处理。
- `models/`：SVM、TextCNN、BERT 三类模型接口实现。
- `analysis/`：模型评估、细粒度分析、痛点挖掘。
- `visualization.py`：图表可视化。
- `exporters.py`：结果导出。
- `pipeline.py`：系统总控流程。
- `main.py`：命令行启动入口。
- `VERSION_LOG.md`：版本日志。

## 快速运行

```bash
python -m system.main \
  --data path/to/reviews.csv \
  --text-column review_text \
  --label-column sentiment_label \
  --aspect-column aspect \
  --output-dir system/outputs
```

## 数据字段建议

建议输入数据至少包含以下字段：

- `review_text`：评论文本
- `sentiment_label`：情感标签（正面 / 负面 / 中性）
- `aspect`：属性标签（价格 / 音质 / 功能 / 舒适度 / 外观 等）
- `sentiment_words`：情感词（可选）
