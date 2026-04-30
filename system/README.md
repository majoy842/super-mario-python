# 情感分析系统代码说明

该目录按照“数据导入 → 文本预处理 → 情感分类 → 模型对比 → 细粒度分析 → 痛点挖掘 → 结果展示与导出”的设计思路组织实现，并已针对你的真实数据字段做了默认适配。

## 默认字段映射

- `content_id`：评论编号
- `content`：评论文本
- `subject`：属性维度
- `sentiment_word`：情感词（用于解释和展示，不建议直接作为主模型特征）
- `sentiment_value`：情感标签（`-1 / 0 / 1`）

## 当前预处理策略

系统默认执行以下严格预处理：

- 按 `content` 去重；
- 删除纯符号评论；
- 删除长度过短评论；
- 截断超长评论；
- 清理异常空格与特殊字符；
- 过滤典型噪声短语，例如“帮顶”“同问”“围观”“马克”；
- 将 `sentiment_value` 统一映射为 `negative / neutral / positive`；
- 保留 `subject` 用于细粒度统计，并将“其他”从重点维度分析中单独区分。

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
  --id-column content_id \
  --text-column content \
  --label-column sentiment_value \
  --aspect-column subject \
  --sentiment-word-column sentiment_word \
  --output-dir system/outputs
```


## 实验推进说明

- 详细实验步骤与推荐顺序见 `EXPERIMENT_GUIDE.md`。
- 论文第4章（4.4~4.9）和第5章（5.1~5.5）如何对应代码、图、表与输出文件，见 `THESIS_USAGE_GUIDE.md`。
- 业务流程分层（单条/小批量/大批量）、开发顺序与论文自然表述模板，也已整理在 `THESIS_USAGE_GUIDE.md`。
