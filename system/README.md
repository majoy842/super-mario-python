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


## 中文字体说明

- 系统已在 `visualization.py` 中加入中文字体自动配置逻辑，会优先检测 Windows 常见字体（如 `Microsoft YaHei`、`SimHei`）以及 `system/assets/fonts/` 下的字体文件。
- 如果你的运行环境仍然出现中文乱码，可将中文字体文件（例如 `SimHei.ttf` 或 `msyh.ttc`）放入 `system/assets/fonts/` 后重新生成图表。


## 痛点挖掘策略说明

- 当前痛点挖掘不再只看强负面评论，而是使用“`negative + neutral_with_trigger + trigger_match`”三类评论共同构成候选痛点样本池。
- 系统会先做领域同义归并，再按 `subject` 属性维度分别聚类，避免把外形、价格、佩戴、搭配建议等内容全部混在一个聚类桶里。
- 默认使用更强的中文停用词与论坛噪声词过滤，降低“的、了、是、我、和”等虚词以及论坛灌水语句对高频词和聚类结果的污染。
