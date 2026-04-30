# 论文结果对照使用说明（第4章）

这份文档专门对应你给出的论文小节 **4.4 ~ 4.9**，告诉你：

1. 每一节该引用系统里的哪部分实现；  
2. 每一节可插入哪些“代码/图/表”；  
3. 应该运行什么命令来得到对应结果文件。

---

## 0. 先跑一次完整流程（统一产出）

> 建议先执行一次完整实验，再按章节挑选结果截图/表格。

```bash
python -m system.main \
  --data system/data/reviews.csv \
  --id-column content_id \
  --text-column content \
  --label-column sentiment_value \
  --aspect-column subject \
  --sentiment-word-column sentiment_word \
  --output-dir system/outputs/thesis_run
```

运行后重点看目录：

- `system/outputs/thesis_run/model_comparison.csv`
- `system/outputs/thesis_run/model_comparison.png`
- `system/outputs/thesis_run/*_metrics.json`
- `system/outputs/thesis_run/*_batch_predictions.csv`
- `system/outputs/thesis_run/aspect_distribution.csv`
- `system/outputs/thesis_run/aspect_distribution.png`
- `system/outputs/thesis_run/focus_aspect_distribution.csv`
- `system/outputs/thesis_run/pain_point_candidates.csv`
- `system/outputs/thesis_run/pain_point_summary.json`
- `system/outputs/thesis_run/pain_point_phrase_wordcloud.png`
- `system/outputs/thesis_run/pain_point_top_table.csv`
- `system/outputs/thesis_run/run_summary.json`

---

## 4.4 基于 TextCNN 的情感分类实现（怎么写、插什么）

### 可引用代码位置

- TextCNN 结构与前向计算：`system/models/textcnn_model.py`
- TextCNN 训练流程：`system/models/textcnn_model.py` 的 `train(...)`
- TextCNN 预测输出：`system/models/textcnn_model.py` 的 `predict(...)`

### 对应你文中的占位

- “代码4-5 TextCNN模型结构代码” → 截取 `TextCNNClassifier` 网络定义部分。
- “代码4-6 TextCNN训练过程代码” → 截取 `train(...)` 中数据编码、损失计算、优化器更新部分。
- “图4-3 TextCNN网络结构图” → 可用模型结构示意图（手绘/流程图工具）或由代码逻辑整理成框图。

### 结果支撑材料

- `textcnn_metrics.json`：TextCNN 的 accuracy / macro_f1 / macro_recall / weighted_f1 / negative_recall 等指标。
- `textcnn_batch_predictions.csv`：TextCNN 批量预测结果示例，可用于样本展示。

---

## 4.5 基于 BERT 的情感分类实现（怎么写、插什么）

### 可引用代码位置

- BERT 数据封装与 batch collate：`system/models/bert_model.py`
- BERT 微调训练：`system/models/bert_model.py` 的 `train(...)`
- BERT 预测与概率输出：`system/models/bert_model.py` 的 `predict(...)`

### 对应你文中的占位

- “代码4-7 BERT数据编码代码” → 截取 `ReviewDataset` + `BertBatchCollator`。
- “代码4-8 BERT模型微调代码” → 截取 `train(...)` 中 tokenizer/model 初始化、loss、optimizer、epoch 循环。
- “图4-4 BERT情感分类实现流程图” → 按“文本输入→分词编码→BERT→分类层→输出标签/概率”绘图。

### 结果支撑材料

- `bert_metrics.json`
- `bert_batch_predictions.csv`

---

## 4.6 融合评论属性信息的细粒度分析实现（subject）

### 可引用代码位置

- 维度识别与统计：`system/analysis/aspect.py`
- 流水线调用入口：`system/pipeline.py` 的 `run_fine_grained_analysis(...)`

### 对应你文中的占位

- “代码4-9 基于subject字段的分类统计代码” → 截取 `FineGrainedAnalyzer.analyze(...)` 的 groupby + pivot 统计。
- “图4-5 不同属性维度情感分布图” → 使用 `aspect_distribution.png`。

### 结果支撑材料

- `aspect_distribution.csv`
- `focus_aspect_distribution.csv`
- `aspect_summary.csv`
- `aspect_distribution.png`

---

## 4.7 用户痛点挖掘模块实现

### 可引用代码位置

- 痛点候选筛选、触发词匹配、短语提取、聚类：`system/analysis/pain_points.py`
- 流水线调用：`system/pipeline.py` 的 `run_pain_point_mining(...)`

### 对应你文中的占位

- “代码4-10 负面评论筛选代码” → 截取 `_build_candidate_pool(...)`（负面/中性抱怨筛选）。
- “代码4-11 K-Means聚类分析代码” → 截取 `_cluster_within_aspect(...)`。
- “代码4-12 高频词统计与词云生成代码” → 截取 `build_wordcloud_frequencies(...)` + `plot_wordcloud(...)` 调用处。
- “图4-6 负面评论聚类结果图” → 可由 `pain_point_summary.json` 的 `aspect_cluster_summary` 转成图表。
- “图4-7 负面评论高频词词云图” → `pain_point_phrase_wordcloud.png`。

### 结果支撑材料

- `pain_point_candidates.csv`
- `pain_point_summary.json`
- `pain_point_phrase_wordcloud.png`
- `pain_point_top_table.csv`

---

## 4.8 系统功能实现（总流程）

### 可引用代码位置

- 系统主控流程：`system/pipeline.py` 的 `run(...)`
- 命令行入口与参数：`system/main.py`
- 导出模块：`system/exporters.py`

### 建议写法

按“数据导入→预处理→模型训练/比较→细粒度分析→痛点挖掘→结果导出”顺序描述，并配 1 张流程图。

### 可插入图

- “图4-8 系统主要功能流程图” → 根据 `run(...)` 的调用顺序绘制。
- 若无界面，不放“系统主界面图”，可放“输出目录结构图”或“关键结果展示拼图”。

---

## 4.9 系统运行效果展示（表+图怎么选）

### 推荐最小可交付组合

1. **表4-2 单条评论分析结果示例表**  
   - 通过 CLI 参数 `--single-text "你的评论文本"` 得到 JSON 输出，整理成表格（原文、归一化文本、预测标签、置信度、属性）。

2. **表4-3 批量评论分析结果示例表**  
   - 使用 `*_batch_predictions.csv`（建议选最佳模型对应文件）。

3. **图4-10 系统运行结果展示图**（建议做 2x2 拼图）  
   - `model_comparison.png`（模型比较）  
   - `aspect_distribution.png`（属性分布）  
   - `pain_point_phrase_wordcloud.png`（痛点词云）  
   - `pain_point_top_table.csv`（可转柱状图）

### 章节结论可直接围绕

- 最佳模型及其关键指标（来自 `model_comparison.csv` 与 `*_metrics.json`）
- 各属性维度情感差异（来自 `aspect_*`）
- 主要用户痛点类别与高频表达（来自 `pain_point_*`）

---

## 附：你最常用的三条命令

```bash
# 1) 看参数
python -m system.main --help

# 2) 跑完整实验
python -m system.main --data system/data/reviews.csv --output-dir system/outputs/thesis_run

# 3) 单条评论分析示例
python -m system.main \
  --data system/data/reviews.csv \
  --output-dir system/outputs/thesis_run \
  --single-text "这款耳机连接不稳而且有底噪，价格也偏高"
```

---

# 第5章实验与测试（5.1 ~ 5.5）输出文件标注

你这次给的 5.1~5.5 文字，下面给你按“论文占位 → 具体输出文件/操作方式”逐条标注。

> 推荐统一输出目录：`system/outputs/thesis_run`，这样所有图表都从一个目录取，写论文不容易乱。

## 5.1 系统测试

### 可直接引用的系统运行证据

- 系统整体执行摘要：`run_summary.json`
- 模型对比总表：`model_comparison.csv`
- 批量预测结果：`*_batch_predictions.csv`
- 细粒度统计结果：`aspect_distribution.csv`、`focus_aspect_distribution.csv`
- 痛点结果：`pain_point_summary.json`、`pain_point_candidates.csv`

### “表5-1 系统功能测试结果表”怎么做

建议按“模块-测试项-输入-输出-是否通过”做表：

1. 数据导入：看 `run_summary.json` 里的 `raw_rows/processed_rows` 是否正常；
2. 预处理：抽查 `*_batch_predictions.csv` 中 `normalized_text`、`tokens` 等字段；
3. 情感分类：看 `model_comparison.csv` 和各 `*_metrics.json` 是否产出；
4. 细粒度分析：看 `aspect_distribution.csv` 是否有 `aspect_detected` 分组；
5. 痛点挖掘：看 `pain_point_summary.json`、`pain_point_top_table.csv` 是否生成；
6. 可视化展示：看 `model_comparison.png`、`aspect_distribution.png`、`pain_point_phrase_wordcloud.png` 是否存在。

---

## 5.2 模型对比实验设计

### “图5-1 模型对比实验流程图”建议内容

按下面流程绘图即可（对应 `pipeline.run()`）：

数据读取 → 预处理 → 训练/测试划分 → 三模型训练与预测 → 指标评估 → 对比排序 → 导出图表/表格

### “表5-2 各模型实验环境与参数设置表”数据来源

- 参数来源：`system/config.py` 中 `TrainingConfig`
  - 如 `test_size`、`batch_size`、`epochs`、`learning_rate`、`max_length`、`bert_model_name` 等；
- 模型实现来源：
  - SVM：`system/models/svm_model.py`
  - TextCNN：`system/models/textcnn_model.py`
  - BERT：`system/models/bert_model.py`

> 写论文时建议把“统一预处理条件”写在表下注释：三模型使用同一份清洗后数据和同一 train/test 划分策略。

---

## 5.3 不同模型实验结果分析

### “表5-3 SVM、TextCNN、BERT模型评价指标对比表”怎么取数

优先使用 `model_comparison.csv`，它已经汇总：

- `accuracy`
- `macro_precision`
- `macro_recall`
- `macro_f1`
- `weighted_precision`
- `weighted_recall`
- `weighted_f1`
- `negative_recall`

可直接复制到论文表格后保留 3~4 位小数。

### “图5-2 三种模型Accuracy与F1值对比图”

可直接使用：

- `model_comparison.png`（当前已是折线图，适合横向比较）。

如果你只想展示 Accuracy + F1，也可从 `model_comparison.csv` 二次筛列作图。

### “图5-3 三种模型混淆矩阵图”

从以下文件读取混淆矩阵：

- `svm_metrics.json` -> `confusion_matrix`
- `textcnn_metrics.json` -> `confusion_matrix`
- `bert_metrics.json` -> `confusion_matrix`

把三个矩阵分别画成热力图并排即可（2D list 就是画图输入）。

---

## 5.4 融合属性信息的实验分析

### “表5-4 不同属性维度下情感分布统计表”

优先使用：

- `aspect_summary.csv`（透视后的表格，最适合直接放论文表）。

如需突出核心维度（去掉“其他”），用：

- `focus_aspect_distribution.csv`

### “图5-4 各属性维度情感分布柱状图”

直接使用：

- `aspect_distribution.png`

如需按论文配色重新画图，数据源仍建议用 `aspect_distribution.csv`。

---

## 5.5 用户痛点挖掘结果分析

### “表5-5 负面评论高频问题词统计表”

建议优先使用：

- `pain_point_top_table.csv`（已经是“痛点短语+频次+类别”结构，最适合论文表）。

辅助说明可用：

- `pain_point_summary.json` 中 `high_frequency_terms`、`aspect_cluster_summary`。

### “图5-5 负面评论聚类结果图”

数据来源：

- `pain_point_summary.json` 的 `aspect_cluster_summary`。

你可以按“方面-簇大小”做柱状图，或按簇展示代表评论。

### “图5-6 负面评论词云图”

直接使用：

- `pain_point_phrase_wordcloud.png`

> 这张图已经是基于痛点短语过滤后的结果，不是原始高频虚词词云。

---

## 第5章一键落地建议（最省事）

1. 先跑一次完整命令（见本文最前面）；  
2. 打开 `system/outputs/thesis_run`；  
3. 按下面顺序拷贝到论文：  
   - 5.1：`run_summary.json` + 文件存在性检查结果  
   - 5.2：`config.py` 参数 + `pipeline.py` 流程图  
   - 5.3：`model_comparison.csv`、`model_comparison.png`、`*_metrics.json`  
   - 5.4：`aspect_summary.csv`、`aspect_distribution.png`  
   - 5.5：`pain_point_top_table.csv`、`pain_point_summary.json`、`pain_point_phrase_wordcloud.png`

---

# 完整业务流程与开发路线（可直接用于实现/答辩）

下面这部分按你给的 A/B/C 流程整理成“可落地服务层设计”，便于后续接 Streamlit 或其他前端。

## A. 单条评论流程（1条）

### 建议后端接口

- `analyze_single(text, with_aspect: bool = False)`

### 标准处理链

1. 文本预处理（normalize）  
2. 情感预测（loaded model inference）  
3. 若 `with_aspect=True`：主方面识别（规则法优先，使用 `aspect_keywords`）  
4. 返回结构化结果（原文、标签、方面、简短解释）

### 建议返回字段

- `original_text`
- `normalized_text`
- `predicted_label`
- `confidence`
- `aspect`（可选）
- `explanation`

---

## B. 小批量评论流程（2~9条）

### 建议后端接口

- `analyze_batch_basic(texts: list[str], with_aspect_distribution: bool = False)`

### 默认输出

- 每条评论预测结果表（DataFrame/CSV）
- 情感分布图（后端生成 PNG）

### 可选输出（勾选方面分布）

- 方面情感分布统计表
- 方面分布图

---

## C. 完整批量流程（>=10条）

### 建议后端接口

- `analyze_batch_full(df, with_fine_grained: bool = True, with_pain_mining: bool = True)`

### 默认步骤

1. 批量情感预测  
2. 情感分布统计

### 勾选细粒度分析时

- 统计不同方面下的情感分布（`aspect_distribution.csv`、`aspect_summary.csv`）

### 勾选痛点挖掘时

- 负面评论筛选  
- 触发词匹配  
- 高频词/短语统计  
- 痛点归类  
- 聚类摘要  
- 词云生成

### 输出建议

- 页面展示 + 文件导出按钮（CSV/JSON/PNG）

---

## 关键技术注意点（实现时必须做）

1. **界面不触发训练，只做推理**  
   - 训练离线完成，在线只加载已训练模型。

2. **模型加载缓存（特别是 BERT）**  
   - 服务启动时或首次请求加载，后续复用实例，避免重复加载。

3. **单条主方面识别优先规则法**  
   - 用 `aspect_keywords` + 归一化映射，稳定且快。

4. **痛点挖掘阈值保护**  
   - 建议：样本 `<10` 直接返回提示“样本量不足，不建议进行痛点挖掘”。

5. **图表统一后端生成，前端只展示**  
   - 统一复用已有 `visualization.py` 与导出逻辑，保证论文图和系统图一致。

---

## 推荐开发顺序（避免前端空转）

1. 先把 pipeline 拆成服务函数。  
2. 完成单条路径（预测 + 主方面）。  
3. 完成基础批量（预测表 + 情感分布图）。  
4. 接入细粒度方面分析。  
5. 接入痛点挖掘（Top表/词云/聚类摘要）。  
6. 最后再接 Streamlit 页面串联。

---

## 论文“怎么写更自然”可直接套用段落（答辩友好版）

### 1）系统流程描述模板

> 本系统按照“单条分析—基础批量分析—完整批量分析”的分层流程设计。对于单条评论，系统主要完成文本预处理、情感预测及可选的主方面识别；对于小批量评论，系统在输出逐条预测结果的同时给出整体情感分布；对于大批量评论，系统进一步执行方面级细粒度统计与痛点挖掘，从而实现从情感识别到问题定位的完整分析闭环。

### 2）工程实现描述模板

> 在工程实现上，系统采用“离线训练、在线推理”的部署思路，避免在界面交互阶段重复训练模型。服务层对模型实例进行缓存管理，降低重复加载开销。可视化图表统一由后端生成并导出，前端仅负责展示与交互，保证实验结果、论文图表和系统输出保持一致。

### 3）鲁棒性与可用性描述模板

> 为提升系统稳定性，痛点挖掘模块设置了样本量阈值保护：当样本规模不足时，系统返回“样本量不足”的提示而不强行输出聚类结果，避免结论失真。该机制使系统在不同规模数据输入场景下均具备较好的可用性与解释性。
