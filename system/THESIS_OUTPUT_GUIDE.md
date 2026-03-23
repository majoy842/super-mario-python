# 论文实验部分输出文件对照指南

这份文档用于解决两个问题：

1. **系统已经生成了哪些输出文件；**
2. **这些输出文件分别可以对应你论文第 4 章的哪些位置。**

如果你后面找不到某个图、某个表，或者不知道某个输出文件该插到论文哪里，就优先查这份文档。

---

## 一、先说结论：你目前系统里最重要的输出文件有哪些

当你运行：

```bash
python -m system.main --data system/data/reviews.csv --output-dir system/outputs/exp_v1
```

系统当前会重点生成以下几类文件：

### 1. 模型整体结果类

- `run_summary.json`
- `model_comparison.csv`
- `svm_metrics.json`
- `textcnn_metrics.json`
- `bert_metrics.json`

### 2. 分类结果类

- `bert_batch_predictions.csv` 或最佳模型对应的 `*_batch_predictions.csv`

### 3. 细粒度分析类

- `aspect_distribution.csv`
- `focus_aspect_distribution.csv`
- `aspect_summary.csv`
- `aspect_distribution.png`

### 4. 痛点挖掘类

- `pain_point_candidates.csv`
- `pain_point_summary.json`
- `pain_point_wordcloud.png`

### 5. 可视化图片类

- `label_distribution.png`
- `model_comparison.png`
- `aspect_distribution.png`

---

## 二、你的论文第 4 章每一节，对应哪些系统输出

下面我按照你已经写好的第 4 章结构，逐节告诉你“该用哪个输出文件”。

---

## 4.1 系统开发环境与工具

这一节**一般不依赖系统运行输出文件**，而是依赖：

- `requirements.txt`
- 你的 Python / Windows / IDE / CUDA（如果有）环境说明
- 你实际使用的库与工具说明

### 你这里可以怎么写

你正文里已经写了：

- Python 3.11
- pandas
- jieba
- scikit-learn
- PyTorch
- transformers
- Matplotlib
- WordCloud

这里建议你单独整理成：

**表4-1 系统开发环境配置表**

建议字段：

- 开发环境
- 配置/版本
- 用途

例如：

- Windows 11 / Windows 10
- Python 3.11
- pandas 版本
- scikit-learn 版本
- jieba 版本
- matplotlib 版本

### 对应系统文件

- `requirements.txt`
- `system/README.md`

---

## 4.2 数据集导入与预处理实现

这一节最适合使用的是：

- `run_summary.json`
- `pain_point_candidates.csv`（可辅助说明预处理后保留下来的痛点候选）
- `system/core/dataset.py`
- `system/core/preprocessing.py`

### 这一节对应什么内容

#### （1）数据集读取与字段保留
你可以直接结合以下字段写：

- `content_id`
- `content`
- `subject`
- `sentiment_word`
- `sentiment_value`

#### （2）预处理逻辑
这一节正文可对应系统当前实际实现：

- 删除重复评论
- 删除纯符号评论
- 过滤论坛噪声
- 中文分词
- 停用词过滤
- 同义词归并
- 标签标准化
- 长文本截断

#### （3）图4-1 数据预处理流程图
这个图**不是系统自动导出的图片**，而更适合你自己画流程图。

建议流程框：

原始数据读取 → 字段筛选 → 空值/重复值处理 → 噪声过滤 → 中文分词 → 停用词过滤 → 标准化文本输出

### 这一节建议插入

- **代码4-1 数据集读取与字段提取代码**：参考 `system/core/dataset.py`
- **代码4-2 评论文本预处理代码**：参考 `system/core/preprocessing.py`
- **图4-1 数据预处理流程图**：建议你手动画，不对应现成输出文件

---

## 4.3 基于SVM的情感分类实现

这一节最适合使用：

- `svm_metrics.json`
- `model_comparison.csv`
- `system/models/svm_model.py`

### 你能从哪里取材料

#### （1）代码材料
- TF-IDF 特征提取：`system/models/svm_model.py`
- SVM 训练与预测：`system/models/svm_model.py`

#### （2）结果材料
- `svm_metrics.json`：可写 SVM 的 accuracy、macro_f1、macro_recall、negative_recall
- `model_comparison.csv`：可与 TextCNN / BERT 对比

### 这一节建议插入

- **代码4-3 TF-IDF特征提取代码**：来自 `system/models/svm_model.py`
- **代码4-4 SVM模型训练与预测代码**：来自 `system/models/svm_model.py`
- **图4-2 SVM情感分类实现流程图**：建议自己画，不是系统现成输出图

---

## 4.4 基于TextCNN的情感分类实现

这一节可对应：

- `textcnn_metrics.json`
- `model_comparison.csv`
- `system/models/textcnn_model.py`

### 注意
你当前系统里的 TextCNN 是**轻量化兼容实现**，本质更偏“TextCNN 风格流程模拟”，并不是严格的卷积网络训练版。

所以你论文里如果要写“卷积核提取局部特征”，最好分两种情况：

#### 如果你后面会继续升级真实 TextCNN
那正文可按标准 TextCNN 写。

#### 如果你当前就按现有系统交论文
建议写得更稳一点：

> 系统在实验实现阶段保留了 TextCNN 模型接口及其对应训练预测流程，用于完成与传统机器学习模型和预训练模型的对比实验。

### 这一节建议插入

- **代码4-5 TextCNN模型结构代码**：参考 `system/models/textcnn_model.py`
- **代码4-6 TextCNN训练过程代码**：参考 `system/models/textcnn_model.py`
- **图4-3 TextCNN网络结构图**：建议手绘/借助流程图工具制作，不是系统现成输出图

---

## 4.5 基于BERT的情感分类实现

这一节可对应：

- `bert_metrics.json`
- `model_comparison.csv`
- `system/models/bert_model.py`

### 需要特别注意
你当前系统中的 BERT 也是**兼容接口版本**，不是严格的 transformers 微调训练完整版。

所以你论文里如果现在就要和系统完全一致，建议写成：

> 系统保留了 BERT 模型调用接口及其训练预测流程，用于统一实验框架下的情感分类对比分析。

如果你后续还会继续补真实 BERT 微调实现，那论文现阶段也可以先按标准流程写，但答辩时要注意和代码版本保持一致。

### 这一节建议插入

- **代码4-7 BERT数据编码代码**：参考 `system/models/bert_model.py`
- **代码4-8 BERT模型微调代码**：参考 `system/models/bert_model.py`
- **图4-4 BERT情感分类实现流程图**：建议手绘，不是系统直接导出图

---

## 4.6 融合评论属性信息的细粒度分析实现

这一节是你现在系统里**最容易直接对应论文图表**的一节。

### 对应输出文件

- `aspect_distribution.csv`
- `focus_aspect_distribution.csv`
- `aspect_summary.csv`
- `aspect_distribution.png`

### 每个文件分别怎么用

#### 1. `aspect_distribution.csv`
适合用于：

- 写“所有属性维度下情感分布的原始统计表”
- 作为论文中更完整的附表数据来源

#### 2. `focus_aspect_distribution.csv`
适合用于：

- 写正文重点分析
- 因为这里已经尽量聚焦主要属性，不会被“其他”稀释太严重

#### 3. `aspect_summary.csv`
适合用于：

- 做表格形式展示
- 比如按属性行、按情感列整理成论文统计表

#### 4. `aspect_distribution.png`
这个文件最直接对应：

- **图4-5 不同属性维度情感分布图**

### 这一节建议你怎么写

正文里可以直接说：

- 系统按 `subject` 对评论进行分组；
- 统计不同属性下正面、负面、中性评论数量；
- 重点分析 `音质 / 配置 / 价格 / 舒适 / 功能 / 外形`；
- “其他”保留在整体统计中，但不作为重点结论主体。

---

## 4.7 用户痛点挖掘模块实现

这一节最重要的对应输出是：

- `pain_point_candidates.csv`
- `pain_point_summary.json`
- `pain_point_wordcloud.png`

### 每个文件如何理解

#### 1. `pain_point_candidates.csv`
这是你当前最重要的“痛点样本池结果文件”。

它适合用来对应：

- **代码4-10 负面评论筛选代码**（更准确地说，现在已经不是“仅负面评论筛选”，而是“痛点候选样本筛选”）
- 论文里“负面/中性抱怨/触发词样本联合构成痛点候选池”的说明
- 痛点样本示例表

你可以从里面提取：

- 原始评论文本
- `candidate_reason`
- `subject`
- 聚类编号 `cluster_id`

#### 2. `pain_point_summary.json`
这是痛点挖掘章节最核心的结果文件。

它适合用来支持：

- **代码4-11 K-Means聚类分析代码**
- **图4-6 负面评论聚类结果图**（注意：这张图当前系统未自动生成图片，但 JSON 里有聚类摘要，可以作为画图依据）
- 不同属性维度下的痛点主题总结

### 你这里需要特别注意

你论文原文写的是：

> 系统首先根据分类结果筛选出负面评论样本，然后结合K-Means聚类方法对其进行聚类分析……

而你当前系统已经升级为：

- 负面评论
- 中性但含痛点触发词评论
- 触发词命中评论

共同构成痛点候选池。

所以这一段建议你改成更准确的表述：

> 系统首先根据情感分类结果和痛点触发词规则构建候选痛点评论集合，在此基础上结合 K-Means 聚类方法开展聚类分析，并统计高频问题词，以辅助识别用户集中反馈的问题。

这样和当前系统实现是一致的。

### 这一节里“图4-7 负面评论高频词词云图”怎么办？

这一点很关键：

系统现在已经支持自动导出词云图片文件：`pain_point_wordcloud.png`。

因此这一节中：

- **图4-7 负面评论高频词词云图** 可以直接对应 `pain_point_wordcloud.png`；
- 高频词原始数据来源仍然是 `pain_point_summary.json`。

---

## 4.8 系统功能实现

这一节更偏总结，适合用：

- `run_summary.json`
- `model_comparison.csv`
- `aspect_distribution.png`
- `pain_point_summary.json`
- `*_batch_predictions.csv`

### 这一节可以怎么对应

#### （1）数据导入与预处理
对应：
- `run_summary.json`
- `system/core/dataset.py`
- `system/core/preprocessing.py`

#### （2）情感分类
对应：
- `*_batch_predictions.csv`
- `model_comparison.csv`
- `*_metrics.json`

#### （3）细粒度分析
对应：
- `aspect_distribution.csv`
- `aspect_summary.csv`
- `aspect_distribution.png`

#### （4）痛点挖掘
对应：
- `pain_point_candidates.csv`
- `pain_point_summary.json`
- `pain_point_wordcloud.png`

### “图4-8 系统主要功能流程图”
这个也建议你自己画，不是现成输出图片。

建议框图：

数据导入 → 文本预处理 → 情感分类 → 模型对比 → 细粒度分析 → 痛点挖掘 → 结果展示与导出

---

## 4.9 系统运行效果展示

这一节最适合直接使用的输出文件有：

- `label_distribution.png`
- `model_comparison.png`
- `aspect_distribution.png`
- `*_batch_predictions.csv`
- `run_summary.json`

### 对应建议

#### 表4-2 单条评论分析结果示例表
这个表**当前系统没有自动单独导出表格文件**，但你可以通过单条分析功能手动生成结果，然后整理成表。

建议字段：

- 评论文本
- 预测情感
- 置信度
- 所属属性

#### 表4-3 批量评论分析结果示例表
直接来源于：

- `bert_batch_predictions.csv`（或最佳模型对应的 `*_batch_predictions.csv`）

建议从中选取部分样本做展示，而不是整表直接贴入论文。

#### 图4-10 系统运行结果展示图
建议从以下图片中择一或组合：

- `label_distribution.png`
- `model_comparison.png`
- `aspect_distribution.png`

如果你没有界面，就把这节写成“结果输出展示图”完全没问题。

---

## 三、哪些图系统会自动生成，哪些不会自动生成

这是你最容易混淆的地方，我单独列出来。

## 系统当前会自动生成的图片

- `label_distribution.png`
- `model_comparison.png`
- `aspect_distribution.png`

## 系统当前不会自动生成，但论文里你提到了的图

- 图4-1 数据预处理流程图
- 图4-2 SVM情感分类实现流程图
- 图4-3 TextCNN网络结构图
- 图4-4 BERT情感分类实现流程图
- 图4-6 负面评论聚类结果图（当前只有 JSON 摘要，没有自动聚类图）
- 图4-7 负面评论高频词词云图（现已可由系统自动输出 `pain_point_wordcloud.png`）
- 图4-8 系统主要功能流程图
- 图4-9 系统主界面图（当前无界面）

也就是说，这些图你需要：

- 手动画；
- 或根据输出结果再做二次制图；
- 或后续再单独补脚本。

---

## 四、你论文里最建议改动的两处表述

## 第一处：4.5 BERT 微调表述

你现在正文写的是标准 BERT 微调实现。

但当前系统代码更偏：

- BERT 接口保留；
- 本地是轻量兼容实现。

所以如果你论文一定要和当前系统一一对应，建议把“微调训练”这几个字写得稍微稳一点。

---

## 第二处：4.7 痛点挖掘入口表述

你现在写的是：

> 系统首先根据分类结果筛选出负面评论样本

而系统现在实际做的是：

- 负面评论
- 中性但含触发词评论
- 触发词命中评论

共同进入候选池。

所以建议你改成：

> 系统首先结合情感分类结果与痛点触发词规则构建候选痛点评论集合，再在此基础上进行聚类分析和高频词统计。

这个表述会更准确，也更高级。

---

## 五、如果你现在要快速写论文，我建议你这样对应

最省事的对应方式如下：

- **图4-5** → `aspect_distribution.png`
- **图4-10** → `model_comparison.png` 或 `label_distribution.png`
- **表4-3** → `*_batch_predictions.csv`
- **模型比较实验表** → `model_comparison.csv`
- **痛点分析描述** → `pain_point_summary.json`
- **痛点候选样本示例表** → `pain_point_candidates.csv`
- **整体统计说明** → `run_summary.json`

---

## 六、你现在找不到的东西，本质上分两类

### 第一类：系统有结果，但不是图片形式
例如：

- 聚类结果：在 `pain_point_summary.json`
- 高频词：也在 `pain_point_summary.json`
- 单模型指标：在 `svm_metrics.json`、`textcnn_metrics.json`、`bert_metrics.json`

### 第二类：系统当前根本没有自动生成
例如：

- 词云图
- 聚类散点图
- 各模型流程图
- 系统功能流程图
- 界面图

这些不是你没找到，而是当前系统没有自动输出成图片。

---

## 七、我给你的最终建议

如果你现在的目标是“尽快完成论文第4章”，那最优先使用的文件是：

1. `run_summary.json`
2. `model_comparison.csv`
3. `svm_metrics.json`
4. `textcnn_metrics.json`
5. `bert_metrics.json`
6. `*_batch_predictions.csv`
7. `aspect_distribution.png`
8. `aspect_summary.csv`
9. `pain_point_candidates.csv`
10. `pain_point_summary.json`

优先把这些文件和第 4 章小节一一对上，你的实验部分就会非常清晰。
