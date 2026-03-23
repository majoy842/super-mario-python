# 第4章插图与表格落位清单

这份清单是给你直接排版论文用的。

你现在如果要写第4章，可以直接按下面的顺序，把“图/表标题 + 对应系统输出文件 + 使用方式”一一放进去。

---

## 一、表格落位清单

| 论文位置 | 标题建议 | 对应系统输出文件 | 使用建议 |
|---|---|---|---|
| 表4-1 | 系统开发环境配置表 | `requirements.txt` | 手动整理为环境、版本、用途三列，不是系统自动导出表。 |
| 表4-2 | 单条评论分析结果示例表 | 单条评论分析命令输出结果 | 手动选 3~5 条评论整理成表格，字段建议为“评论文本 / 预测情感 / 置信度 / 属性维度”。 |
| 表4-3 | 批量评论分析结果示例表 | `*_batch_predictions.csv` | 从最佳模型批量预测文件中截取 10~20 条代表性样本，不建议整表贴入正文。 |
| 模型比较实验表 | 三种模型分类效果对比表 | `model_comparison.csv` | 重点展示 `accuracy / macro_f1 / macro_recall / weighted_f1 / negative_recall`。 |
| 属性情感统计表 | 不同属性维度情感统计表 | `aspect_summary.csv` | 最适合放在4.6节，按属性行、情感列整理。 |
| 痛点样本示例表 | 候选痛点评论示例表 | `pain_point_candidates.csv` | 可选取不同 `subject` 和 `cluster_id` 的代表性样本。 |

---

## 二、图片落位清单

| 论文位置 | 标题建议 | 对应系统输出文件 | 是否系统自动生成 | 使用建议 |
|---|---|---|---|---|
| 图4-1 | 数据预处理流程图 | 无直接输出文件 | 否 | 建议手绘流程图：数据读取 → 清洗 → 去重 → 分词 → 停用词过滤 → 标准化。 |
| 图4-2 | SVM情感分类实现流程图 | 无直接输出文件 | 否 | 建议手绘：TF-IDF → SVM训练 → 测试集预测 → 指标输出。 |
| 图4-3 | TextCNN网络结构图 | 无直接输出文件 | 否 | 建议手绘或使用流程图软件绘制。 |
| 图4-4 | BERT情感分类实现流程图 | 无直接输出文件 | 否 | 建议手绘：分词编码 → 模型输入 → 分类输出。 |
| 图4-5 | 不同属性维度情感分布图 | `aspect_distribution.png` | 是 | 可直接插入论文4.6节。 |
| 图4-6 | 痛点聚类结果图 | `pain_point_summary.json` | 否（当前仅输出JSON摘要） | 需根据 `aspect_cluster_summary` 二次制图，或改成“聚类结果摘要表”。 |
| 图4-7 | 负面评论高频词词云图 | `pain_point_wordcloud.png` | 是 | 可直接插入论文4.7节。 |
| 图4-8 | 系统主要功能流程图 | 无直接输出文件 | 否 | 建议手绘：数据导入 → 预处理 → 分类 → 模型对比 → 细粒度分析 → 痛点挖掘 → 展示与导出。 |
| 图4-9 | 系统主界面图 / 结果输出示意图 | `label_distribution.png` / `model_comparison.png` / `aspect_distribution.png` | 是（若无界面） | 你没有图形界面时，建议改成“结果输出示意图”。 |
| 图4-10 | 系统运行结果展示图 | `model_comparison.png` 或 `label_distribution.png` 或 `aspect_distribution.png` | 是 | 任选一张最能体现系统完整效果的图片，或将多图拼接后展示。 |

---

## 三、每个小节最推荐放什么

### 4.1 系统开发环境与工具

最推荐放：

- **表4-1 系统开发环境配置表**

来源：

- `requirements.txt`
- 你自己的 Windows / Python / 开发工具信息

---

### 4.2 数据集导入与预处理实现

最推荐放：

- **代码4-1**：`system/core/dataset.py`
- **代码4-2**：`system/core/preprocessing.py`
- **图4-1**：手绘数据预处理流程图

辅助使用：

- `run_summary.json`

---

### 4.3 基于SVM的情感分类实现

最推荐放：

- **代码4-3**：`system/models/svm_model.py` 中 TF-IDF 特征提取部分
- **代码4-4**：`system/models/svm_model.py` 中 SVM 训练预测部分
- **图4-2**：手绘 SVM 实现流程图

辅助使用：

- `svm_metrics.json`
- `model_comparison.csv`

---

### 4.4 基于TextCNN的情感分类实现

最推荐放：

- **代码4-5**：`system/models/textcnn_model.py`
- **代码4-6**：`system/models/textcnn_model.py`
- **图4-3**：手绘 TextCNN 网络结构图

辅助使用：

- `textcnn_metrics.json`
- `model_comparison.csv`

---

### 4.5 基于BERT的情感分类实现

最推荐放：

- **代码4-7**：`system/models/bert_model.py`
- **代码4-8**：`system/models/bert_model.py`
- **图4-4**：手绘 BERT 流程图

辅助使用：

- `bert_metrics.json`
- `model_comparison.csv`

---

### 4.6 融合评论属性信息的细粒度分析实现

最推荐放：

- **代码4-9**：按 `subject` 分组统计部分（参考 `system/analysis/aspect.py`）
- **图4-5**：`aspect_distribution.png`
- **属性统计表**：`aspect_summary.csv`

如果正文篇幅有限：

- 图正文放 `aspect_distribution.png`
- 表附录放 `aspect_summary.csv`

---

### 4.7 用户痛点挖掘模块实现

最推荐放：

- **代码4-10**：痛点候选样本筛选（参考 `system/analysis/pain_points.py`）
- **代码4-11**：K-Means 聚类逻辑（参考 `system/analysis/pain_points.py`）
- **代码4-12**：词云生成逻辑（参考 `system/visualization.py`）
- **图4-7**：`pain_point_wordcloud.png`

补充说明：

- **图4-6** 当前没有系统自动生成图片文件；
- 如果你赶时间，可以把图4-6改成“聚类结果摘要表”，数据来源是 `pain_point_summary.json`。

---

### 4.8 系统功能实现

最推荐放：

- **图4-8**：手绘系统主要功能流程图
- 或者补充一张“结果输出示意图”

辅助使用：

- `run_summary.json`
- `model_comparison.csv`
- `aspect_distribution.png`
- `pain_point_summary.json`

---

### 4.9 系统运行效果展示

最推荐放：

- **表4-2**：单条评论分析结果示例表
- **表4-3**：批量评论分析结果示例表
- **图4-10**：`model_comparison.png` 或 `label_distribution.png` 或 `aspect_distribution.png`

如果你想让“运行效果展示”更完整，建议优先选：

1. `model_comparison.png`
2. `aspect_distribution.png`
3. `pain_point_wordcloud.png`

---

## 四、你现在最省事的最终排版方案

如果你想尽快完成第4章，我建议你直接用下面这套：

### 正文一定放

- 表4-1：系统开发环境配置表
- 图4-5：`aspect_distribution.png`
- 图4-7：`pain_point_wordcloud.png`
- 图4-10：`model_comparison.png`
- 模型比较实验表：`model_comparison.csv`
- 属性统计表：`aspect_summary.csv`

### 正文可选放

- 表4-2：单条评论分析结果示例表
- 表4-3：批量评论分析结果示例表
- 痛点样本示例表：`pain_point_candidates.csv`

### 建议手绘

- 图4-1
- 图4-2
- 图4-3
- 图4-4
- 图4-8

### 可以不单独画，改成表或文字说明

- 图4-6（聚类结果图）
- 图4-9（若无界面则改成结果输出示意图）

---

## 五、你现在就可以照着做的最终顺序

1. 先从 `model_comparison.csv` 做模型比较表；
2. 再把 `aspect_distribution.png` 放到图4-5；
3. 再把 `pain_point_wordcloud.png` 放到图4-7；
4. 再从 `aspect_summary.csv` 做属性统计表；
5. 再从 `*_batch_predictions.csv` 截部分样本做表4-3；
6. 最后挑一张 `model_comparison.png` 或 `label_distribution.png` 做图4-10。

这样你的第4章基本就可以完整落地。
