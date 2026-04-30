# 情感分析系统实验推进指南

这份指南按照“先准备数据，再检查预处理，再跑完整实验，再做结果分析”的顺序编写，适合你后续直接照着推进论文实验。

## 一、先明确你现在要做的三层实验

建议你把整个实验拆成三层：

1. **第一层：基础情感分类实验**
   - 比较 `SVM`、`TextCNN`、`BERT` 三个模型；
   - 重点观察 `Macro-F1`、`negative recall`、混淆矩阵；
   - 不要只看 Accuracy。

2. **第二层：细粒度属性统计实验**
   - 按 `subject` 统计不同属性维度下的情感分布；
   - 正文重点分析：`音质 / 配置 / 价格 / 舒适 / 功能 / 外形`；
   - `其他` 保留在总表中，但不要作为重点结论主体。

3. **第三层：痛点挖掘实验**
   - 先筛选负面评论；
   - 再做高频词统计；
   - 再做 K-Means 聚类；
   - 最后结合 `subject` 看不同属性下的主要负面问题。

---

## 二、正式实验前先准备什么

在运行系统前，请先准备以下内容。

### 1. 准备数据文件
你的数据建议至少包含以下列：

- `content_id`
- `content`
- `subject`
- `sentiment_word`
- `sentiment_value`

其中：

- `content` 是主文本；
- `subject` 用于细粒度分析；
- `sentiment_value` 是训练标签；
- `sentiment_word` 建议只用于结果展示，不直接作为主模型训练特征。

### 2. 放置数据文件
建议把你的真实数据文件放到例如以下位置：

- `system/data/reviews.csv`

如果后面你在 Windows 根目录 `C:\Users\ASUS\Desktop\system` 中继续开发，也建议保持类似目录结构：

- `data/`：原始数据
- `outputs/`：实验结果
- `logs/`：运行日志
- `models/`：训练后模型

### 3. 安装依赖
如果当前环境还没有安装依赖，先执行：

```bash
pip install -r requirements.txt
```

---

## 三、实验时先运行哪个，再运行哪个

这是你最需要的部分。后续建议严格按下面顺序进行。

## 第一步：先只检查命令行是否正常

先运行：

```bash
python -m system.main --help
```

这一步的目的不是做实验，而是确认：

- 系统入口能不能正常打开；
- 命令行参数是否识别正常；
- 当前 Python 环境没有明显导入错误。

如果这一步都失败，不要直接跑完整实验，先修环境。

---

## 第二步：先用样例数据跑通全流程

先运行：

```bash
python -m system.main --data system/sample_reviews.csv --output-dir system/outputs/sample_run
```

这一步的作用是：

- 验证系统完整流程能不能跑通；
- 确认输出文件是否生成；
- 先熟悉系统产物长什么样。

你应该重点查看输出目录里有没有这些文件：

- `run_summary.json`
- `model_comparison.csv`
- `svm_metrics.json`
- `textcnn_metrics.json`
- `bert_metrics.json`
- `*_batch_predictions.csv`
- `aspect_distribution.csv`
- `focus_aspect_distribution.csv`
- `pain_point_summary.json`

如果样例数据能跑通，再换成真实数据集。

---

## 第三步：再用真实数据跑第一次完整实验

确认真实数据路径后，运行：

```bash
python -m system.main \
  --data system/data/reviews.csv \
  --id-column content_id \
  --text-column content \
  --label-column sentiment_value \
  --aspect-column subject \
  --sentiment-word-column sentiment_word \
  --output-dir system/outputs/exp_v1
```

这一步是你的第一次正式实验。

运行完成后，建议你按以下顺序查看结果。

### 先看 `run_summary.json`
先确认：

- 原始数据量是多少；
- 预处理后数据量是多少；
- 最佳模型被系统选成了哪个；
- 负面评论有多少条。

### 再看 `model_comparison.csv`
重点看：

- `macro_f1`
- `macro_recall`
- `negative_recall`
- `accuracy`

建议你的论文比较表也按这个顺序展示，不要把 Accuracy 放在唯一核心位置。

### 再看每个模型的 `*_metrics.json`
这里面适合提取：

- 各类别 precision / recall / f1；
- confusion matrix；
- 是否存在大量把负面预测成中性的情况。

这一步是你后面写“类别不平衡影响模型识别效果”的关键依据。

---

## 第四步：再看细粒度分析结果

完整实验跑完之后，再看：

- `aspect_distribution.csv`
- `focus_aspect_distribution.csv`
- `aspect_summary.csv`

建议你这样使用：

### 1. 先看整体属性分布
先从 `aspect_distribution.csv` 观察所有属性，包括“其他”。

### 2. 再看重点属性
再从 `focus_aspect_distribution.csv` 只看：

- 音质
- 配置
- 价格
- 舒适
- 功能
- 外形

这一步更适合写进正文，因为不会被“其他”稀释掉。

### 3. 特别关注“舒适”
虽然样本少，但如果负面比例高，很适合写成论文亮点。

可以直接用类似表述：

> 不同属性维度下的情感分布存在差异，部分维度虽然样本量较少，但负面反馈更集中，说明这些维度更可能对应用户敏感体验问题。

---

## 第五步：最后做痛点挖掘分析

最后查看：

- `negative_comments.csv`
- `pain_point_summary.json`

建议分析顺序如下：

### 1. 先看负面评论样本
确认筛出来的负面评论是否合理。

### 2. 再看高频词
观察用户最常提到的问题词，比如：

- 音质差
- 佩戴不舒服
- 价格贵
- 做工一般
- 杂音
- 续航差

### 3. 再看聚类摘要
查看 K-Means 聚类后，每一类负面评论主要在抱怨什么。

### 4. 最后结合 `subject`
如果你后续继续扩展代码，可以进一步统计：

- 哪个 `subject` 下负面最多；
- 哪个 `subject` 的高频负面词最集中；
- 是否存在某些属性维度虽然样本少，但问题集中。

---

## 四、你后面写论文时建议怎么组织实验章节

建议实验章节也按系统输出顺序写。

### 4.1 数据集与预处理
这里写：

- 数据字段说明；
- 类别不平衡情况；
- `subject` 分布；
- 去重、噪声过滤、短文本删除、长文本截断等预处理策略。

### 4.2 模型对比实验
这里写：

- SVM、TextCNN、BERT 三类模型；
- Accuracy、Macro-F1、Macro-Recall、negative recall；
- confusion matrix 分析。

### 4.3 细粒度情感分析
这里写：

- 按 `subject` 的情感分布；
- “其他”类单独说明；
- 核心维度重点分析。

### 4.4 用户痛点挖掘
这里写：

- 负面评论筛选；
- 高频词统计；
- 聚类结果；
- 不同属性下的主要负面问题。

---

## 五、每次更新系统后你应该怎么做

以后每次系统代码更新，建议你都按下面顺序重新验证一次：

1. 先运行：
   ```bash
   python -m system.main --help
   ```
2. 再运行样例数据：
   ```bash
   python -m system.main --data sample_reviews.csv --output-dir outputs/sample_run
   ```
3. 再运行真实数据：
   ```bash
   python -m system.main --data data/reviews.csv --output-dir outputs/exp_vX
   ```
4. 对比本次 `model_comparison.csv` 和上个版本结果是否变化；
5. 在 `VERSION_LOG.md` 中记录：
   - 改了哪些文件；
   - 每个文件改了什么；
   - 实验结果有无变化；
   - 是否影响论文结论。

---

## 六、你下一步最建议做什么

如果你现在准备正式推进实验，最建议的顺序是：

1. 先把真实数据放进系统目录；
2. 跑第一版完整实验；
3. 检查 `model_comparison.csv`；
4. 看 `negative_recall` 是否过低；
5. 看 `focus_aspect_distribution.csv`；
6. 看 `pain_point_summary.json`；
7. 再决定是否要做下一轮模型优化。

如果你愿意，我下一步可以继续直接帮你做：

- **类别不平衡处理**（如 class weight、重采样）；
- **真实 TextCNN / BERT 训练版**；
- **词云图导出**；
- **按 subject 分组的负面痛点分析增强版**；
- **实验章节文字模板**。
