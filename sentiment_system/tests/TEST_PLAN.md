# 情感分析系统完整测试方案（Python 3.11 / PyCharm）

## 1. 测试目标

验证系统在 Python 3.11 环境下可稳定完成：

1. 数据预处理
2. 模型训练
3. 分析挖掘
4. 可视化生成
5. 全流程串联执行

---

## 2. 测试环境

- Python: 3.11.x
- IDE: PyCharm（Community 或 Professional 均可）
- 依赖安装命令：

```bash
pip install -r requirements.txt
```

---

## 3. 分模块测试用例

### 3.1 预处理模块（`src/preprocess.py`）

**输入**：`data/raw/reviews.csv`

**检查点**：
- 输出文件 `data/processed/reviews_clean.csv` 已生成
- 输出字段至少包含 `text`、`label`
- 样本数量大于 0
- 文本无空串

**执行方式**：
- 在 PyCharm 直接运行 `src/preprocess.py`
- 或运行 `tests/test_by_module.py` 的 `test_step_1_preprocess`

### 3.2 训练模块（`src/train.py`）

**前置条件**：预处理输出文件存在

**检查点**：
- 生成模型文件 `models/sentiment_model.joblib`
- 生成指标文件 `outputs/metrics.json`
- `metrics.json` 中包含 `accuracy`

**执行方式**：
- 在 PyCharm 直接运行 `src/train.py`
- 或运行 `tests/test_by_module.py` 的 `test_step_2_train`

### 3.3 分析模块（`src/analyze.py`）

**前置条件**：模型已训练完成

**检查点**：
- 生成 `outputs/analysis.json`
- 生成 `outputs/predictions.csv`
- `analysis.json` 中包含 `pred_positive_ratio`

**执行方式**：
- 在 PyCharm 直接运行 `src/analyze.py`
- 或运行 `tests/test_by_module.py` 的 `test_step_3_analyze`

### 3.4 可视化模块（`src/visualize.py`）

**前置条件**：分析结果已存在

**检查点**：
- 生成 `outputs/summary_bar.png`
- 生成 `outputs/top_words.png`
-（可选）生成 `outputs/positive_wordcloud.png`

**执行方式**：
- 在 PyCharm 直接运行 `src/visualize.py`
- 或运行 `tests/test_by_module.py` 的 `test_step_4_visualize`

---

## 4. 全流程回归测试

### 4.1 单命令执行

```bash
python sentiment_system/run_pipeline.py
```

**检查点**：
- 控制台打印流程完成信息
- `outputs/` 下核心产物全部存在（metrics、analysis、predictions、图片）

### 4.2 分步骤执行（推荐调试）

依次运行：

```bash
python sentiment_system/src/preprocess.py
python sentiment_system/src/train.py
python sentiment_system/src/analyze.py
python sentiment_system/src/visualize.py
```

---

## 5. 异常与边界测试建议

1. **缺字段测试**：删除输入 CSV 的 `label` 列，验证预处理应抛出缺字段异常。
2. **空文本测试**：构造空文本样本，验证预处理后会被过滤。
3. **最小数据量测试**：使用非常小的数据集，验证训练是否能给出明确报错或正常完成。
4. **模型缺失测试**：删除模型文件后运行分析模块，应给出明确异常。
5. **字体缺失测试**：无中文字体情况下运行可视化，确认系统仍能输出非词云图。

---

## 6. PyCharm 调试建议

1. 将工作目录设为仓库根目录（`super-mario-python`）。
2. 为四个模块分别创建 Run Configuration（便于单模块反复测试）。
3. 在关键位置打断点：
   - `preprocess.py`: 清洗后 DataFrame
   - `train.py`: `metrics` 内容
   - `analyze.py`: `result` 内容
   - `visualize.py`: 图片保存路径
4. 每次改动后先跑分模块，再跑全流程。

