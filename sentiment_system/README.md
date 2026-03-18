# 情感分析系统（简化版）

这是一个适合本科生学习的简单情感分析小项目，支持 Python 3.11。

## 功能
1. 数据预处理
2. 模型训练（TF-IDF + 逻辑回归）
3. 情感分析（预测标签和关键词统计）
4. 可视化（柱状图 + 词云）

## 安装
```bash
pip install -r requirements.txt
```

## PyCharm 测试（推荐）
你可以直接在 PyCharm 中测试：

1. 打开项目根目录 `super-mario-python`
2. 解释器选择 Python 3.11 虚拟环境
3. 直接右键运行：
   - `sentiment_system/run_pipeline.py`（一键全流程）
   - 或者逐个运行 `sentiment_system/src/` 下的 4 个模块
4. 如果你想自动检查每一步，运行：
   - `sentiment_system/tests/test_by_module.py`

> 现在所有脚本都使用“基于文件位置”的路径，PyCharm 无论工作目录怎么设置都可以直接运行。

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

# 也可以在 PyCharm 里直接右键这些文件运行
```

### 方式3：运行测试脚本
```bash
python sentiment_system/tests/test_by_module.py
```

## 输入数据格式
CSV 两列：
- text：评论文本
- label：0 或 1

## 输出文件
会生成在 `sentiment_system/outputs/`：
- metrics.json
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png
- positive_wordcloud.png

更多测试说明见：`sentiment_system/tests/TEST_PLAN.md`
