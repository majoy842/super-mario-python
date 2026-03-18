# 情感分析系统（简化版）

这是一个适合本科生学习的情感分析小项目，支持 Python 3.11。

## 功能
1. 数据预处理（兼容两种列名）
2. 多模型训练与对比（LR / NB / SGD）
3. 情感分析（支持标签 -1/0/1）
4. 可视化（柱状图 + 词云）

## 支持的数据列
- 方案A：`text,label`
- 方案B：`content,sentiment_value`（你提供的数据格式）

标签支持：`1`（正向）、`0`（中性）、`-1`（负向）。

## 安装
```bash
pip install -r requirements.txt
```

## PyCharm 测试（推荐）
1. 打开项目根目录 `super-mario-python`
2. 解释器选择 Python 3.11 虚拟环境
3. 直接右键运行：
   - `sentiment_system/run_pipeline.py`（一键全流程）
   - 或者逐个运行 `sentiment_system/src/` 下 4 个模块

> 所有脚本都使用“基于文件位置”的路径，PyCharm 工作目录可任意。

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
```

### 方式3：运行测试脚本
```bash
python sentiment_system/tests/test_by_module.py
```

## 输出文件
会生成在 `sentiment_system/outputs/`：
- metrics.json（包含多模型对比结果和最佳模型）
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png
- positive_wordcloud.png

版本更新日志见：`sentiment_system/CHANGELOG.md`
