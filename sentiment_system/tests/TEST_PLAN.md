# 简单测试方案（PyCharm版）

## 环境
- Python 3.11
- 安装依赖：`pip install -r requirements.txt`

## 0. PyCharm 配置
1. 打开仓库根目录 `super-mario-python`
2. 设置解释器为 Python 3.11
3. Working directory 建议设为项目根目录

---

## 1. 分模块测试
在 PyCharm 中按顺序运行：

- `sentiment_system/src/preprocess.py`
- `sentiment_system/src/train.py`
- `sentiment_system/src/analyze.py`
- `sentiment_system/src/visualize.py`

检查 `sentiment_system/outputs/` 是否有：
- metrics.json
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png

## 2. 自动分步测试
运行：

```bash
python sentiment_system/tests/test_by_module.py
```

控制台出现“测试通过”表示核心流程正常。

## 3. 一键全流程
运行：

```bash
python sentiment_system/run_pipeline.py
```

控制台出现“完成！”并且输出图片存在即可。
