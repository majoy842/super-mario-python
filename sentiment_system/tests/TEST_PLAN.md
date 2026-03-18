# 简单测试方案（本科生版本）

## 环境
- Python 3.11
- 安装依赖：`pip install -r requirements.txt`

## 1. 分模块测试
按顺序运行：

```bash
python sentiment_system/src/preprocess.py
python sentiment_system/src/train.py
python sentiment_system/src/analyze.py
python sentiment_system/src/visualize.py
```

检查输出目录 `sentiment_system/outputs/` 是否有：
- metrics.json
- analysis.json
- predictions.csv
- summary_bar.png
- top_words.png

## 2. 一键测试

```bash
python sentiment_system/tests/test_by_module.py
```

看到“测试通过”即成功。

## 3. 全流程运行

```bash
python sentiment_system/run_pipeline.py
```

看到“完成！”并且输出图片存在即可。
