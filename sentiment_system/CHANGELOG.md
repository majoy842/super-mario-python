# Changelog

## v1.0.0
- 初始版本：提供预处理、训练、分析、可视化完整流程。
- 支持 PyCharm 直接运行与分模块测试。

## v1.1.0
- 修复路径问题：改为基于文件位置定位，工作目录可任意。
- 改进文档与测试说明，强化 PyCharm 使用体验。

## v1.2.0
- 预处理支持两种数据列：`text,label` 与 `content,sentiment_value`。
- 训练升级为多方法对比：LogisticRegression、MultinomialNB、SGDClassifier(log_loss)。
- 指标输出包含模型对比结果与最佳模型选择。
- 分析支持三分类标签（-1/0/1），新增预测分布统计。

## v1.2.1
- 预处理移除 `text,label` 兼容逻辑，只保留 `content,sentiment_value`。
- 示例原始数据列名同步为 `content,sentiment_value`。
- 文档同步更新为统一第二种数据格式。

## v1.2.2
- 默认原始数据路径切换为 `sentiment_system/data/raw/earphone_sentiment.csv`。
- 运行脚本、预处理和测试脚本同步使用新数据集路径。

## v1.2.3
- 训练结果新增 Accuracy、Precision、Recall、F1-score 四个关键指标。
- `metrics.json` 改为输出更完整的模型评估结果。
- 训练脚本、总流程脚本和测试脚本同步适配新指标字段。

## v1.2.4
- 可视化新增 `training_curves.png`，包含 Loss 曲线和 Accuracy 曲线。
- 训练阶段记录 SGD 模型每轮 loss/accuracy，用于更直观展示训练过程。
- 测试脚本同步检查训练曲线图片是否生成。

## v1.2.5
- 继承可视化中文显示修复：Matplotlib 默认使用 `SimHei`，并关闭负号乱码。
- 词云默认使用 `C:/Windows/Fonts/simhei.ttf`，改善 Windows / PyCharm 下中文词云显示。
