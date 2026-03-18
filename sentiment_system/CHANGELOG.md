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
