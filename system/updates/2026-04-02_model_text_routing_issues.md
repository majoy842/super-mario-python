# 模型文本输入列路由修复记录（2026-04-02）

## 发现的问题

1. 训练阶段三模型统一使用 `normalized_text`，未按模型特性分流。
2. 在线单条/批量推理同样统一走 `normalized_text`，导致 BERT 丢失部分上下文信息。
3. 输出中虽已有 `content/clean_text/normalized_text`，但 pipeline 未明确声明“哪个模型用哪列”。

## 修复方案

1. 在 pipeline 增加 `_text_column_for_model(model_name)`：
   - BERT -> `clean_text`
   - SVM/TextCNN -> `normalized_text`
2. `train_and_compare()` 按模型动态选择训练与预测输入列。
3. `analyze_single()` 同时构造 `clean_text` 和 `normalized_text`，并按模型选择推理输入。
4. `batch_analyze()` 按模型选择输入列，并输出 `prediction_text_column` 便于核对。

## 预期收益

- BERT 保留更完整语序和上下文，减少过度规范化带来的语义损失。
- 传统模型继续使用 `normalized_text`，维持分词统计特征优势。
- 推理结果可追溯：明确知道每条预测使用了哪种文本列。

