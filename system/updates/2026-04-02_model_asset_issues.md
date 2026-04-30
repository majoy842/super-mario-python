# 模型资产与交互链路问题排查记录（2026-04-02）

## 发现的问题

1. **原始数据列名不一致**
   - 问题：`build_all_models.py` 使用了 `text/label`，与系统标准数据列 `content_id/content/subject/sentiment_word/sentiment_value` 不一致。
   - 影响：离线构建模型时可能读取不到正确列，导致流程报错或行为异常。

2. **BERT 可被“静默跳过”**
   - 问题：训练流程对异常统一 `except` 并继续，若 BERT 失败仍可能让流程“看起来成功”。
   - 影响：最终 artifacts 缺失 BERT，UI 端出现“BERT model has not been trained / 未加载”。

3. **全流程未强制校验必需模型**
   - 问题：`run()` 之前允许模型缺失仍继续执行。
   - 影响：用户跑完全流程后仍需手工 debug 为什么没有 BERT。

## 已实施修复

1. 修复离线构建入口列名为标准列。
2. 新增 `TrainingConfig.required_models` 与 `allow_skip_models`。
3. `train_and_compare()` 若必需模型失败且不允许跳过，直接抛出 `RuntimeError` 终止流程。
4. `run()` 强制 `require_all_models=True`，确保缺失模型会触发补训/报错，不再“悄悄缺失”。
5. 样例数据 `system/data/reviews.csv` 已改为标准列结构，避免后续误导。

