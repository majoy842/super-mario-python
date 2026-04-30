# 系统版本日志

## v1.7.3

### 上个版本信息
- v1.7.2 已支持强制重训通道，但痛点词云仍可能混入“反向词/伪短语”（如 `价格不贵`、`好听`、`东西驱动不足`），标准标签收敛仍不够彻底。

### 本次更新内容
- 增加“反向词”拦截：`不贵/好听/满意/不错/可以/还行` 等正向或中性反转表达不再进入痛点短语。
- 新增短语标准化归一：将原始表达统一收敛到标准标签（如 `价格偏高`、`驱动不足`、`连接延迟`、`连接不稳`、`噪声明显`、`佩戴不适`、`做工较差`、`解析较差`）。
- 引入痛点标签白名单：词云与痛点短语统计仅保留白名单标签，白名单外短语直接过滤。
- pipeline 透传 `pain_point_label_whitelist` 到 `PainPointMiner`，保证配置可控。

### 本次涉及文件与修改内容
- `system/analysis/pain_points.py`：新增反向词过滤、短语归一化映射与白名单校验逻辑。
- `system/config.py`：`AnalysisConfig` 新增 `pain_point_label_whitelist` 默认标准标签集。
- `system/pipeline.py`：初始化 `PainPointMiner` 时新增白名单配置透传。
- `system/VERSION_LOG.md`：新增 v1.7.3 记录（置顶）。

## v1.7.2

### 上个版本信息
- v1.7.1 已改善痛点短语质量，但模型准备流程默认“先检测资产再决定是否训练”，不满足后台微调场景下“每次都重训”的需求。

### 本次更新内容
- 在 pipeline 增加强制重训通道：`prepare_models_for_inference(..., force_retrain=True)`，会跳过已存在资产并直接重训。
- 新增便捷方法 `force_retrain_models(...)`，用于后台直接触发“全量重训”。
- CLI 新增 `--force-retrain` 参数，可命令行一键强制重训后再执行全流程。
- `build_all_models.py` 改为默认走 `force_retrain=True`，确保离线构建始终刷新模型。
- 服务层新增 `force_retrain_models(system)`，便于后台或管理接口触发重训。

### 本次涉及文件与修改内容
- `system/pipeline.py`：新增 `force_retrain` 通道与 `force_retrain_models` 方法；`run()` 支持强制重训参数。
- `system/main.py`：新增 `--force-retrain` CLI 参数并透传到 `run()`。
- `build_all_models.py`：离线构建默认强制重训。
- `app/services/predict_service.py`：新增服务层重训入口。
- `system/VERSION_LOG.md`：新增 v1.7.2 记录（置顶）。

## v1.7.1

### 上个版本信息
- v1.7.0 已引入问题片段与结构化导出，但短语生成仍存在“相邻词机械拼接”噪声，出现 `不错价格` 等非自然痛点短语。

### 本次更新内容
- 收紧痛点短语合法性规则：短语必须同时包含“领域词（如价格/连接/驱动/佩戴/噪声等）”与“负向词（如贵/不足/不适/延迟/明显等）”，否则过滤。
- 新增短语级无效词过滤（`不错/还行/可以/意思/实在/没兴趣/提升很大/差点/可能贵` 等），抑制口语碎片拼接。
- 扩充同义归并规则，统一为更论文化标签：`价格高`、`驱动不足`、`连接延迟`、`噪声明显`、`佩戴不适`。
- 配置层同步补强 `weak_terms` 与 `pain_negative_descriptors`，提升词云纯度和痛点标签稳定性。

### 本次涉及文件与修改内容
- `system/analysis/pain_points.py`：新增 `_is_valid_phrase` 并约束 n-gram 保留逻辑；新增短语级无效词过滤。
- `system/config.py`：增强弱信息过滤词、负向描述词与短语归并映射。
- `system/VERSION_LOG.md`：新增 v1.7.1 记录（置顶）。

## v1.7.0

### 上个版本信息
- v1.6.9 已强化 GPU 训练校验，但痛点挖掘结果仍偏“负面词频统计”，对转折句、问题片段与维度分布的结构化表达不足。

### 本次更新内容
- 新增“负面问题片段”提取：在痛点候选池中增加 `pain_point_clauses` / `pain_point_clause_text`，优先保留“但是/不过/可惜/就是”等转折后的子句。
- 痛点关键词与词云统计改为优先基于 `pain_point_clause_text`，减少“不错/还行”这类非核心抱怨表达混入。
- 增加结构化导出：
  - `pain_point_clean_clauses.csv`（清洗后问题片段）
  - `pain_point_phrase_table.csv`（短语频次）
  - `pain_point_normalized_table.csv`（归并后痛点表）
  - `pain_point_aspect_distribution.csv`（维度占比）
  - `pain_point_cluster_summary.json`（聚类摘要）
- `pain_point_summary.json` 新增 `pain_point_aspect_distribution`，支持论文“图+表+文本”联动展示。

### 本次涉及文件与修改内容
- `system/analysis/pain_points.py`：新增转折句切分与负面子句抽取；新增短语表/维度分布构建函数；统计逻辑改为优先问题片段。
- `system/pipeline.py`：痛点挖掘阶段新增多份结构化输出与摘要字段。
- `system/VERSION_LOG.md`：新增 v1.7.0 记录（置顶）。

## v1.6.9

### 上个版本信息
- v1.6.8 已提供训练耗时估算，但用户反馈 BERT 实际在 CPU 上运行，未充分利用 GPU 资源。

### 本次更新内容
- 强化 BERT 设备选择逻辑：`TrainingConfig.device` 支持 `cuda/gpu/cpu/auto`，并在 `device="cuda"` 时强制校验 CUDA 可用性。
- 新增 `TrainingConfig.require_gpu`：当设为 `True` 且检测不到 CUDA 时，直接报错终止，避免“默默回退 CPU”导致长时间训练。
- 训练开始时打印实际 CUDA 设备名称；若最终走 CPU，输出明确告警信息，提示检查 CUDA/PyTorch 环境或显式设置 `training.device='cuda'`。

### 本次涉及文件与修改内容
- `system/config.py`：`TrainingConfig` 新增 `require_gpu`。
- `system/models/bert_model.py`：设备解析与 GPU 强制校验、CUDA 设备日志、CPU 回退告警。
- `system/VERSION_LOG.md`：新增 v1.6.9 记录（置顶）。

## v1.6.8

### 上个版本信息
- v1.6.7 已将 BERT 训练策略调整为效果优先，但运行摘要中仍缺少“按数据规模估算训练耗时”的信息，用户难以提前评估执行成本。

### 本次更新内容
- 在 pipeline 新增 `_estimate_training_time(processed_rows)`：基于样本规模、训练集比例、batch size、epoch、序列长度与设备类型给出各模型预计训练耗时。
- `run()` 返回值新增 `training_time_estimate` 字段，并写入 `run_summary.json`，便于 CLI / 报告直接查看。
- 估算信息包含：`train_rows`、`steps_per_epoch`、`total_steps`、每模型预计分钟数和总体预计分钟数。

### 本次涉及文件与修改内容
- `system/pipeline.py`：新增训练耗时估算方法，并在运行摘要输出估算结果。
- `system/VERSION_LOG.md`：新增 v1.6.8 记录（置顶）。

## v1.6.7

### 上个版本信息
- v1.6.6 已优化 BERT 训练耗时，但用户反馈当前目标是“模型性能优先”，需要进一步强化训练稳定性与泛化表现，而非单纯追求速度。

### 本次更新内容
- 在 BERT 训练中新增更稳健的优化器策略：引入 `weight_decay` 与线性 warmup 学习率调度（`get_linear_schedule_with_warmup`）。
- 新增梯度裁剪（`max_grad_norm`），降低训练不稳定和梯度爆炸风险。
- 将 TF32 开关改为显式配置项 `bert_tf32`，默认关闭以优先数值精度；需要提速时可手动开启。
- 保留 v1.6.6 的数据管线优化（批量分词与高效 collator），在不牺牲质量目标下减少无效 CPU 开销。

### 本次涉及文件与修改内容
- `system/config.py`：`TrainingConfig` 新增 `weight_decay`、`warmup_ratio`、`max_grad_norm`、`bert_tf32`。
- `system/models/bert_model.py`：接入 warmup scheduler、梯度裁剪、可配置 TF32 开关，并更新训练日志输出。
- `system/VERSION_LOG.md`：新增 v1.6.7 记录（置顶）。

## v1.6.6

### 上个版本信息
- v1.6.5 已完成 `system/system` 兼容层去重，但 BERT 训练阶段仍存在 CPU 侧分词与逐 batch 动态构造开销，GPU 利用率不稳定，训练耗时偏长。

### 本次更新内容
- 优化 BERT 数据管线：训练与预测改为“先批量分词一次，再 DataLoader 取已编码样本”，减少每个 batch 的重复 tokenizer 开销。
- 优化 padding 策略：`DataCollatorWithPadding` 改为在 collator 初始化时复用，并在 CUDA 下启用 `pad_to_multiple_of=8`，提升 Tensor Core 利用率。
- 增加 GPU 训练加速开关：在 CUDA 下启用 TF32 与 cuDNN benchmark。
- 优化 DataLoader worker 策略：当使用 GPU 且未显式配置 worker 时，自动使用 2 个 worker 并启用 `persistent_workers`。

### 本次涉及文件与修改内容
- `system/models/bert_model.py`：重构数据集与 collator、训练/预测前批量分词、CUDA 加速参数与 worker 策略。
- `system/VERSION_LOG.md`：新增 v1.6.6 记录（置顶）。

## v1.6.5

### 上个版本信息
- v1.6.4 已完成模型文本列路由修复，但 `system/system` 下仍保留多份镜像实现，后续修复容易出现“双份代码不同步”。

### 本次更新内容
- 将 `system/system` 的 analysis/core/models/exporters 镜像实现统一改为兼容层（shim），直接复用顶层 `system/*` 实现。
- 保留旧导入路径兼容性（`system.system.*` 仍可导入），同时消除重复业务逻辑维护点。
- 通过去重降低后续迭代风险：功能修复只需改一处顶层实现即可。

### 本次涉及文件与修改内容
- `system/system/analysis/aspect.py`、`system/system/analysis/evaluator.py`、`system/system/analysis/pain_points.py`：改为兼容层转发。
- `system/system/core/dataset.py`、`system/system/core/preprocessing.py`：改为兼容层转发。
- `system/system/models/base.py`、`system/system/models/bert_model.py`、`system/system/models/svm_model.py`、`system/system/models/textcnn_model.py`：改为兼容层转发。
- `system/system/exporters.py`：改为兼容层转发。
- `system/VERSION_LOG.md`：新增 v1.6.5 记录（置顶）。

## v1.6.4

### 上个版本信息
- v1.6.3 已修复列名与必需模型约束，但模型训练与推理仍统一使用 `normalized_text`，未按 BERT 与传统模型差异化路由。

### 本次更新内容
- 在 pipeline 增加 `_text_column_for_model(...)`：BERT 使用 `clean_text`，SVM/TextCNN 使用 `normalized_text`。
- `train_and_compare()` 改为按模型动态选择训练和预测文本列。
- `analyze_single()` 同时输出 `clean_text` 与 `normalized_text`，并标注 `inference_text_column`。
- `batch_analyze()` 按模型列预测，并新增 `prediction_text_column` 结果字段。
- 新增更新记录 `system/updates/2026-04-02_model_text_routing_issues.md`，归档问题与修复理由。

### 本次涉及文件与修改内容
- `system/pipeline.py`：新增模型文本列路由并改造训练/推理输入。
- `system/updates/2026-04-02_model_text_routing_issues.md`：新增路由问题排查记录。
- `system/VERSION_LOG.md`：新增 v1.6.4 记录（置顶）。

## v1.6.3

### 上个版本信息
- v1.6.2 已新增离线全模型构建入口，但仍存在“样例数据列名不一致”和“BERT 失败可被静默跳过导致流程表面成功”的风险。

### 本次更新内容
- 修复离线构建脚本列名，统一使用原始数据标准列：`content_id/content/subject/sentiment_word/sentiment_value`。
- 为训练配置新增必需模型约束：`required_models` 与 `allow_skip_models`。
- `train_and_compare()` 中若必需模型（默认 BERT）失败且不允许跳过，直接抛出错误终止流程，避免“跑完全程却没有 BERT”。
- `run()` 改为 `require_all_models=True`，确保全流程执行时会补齐缺失模型或显式失败。
- 样例数据 `system/data/reviews.csv` 改为标准列结构。
- 新增更新目录记录 `system/updates/2026-04-02_model_asset_issues.md`，归档本次排查问题与修复点。

### 本次涉及文件与修改内容
- `build_all_models.py`：修复列名为标准字段。
- `system/config.py`：`TrainingConfig` 新增 `required_models` 与 `allow_skip_models`。
- `system/pipeline.py`：新增必需模型失败即报错逻辑；`run()` 强制全模型准备。
- `system/data/reviews.csv`：重构为标准列样例数据。
- `system/updates/2026-04-02_model_asset_issues.md`：新增问题排查记录。
- `system/VERSION_LOG.md`：新增 v1.6.3 记录（置顶）。

## v1.6.2

### 上个版本信息
- v1.6.1 已修复保存/加载异常可观测性与短路逻辑，但缺少一键离线补齐脚本；同时 artifacts 目录位置需要统一到 `system/outputs/artifacts` 以便页面与排障一致。

### 本次更新内容
- 新增 `build_all_models.py` 离线构建入口：强制走 `prepare_models_for_inference(..., require_all_models=True)`，用于补齐缺失模型资产。
- `ExportConfig` 新增 `artifacts_dir`，默认统一为 `system/outputs/artifacts`。
- pipeline 的模型资产目录改为使用 `config.export.artifacts_dir`，避免隐式推导路径造成误判。
- app 服务层 `get_system(...)` 同步支持传入 `artifacts_dir`，保证 UI 与离线构建目录一致。

### 本次涉及文件与修改内容
- `build_all_models.py`：新增离线全模型资产构建脚本。
- `system/config.py`：`ExportConfig` 新增 `artifacts_dir`。
- `system/pipeline.py`：资产目录改为显式配置。
- `app/services/predict_service.py`：系统初始化新增 `artifacts_dir` 参数。
- `system/VERSION_LOG.md`：新增 v1.6.2 记录（置顶）。

## v1.6.1

### 上个版本信息
- v1.6.0 已补齐模型保存/加载与 UI 可用性控制，但 pipeline 中保存/加载异常仍有静默吞掉问题，且 `prepare_models_for_inference` 在部分场景会被已就绪模型短路，导致缺失模型不补齐。

### 本次更新内容
- 强化模型保存日志：`_save_model_if_possible` 改为返回 `(ok, message)`，并打印完整错误堆栈；保存失败会写入比较结果状态。
- 强化模型加载日志：`load_available_models` 对“未找到/加载成功/加载失败”均打印可追踪日志，并输出异常堆栈。
- 修复推理准备短路逻辑：`prepare_models_for_inference(..., require_all_models=False)` 新增 `require_all_models` 参数；当要求全模型时，若存在缺失模型将继续训练补齐。
- comparison 缓存状态新增 `missing` 标记，便于界面和排障快速识别缺失模型。

### 本次涉及文件与修改内容
- `system/pipeline.py`：保存与加载日志增强、状态回传增强、推理准备逻辑修复。
- `system/VERSION_LOG.md`：新增 v1.6.1 记录（置顶）。

## v1.6.0

### 上个版本信息
- v1.5.9 已完成 Streamlit 交互层和 app/services 分层，但模型资产管理仍不完整：未统一保存/加载权重，界面可能调用未就绪模型导致预测报错。

### 本次更新内容
- 为三类模型补齐持久化能力：
  - SVM 新增 `save/load/is_ready`；
  - TextCNN 新增 `save/load/is_ready`；
  - BERT 新增 `save/load/is_ready`（含 tokenizer 与 metadata）。
- 在 pipeline 里新增模型资产管理：训练后自动保存模型、启动时自动加载可用模型、暴露 `model_status()`。
- 优化 app 服务层与页面：
  - 页面只显示/提示模型可用状态；
  - 预测前强校验模型是否已加载；
  - 按钮执行增加异常兜底，避免 traceback 直接打断交互。
- 目标：从“仅有实验结果文件”升级为“可复用模型 + 可稳定在线推理”。

### 本次涉及文件与修改内容
- `system/models/base.py`：新增模型持久化接口约定。
- `system/models/svm_model.py`：新增保存/加载/就绪检测。
- `system/models/textcnn_model.py`：新增保存/加载/就绪检测与结构参数持久化。
- `system/models/bert_model.py`：新增保存/加载/就绪检测（model/tokenizer/metadata）。
- `system/pipeline.py`：新增模型保存、自动加载和可用状态检测逻辑。
- `app/services/predict_service.py`：新增模型就绪检测与预测前校验。
- `app/streamlit_app.py`：模型可用性展示、不可用提示和错误兜底。
- `system/VERSION_LOG.md`：新增 v1.6.0 记录（置顶）。

## v1.5.9

### 上个版本信息
- v1.5.8 已完成 pipeline 服务化与流程分层，但用户交互层尚未独立，缺少可直接演示的 Streamlit 页面与 app/services 分层调用。

### 本次更新内容
- 新增 `app/streamlit_app.py` 作为用户交互入口，实现输入、模式识别、参数区、分析按钮与结果 tabs 展示。
- 新增 `app/services/` 服务层：`mode_router.py`、`predict_service.py`、`aspect_service.py`、`pain_service.py`、`export_service.py`，实现 UI 与系统服务解耦。
- 页面按 comment 数量自动进入 single / small_batch / full_batch 流程，并通过 `st.session_state` 统一保存 `comments/mode/results`。
- 新增 `@st.cache_resource` 模型预热与缓存加载逻辑，避免每次交互重复加载重模型。
- 依赖补充 `streamlit` 到 `system/requirements.txt`。

### 本次涉及文件与修改内容
- `app/streamlit_app.py`：新增页面入口与交互流程。
- `app/services/*.py`：新增用户层服务封装。
- `app/__init__.py`、`app/services/__init__.py`：新增包初始化文件。
- `system/requirements.txt`：新增 `streamlit` 依赖。
- `system/VERSION_LOG.md`：新增 v1.5.9 记录（置顶）。

## v1.5.8

### 上个版本信息
- v1.5.7 主要补充了流程与论文模板说明，但尚未把“单条/小批量/大批量分层流程、离线训练在线推理、痛点阈值保护”真正落到系统代码。

### 本次更新内容
- 在 `pipeline.py` 新增业务路由能力 `route_by_volume(...)`，按评论数量自动进入单条、基础批量（2~9）或完整批量（>=10）流程。
- 新增模型推理准备与缓存逻辑：`prepare_models_for_inference(...)`、`_resolve_inference_model(...)`，避免界面阶段重复训练。
- 优化单条分析输出：支持可选主方面识别、返回模型名称与简短解释。
- 新增痛点挖掘阈值保护：样本量低于 `pain_point_min_samples` 时跳过挖掘并输出明确提示。
- 配置层新增 `pain_point_min_samples`（默认 10）。

### 本次涉及文件与修改内容
- `system/pipeline.py`：新增流程路由、模型准备缓存、阈值保护与单条输出增强。
- `system/config.py`：`AnalysisConfig` 新增 `pain_point_min_samples`。
- `system/VERSION_LOG.md`：新增 v1.5.8 记录（置顶）。

## v1.5.7

### 上个版本信息
- v1.5.6 已完成第5章（5.1~5.5）输出文件映射，但尚未把完整业务流程分层、开发顺序与“论文自然写法模板”沉淀为可直接复用内容。

### 本次更新内容
- 在 `THESIS_USAGE_GUIDE.md` 新增“完整业务流程与开发路线”章节，覆盖 A/B/C 三种输入规模流程（单条、2~9条、10条及以上）。
- 新增关键技术注意点：离线训练在线推理、模型缓存、规则法主方面识别、痛点挖掘阈值保护、后端统一生成图表。
- 新增推荐开发顺序（先服务后页面）和“论文怎么写才自然”的可直接套用段落模板。
- 更新 `README.md` 入口说明，明确该指南已包含业务流程与论文表达模板。

### 本次涉及文件与修改内容
- `system/THESIS_USAGE_GUIDE.md`：新增流程分层、开发路线、论文表达模板内容。
- `system/README.md`：补充指南能力范围说明。
- `system/VERSION_LOG.md`：新增 v1.5.7 记录（置顶）。

## v1.5.6

### 上个版本信息
- v1.5.5 已补充第4章（4.4~4.9）论文映射，但第5章（5.1~5.5）的测试与实验结果文件对应关系仍未明确标注。

### 本次更新内容
- 在 `THESIS_USAGE_GUIDE.md` 新增“第5章实验与测试（5.1~5.5）输出文件标注”章节。
- 逐条补充表5-1~表5-5、图5-1~图5-6的具体数据来源、输出文件路径和使用方式。
- 补充第5章一键落地顺序，明确每小节应优先使用的结果文件。
- 更新 `README.md` 文案，明确该指南同时覆盖第4章和第5章。

### 本次涉及文件与修改内容
- `system/THESIS_USAGE_GUIDE.md`：新增第5章（5.1~5.5）使用标注与文件映射。
- `system/README.md`：更新指南覆盖范围说明。
- `system/VERSION_LOG.md`：新增 v1.5.6 记录（置顶）。

## v1.5.5

### 上个版本信息
- v1.5.4 已修复模型对比图中文字体问题，但论文写作阶段仍缺少“章节-代码-结果文件”的一一对应说明。

### 本次更新内容
- 新增论文结果映射文档 `THESIS_USAGE_GUIDE.md`，按 4.4~4.9 小节标注可引用代码、可插入图表与对应输出文件。
- 补充“一次运行产出全部论文素材”的统一命令和最小可交付表图建议，降低论文整理成本。
- 在 `README.md` 增加该文档入口，方便直接跳转使用。

### 本次涉及文件与修改内容
- `system/THESIS_USAGE_GUIDE.md`：新增论文章节到系统产物的对照使用说明。
- `system/README.md`：新增论文使用指南入口。
- `system/VERSION_LOG.md`：新增 v1.5.5 记录（置顶）。

## v1.5.4

### 上个版本信息
- v1.5.3 已将模型对比改为折线图并补齐指标，但 Matplotlib 中文字体未统一配置，标题在部分环境会显示为方框。

### 本次更新内容
- 在 `ResultVisualizer` 中新增 Matplotlib 中文字体自动配置：优先显式字体路径，其次自动探测 Windows/macOS/Linux 常见中文字体。
- 全局设置 `axes.unicode_minus=False`，避免负号显示异常。
- 模型对比图的标题、坐标轴与图例统一应用中文字体属性，修复“中文方框字”问题。

### 本次涉及文件与修改内容
- `system/visualization.py`：新增 `_configure_plot_font` 并将字体属性应用到模型对比图文本元素。
- `system/VERSION_LOG.md`：新增 v1.5.4 记录（置顶）。

## v1.5.3

### 上个版本信息
- v1.5.2 已修复 `PainPointMiner` 参数兼容问题，但模型对比图仍使用柱状图，且图中未展示完整指标集合。

### 本次更新内容
- 将模型对比图改为折线图（含 marker），对齐“模型-指标趋势对比”展示方式。
- 扩充模型对比指标输出：新增 `macro_precision`、`weighted_precision`、`weighted_recall`，并在图中统一展示全部数值指标。
- 调整模型排序优先级为 `macro_f1 -> negative_recall -> weighted_f1 -> accuracy`，提升对不平衡数据的对比可读性。

### 本次涉及文件与修改内容
- `system/pipeline.py`：模型对比表补充 precision/recall 指标并更新排序规则。
- `system/visualization.py`：`plot_model_comparison` 改为折线图，展示全部指标并优化图例与坐标轴。
- `system/VERSION_LOG.md`：新增 v1.5.3 记录（置顶）。

## v1.5.2

### 上个版本信息
- v1.5.1 已完成痛点短语收口优化，但 `PainPointMiner` 构造参数在镜像路径文件中与 pipeline 调用不一致，存在 `weak_terms` 参数兼容风险。

### 本次更新内容
- 同步修复镜像路径 `system/system/analysis/pain_points.py`，使其 `PainPointMiner.__init__` 与主实现一致，补齐 `weak_terms` 及相关参数。
- 消除“pipeline 传参包含 `weak_terms`，而镜像类定义未声明”导致的潜在 `TypeError`。

### 本次涉及文件与修改内容
- `system/system/analysis/pain_points.py`：与主实现对齐，补齐构造函数参数与对应逻辑。
- `system/VERSION_LOG.md`：新增 v1.5.2 记录（置顶）。

## v1.5.1

### 上个版本信息
- v1.5.0 已修复日志顺序，但痛点词云中仍存在“不完整短语、同类痛点拆散、少量非痛点表达混入”的收敛问题。

### 本次更新内容
- 增强“完整痛点短语”约束：过滤明显半句表达（如“有点/没有/觉得/应该”等残缺片段），优先保留可解释的完整负面短语。
- 强化同义归并：补充驱动/连接/佩戴/噪声/价格等常见变体映射，减少频次被同义表达分散。
- 压低裸负面词与非痛点表达：对“没有/不足/不会/不行”等单词级弱解释项进行抑制，并过滤“没问题/不错”等正中性短语。
- 维持“词云 + Top表 + 痛点分类”输出链路，进一步提升汇报可读性。

### 本次涉及文件与修改内容
- `system/config.py`：新增 `incomplete_phrase_terms`、`bare_negative_terms`、`positive_neutral_terms` 并扩充 `phrase_normalization_map`。
- `system/analysis/pain_points.py`：新增完整短语过滤与正中性剔除逻辑，强化归一化后痛点短语筛选。
- `system/pipeline.py`：将新增短语过滤配置透传给 `PainPointMiner`。
- `system/VERSION_LOG.md`：新增 v1.5.1 记录并保持“新版本置顶”顺序。

## v1.5.0

### 上个版本信息
- v1.4.9 已完成短语归一化和痛点 Top 表导出，但用户指出日志顺序要求应为“新版本加在顶部”，且 v1.4.6 位置存在乱序。

### 本次更新内容
- 修正版本日志排序为顶部递减：最新版本固定加在顶部，旧版本向下排列。
- 调整 v1.4.6 / v1.4.7 / v1.4.8 / v1.4.9 的相对顺序，消除历史乱序。

### 本次涉及文件与修改内容
- `system/VERSION_LOG.md`：修复版本顺序并新增 v1.5.0 维护记录。

## v1.4.9

### 上个版本信息
- v1.4.8 已完成日志顺序与重复镜像文件清理，但痛点词云里仍有短语变体分散、弱信息词混入、主题词与痛点词混杂的问题。

### 本次更新内容
- 新增短语归一化映射（如“驱动力不足/推不动”归并为“驱动不足”），减少频次被同义变体拆散。
- 新增弱信息词与主题词压制策略，优先保留负面痛点短语，降低“场景词/口头词”干扰。
- 增加痛点 Top 表输出：在词云之外，导出 `pain_point_top_table.csv` 和 JSON 字段，便于直接交付业务结论。
- 新增按方面桶聚合（驱动类/连接类/音质类/佩戴类/价格类）用于痛点表解释。

### 本次涉及文件与修改内容
- `system/config.py`：新增 `weak_terms`、`topic_terms`、`phrase_normalization_map`、`pain_aspect_mapping` 等配置。
- `system/analysis/pain_points.py`：新增短语归一化、负面短语筛选、弱信息过滤、痛点 Top 表构建逻辑。
- `system/pipeline.py`：痛点挖掘阶段新增 `pain_point_top_table.csv` 导出与 JSON 汇总字段。
- `system/VERSION_LOG.md`：新增 v1.4.9 记录。

## v1.4.8

### 上个版本信息
- v1.4.7 修复了中文词云字体兼容问题，但用户指出版本日志更新顺序应按“从下往上（追加到末尾）”维护，同时 `system/system` 下仍有不必要的重复镜像文件。

### 本次更新内容
- 调整日志更新方式：本次版本记录按要求追加在文件末尾，后续沿用“从下往上更新（在底部新增）”。
- 删除不需要的重复镜像文件：`system/system/config.py`、`system/system/pipeline.py`、`system/system/visualization.py`。
- 为避免删除后导入异常，`system/system/__init__.py` 改为直接复用顶层 `system.config` 与 `system.pipeline`。

### 本次涉及文件与修改内容
- `system/system/config.py`：删除（去除重复镜像）。
- `system/system/pipeline.py`：删除（去除重复镜像）。
- `system/system/visualization.py`：删除（去除重复镜像）。
- `system/system/__init__.py`：改为从顶层 `system` 包导入 `SystemConfig` 和 `SentimentAnalysisSystem`。
- `system/VERSION_LOG.md`：新增 v1.4.8 记录，并按底部追加方式维护。

## v1.4.7

### 上个版本信息
- v1.4.6 已补齐历史日志，但词云渲染仍可能因中文字体缺失出现“方框字”。

### 本次更新内容
- 修复词云中文字体兼容问题：在 `WordCloud` 生成时增加字体路径解析逻辑，优先使用显式配置字体，其次自动探测常见中文字体（Windows/macOS/Linux 与项目内字体目录）。
- 导出配置新增 `wordcloud_font_path` 字段，支持在配置层显式指定字体绝对路径（如 Windows 的微软雅黑）。
- `ResultVisualizer` 构造函数增加字体参数传入，确保 pipeline 统一接入字体配置。

### 本次涉及文件与修改内容
- `system/visualization.py`：新增 `_resolve_wordcloud_font`，词云生成时按候选路径自动选择可用中文字体。
- `system/config.py`：`ExportConfig` 新增 `wordcloud_font_path`。
- `system/pipeline.py`：将 `wordcloud_font_path` 传入 `ResultVisualizer`。
- `system/VERSION_LOG.md`：新增 v1.4.7 记录。

## v1.4.6

### 上个版本信息
- v1.4.5 完成了痛点短语词云优化，但此前多次迭代未及时补齐版本日志，历史记录出现缺口。

### 本次更新内容
- 补齐最近未登记的版本变更记录，新增并完善 v1.4.4 / v1.4.5 两个版本条目。
- 统一恢复日志结构：每个版本均包含“上个版本信息 / 本次更新内容 / 本次涉及文件与修改内容”。

### 本次涉及文件与修改内容
- `system/VERSION_LOG.md`：补录并校正 v1.4.4、v1.4.5 历史记录，新增 v1.4.6 维护记录。

## v1.4.5

### 上个版本信息
- v1.4.4 已完成 `system/system` 对称目录补齐，但痛点词云仍偏向主题词，未充分聚焦“负面抱怨表达”。

### 本次更新内容
- 痛点候选池新增“抱怨模式”识别：除负面标签、触发词外，额外识别中性抱怨句（如不识别、断连、卡顿、发热等）。
- 词云统计单位升级为“痛点短语”优先：增加二元/三元短语统计，并加入“方面词 + 负向描述词”配对抽取。
- 停用词策略细化：新增场景词集合，并显式保留否定词、程度词、情绪词，避免痛点语义被打散。
- 输出层改为 `pain_point_phrase_wordcloud.png`，并提升词云参数以展示更多中频痛点短语。

### 本次涉及文件与修改内容
- `system/analysis/pain_points.py`：新增 complaint pattern、短语抽取与词云频次构建逻辑。
- `system/config.py`：新增场景词、保留词（否定/程度/情绪）、方面词与负向描述词配置。
- `system/pipeline.py`：痛点词云改为基于候选痛点短语统计，更新 `pain_point_summary.json` 与 `run_summary.json` 的策略说明。
- `system/visualization.py`：词云绘制参数增强（`max_words`、`min_font_size`、`collocations=False`）。
- `system/system/analysis/pain_points.py`、`system/system/config.py`、`system/system/pipeline.py`、`system/system/visualization.py`：同步镜像更新，保持双层目录一致。

## v1.4.4

### 上个版本信息
- v1.4.3 删除了 `system/resources/__init__.py`，但目录层级与用户预期的对称结构仍不一致。

### 本次更新内容
- 新增 `system/system/` 对称包结构，并同步核心模块（`analysis/core/models/resources/config/pipeline/main`）以兼容对称目录运行方式。

### 本次涉及文件与修改内容
- `system/system/__init__.py`：新增包导出。
- `system/system/analysis/*`、`system/system/core/*`、`system/system/models/*`：新增对称实现文件。
- `system/system/config.py`、`system/system/pipeline.py`、`system/system/main.py`、`system/system/exporters.py`、`system/system/visualization.py`：新增并对齐顶层实现。
- `system/system/resources/stopwords_zh.txt`：新增资源镜像。

## v1.4.3

### 上个版本信息
- v1.4.2 已完成配置模板对齐，但 `system/resources/__init__.py` 仍保留了不必要的包导出代码。

### 本次更新内容
- 按要求删除 `system/resources/__init__.py`，`resources` 目录仅保留静态资源文件（如停用词表）。

### 本次涉及文件与修改内容
- `system/resources/__init__.py`：删除。
- `system/VERSION_LOG.md`：新增 v1.4.3 记录。

## v1.4.2

### 上个版本信息
- v1.4.1 已完成单配置文件合并，但用户要求配置内容需与给定模板逐字段完全一致。

### 本次更新内容
- 将 `system/config.py` 按用户提供模板逐字段对齐（包含默认字段映射、预处理、训练、分析、导出全部配置项）。
- 保持其它模块引用不变，继续以该单一配置源驱动系统。

### 本次涉及文件与修改内容
- `system/config.py`：按指定模板做一致性对齐。
- `system/VERSION_LOG.md`：新增 v1.4.2 记录。

## v1.4.1

### 上个版本信息
- v1.4.0 同时保留了 `system/config.py` 与 `system/resources/config.py` 两套配置入口，存在重复维护成本。

### 本次更新内容
- 合并为单一配置文件：仅保留 `system/config.py`。
- 删除 `system/resources/config.py`，并调整 `system/resources/__init__.py` 改为从上级 `system.config` 引用 `SystemConfig`。

### 本次涉及文件与修改内容
- `system/config.py`：承载完整配置定义（Data/Preprocess/Training/Analysis/Export/SystemConfig）。
- `system/resources/config.py`：删除。
- `system/resources/__init__.py`：更新导入路径，避免双配置源。

## v1.4.0

### 上个版本信息
- v1.3.0 已完成输出文件治理与日志修复，但词云结果仍被大量虚词（如“的、了、是、和、我”）主导，情感与痛点词不够突出。

### 本次更新内容
- 新增中文停用词文件 `system/resources/stopwords_zh.txt`，并在配置与痛点挖掘中联动使用。
- 强化痛点关键词提取：过滤停用词、过滤短 token，并加入二元短语统计（如“物流 慢”“质量 差”）。
- 痛点挖掘新增“情感分层词云”：分别输出正向/负向词云与对应高频词，避免所有评论混合导致信息失真。
- 恢复模型训练容错：在缺少 `torch/transformers` 时自动跳过重模型，保证 SVM 路径可稳定跑通并输出分析结果。

### 本次涉及文件与修改内容
- `system/resources/stopwords_zh.txt`：新增中文停用词表。
- `system/resources/config.py`：新增 `analysis.extra_stopwords` 配置。
- `system/analysis/pain_points.py`：加入停用词过滤、最小词长过滤、短语统计能力。
- `system/pipeline.py`：新增正负词云与词频输出，训练阶段重模型失败自动跳过。

## v1.3.0

### 上个版本信息
- v1.2.0 完成了实验指南与字段适配文档，但仓库里仍包含运行产物文件（`outputs/sample_run/*.csv|*.json|*.txt`），不利于代码仓库整洁与版本审阅。
- v1.2.0 日志中对 v1.1.0/v1.0.0 的“文件级变更清单”记录不完整，部分历史信息在迭代中被弱化。

### 本次更新内容
- 移除 `system/outputs/sample_run/` 下全部运行产物，仅保留 `.gitkeep` 占位，后续输出由运行时动态生成。
- 补全并还原版本日志结构，明确每个版本的“上个版本信息 / 本次更新内容 / 本次涉及文件与修改内容”。
- 延续约定：后续每次迭代都必须记录文件级改动与影响说明。

### 本次涉及文件与修改内容
- `system/outputs/sample_run/.gitkeep`：新增占位文件，保留目录结构。
- `system/outputs/sample_run/*.csv|*.json|*.txt`：从仓库移除运行产物。
- `system/VERSION_LOG.md`：补全历史版本缺失描述并新增 v1.3.0 记录。

## v1.2.0

### 上个版本信息
- v1.1.0 已经完成真实数据字段适配和严格预处理，但尚未把“每次更新具体修改了哪些文件、每个文件改了什么”写成固定记录格式。
- 同时，上一版对实验推进顺序说明不够详细，用户仍然不清楚应该先运行哪个命令、再查看哪些结果文件。

### 本次更新内容
- 新增 `system/EXPERIMENT_GUIDE.md`，详细说明实验应如何一步一步推进，包括先检查命令行、再跑样例数据、再跑真实数据、再查看模型对比、再看细粒度分析和痛点挖掘。
- 更新 `system/README.md` 和根目录 `README.md`，增加实验指南入口。
- 约定从本版本开始，后续每次版本更新都必须在版本日志中写明“修改了哪些系统文件，以及每个文件分别做了什么修改”。

### 本次涉及文件与修改内容
- `system/EXPERIMENT_GUIDE.md`：新增详细实验推进文档。
- `system/README.md`：新增实验指南入口说明。
- `README.md`：新增系统实验指南入口。
- `system/VERSION_LOG.md`：新增 v1.2.0 版本记录，并明确后续版本日志的文件级变更记录要求。

## v1.1.0

### 上个版本信息
- v1.0.0 完成了系统基础代码搭建，但默认字段仍偏通用，尚未针对真实数据集 `content / subject / sentiment_value` 做充分适配。
- 旧版预处理对短噪声评论、纯符号评论、超长评论、无意义英文片段以及 `subject=其他` 的重点分析控制仍不够严格。

### 本次更新内容
- 将系统默认字段切换为真实数据集字段：`content_id`、`content`、`subject`、`sentiment_word`、`sentiment_value`。
- 新增更严格的数据预处理逻辑：按 `content` 去重、删除纯符号、删除过短文本、过滤典型噪声短语、截断超长文本、清理异常字符。
- 新增标签标准化逻辑，将 `-1 / 0 / 1` 统一映射为 `negative / neutral / positive`。
- 调整模型评估逻辑，突出 `Macro-F1`、`Macro-Recall`、`negative recall` 和混淆矩阵，弱化仅看 Accuracy 的风险。
- 调整细粒度分析逻辑：保留“其他”参与整体统计，但默认在重点属性分析中聚焦 `音质 / 配置 / 价格 / 舒适 / 功能 / 外形`。

### 本次涉及文件与修改内容
- `system/resources/config.py`：默认字段映射与分析配置增强。
- `system/core/preprocessing.py`：严格清洗、噪声过滤、标签归一化策略。
- `system/analysis/evaluator.py`：加入宏平均、负类召回与混淆矩阵指标。
- `system/analysis/aspect.py`：新增重点属性聚焦分析能力。
- `system/analysis/pain_points.py`：增强触发词与同义词归并的痛点挖掘。

## v1.0.0

### 版本概述
- 初始版本，完成耳机评论情感分析系统的基础代码搭建。
- 系统实现遵循既定设计思路：数据导入、文本预处理、情感分类、模型对比、细粒度分析、痛点挖掘、结果展示与导出。

### 本次涉及文件与修改内容
- `system/core/`：数据读取与基础预处理框架。
- `system/models/`：SVM/TextCNN/BERT 三类模型接口与实现骨架。
- `system/analysis/`：评估、细粒度分析、痛点挖掘基础模块。
- `system/pipeline.py`、`system/main.py`：流程编排与命令行入口。
