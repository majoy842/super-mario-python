from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    data_path: Path
    id_column: str = "content_id"
    text_column: str = "content"
    aspect_column: str = "subject"
    sentiment_word_column: str = "sentiment_word"
    label_column: str = "sentiment_value"
    encoding: str = "utf-8-sig"


@dataclass
class PreprocessConfig:
    remove_duplicates: bool = True
    lowercase: bool = True
    remove_digits: bool = False
    stopwords_path: Path | None = Path("system/resources/stopwords_zh.txt")
    custom_dict_path: Path | None = None
    min_token_length: int = 1
    min_text_length: int = 5
    max_text_length: int = 512
    drop_symbol_only: bool = True
    drop_meaningless_english: bool = True
    noise_phrases: List[str] = field(
        default_factory=lambda: [
            "帮顶", "同问", "围观", "马克", "mark", "支持斑竹", "支持版主", "路过", "蹲个", "差评", "哈哈", "呵呵", "看看就很开心", "起码我是这样", "沙发", "学习一下", "学习了", "顶", "支持",
        ]
    )
    synonym_map: Dict[str, str] = field(
        default_factory=lambda: {
            "硌耳朵": "佩戴不适", "夹耳": "佩戴不适", "不舒服": "佩戴不适", "戴久了疼": "佩戴不适", "底噪": "噪声", "电流声": "噪声", "杂音": "噪声", "推不动": "驱动不足", "直推": "驱动不足", "太贵": "价格偏高", "溢价": "价格偏高", "不值": "价格偏高", "体积大": "体积过大", "太大": "体积过大", "笨重": "体积过大", "太重": "体积过大", "延迟": "连接延迟", "卡顿": "连接不稳",
        }
    )


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 128
    device: str = "auto"
    require_gpu: bool = False
    num_workers: int = 0
    use_amp: bool = True
    use_class_weight: bool = True
    textcnn_vocab_size: int = 20000
    textcnn_embedding_dim: int = 128
    textcnn_num_filters: int = 128
    textcnn_kernel_sizes: List[int] = field(default_factory=lambda: [3, 4, 5])
    textcnn_dropout: float = 0.5
    bert_model_name: str = "bert-base-chinese"
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    bert_tf32: bool = False
    required_models: List[str] = field(default_factory=lambda: ["BERT"])
    allow_skip_models: bool = False


@dataclass
class AnalysisConfig:
    negative_labels: List[str] = field(default_factory=lambda: ["negative", "负面", "neg", "-1", -1])
    neutral_labels: List[str] = field(default_factory=lambda: ["neutral", "中性", "0", 0])
    positive_labels: List[str] = field(default_factory=lambda: ["positive", "正面", "1", 1])
    supported_aspects: List[str] = field(default_factory=lambda: ["音质", "配置", "价格", "舒适", "功能", "外形", "其他"])
    focus_aspects: List[str] = field(default_factory=lambda: ["音质", "配置", "价格", "舒适", "功能", "外形"])
    aspect_keywords: Dict[str, List[str]] = field(default_factory=lambda: {"价格": ["价格", "便宜", "贵", "性价比", "溢价", "不值"], "音质": ["音质", "声音", "低音", "高音", "解析", "底噪", "电流声", "刺耳"], "功能": ["功能", "连接", "蓝牙", "操作", "通话", "延迟", "卡顿", "airplay", "usb"], "舒适": ["舒适", "佩戴", "耳罩", "压耳", "重量", "夹耳", "硌耳朵"], "外形": ["外形", "外观", "颜值", "设计", "做工", "颜色", "体积", "太大", "笨重"], "配置": ["配置", "参数", "单元", "芯片", "规格", "直推", "推不动", "搭配", "音源"], "其他": []})
    pain_point_top_k: int = 60
    pain_point_min_samples: int = 10
    cluster_count: int = 3
    include_other_in_focus_analysis: bool = False
    pain_point_trigger_terms: List[str] = field(default_factory=lambda: ["太大", "太重", "硌", "夹", "漏音", "底噪", "电流声", "噪声", "发热", "贵", "不值", "不稳", "卡顿", "延迟", "推不动", "刺耳", "佩戴不适", "驱动不足", "价格偏高", "体积过大", "识别不了", "断连", "收音闷"])
    forum_noise_terms: List[str] = field(default_factory=lambda: ["帮顶", "同问", "围观", "支持", "顶一下", "蹲", "马克", "哈哈", "呵呵", "楼主", "斑竹", "版主", "咸鱼", "二手", "路过", "香不香", "值不值"])
    pain_point_synonyms: Dict[str, str] = field(default_factory=lambda: {"硌耳朵": "佩戴不适", "夹耳": "佩戴不适", "不舒服": "佩戴不适", "戴久了疼": "佩戴不适", "底噪": "噪声", "电流声": "噪声", "杂音": "噪声", "太贵": "价格偏高", "溢价": "价格偏高", "不值": "价格偏高", "太大": "体积过大", "笨重": "体积过大", "太重": "体积过大", "推不动": "驱动不足", "直推": "驱动不足", "卡顿": "连接不稳", "延迟": "连接延迟"})
    extra_stopwords: List[str] = field(default_factory=lambda: ["的", "了", "是", "都", "就", "我", "有", "和", "这", "可以", "如果", "还是", "知道", "那么", "也"])
    scene_terms: List[str] = field(default_factory=lambda: ["耳机", "电脑", "解码", "耳放", "产品", "这个", "那个"])
    preserve_negation_words: List[str] = field(default_factory=lambda: ["不", "没", "不会", "不能", "无法"])
    preserve_degree_words: List[str] = field(default_factory=lambda: ["很", "太", "有点", "比较", "特别"])
    preserve_emotion_words: List[str] = field(default_factory=lambda: ["离谱", "无语", "失望", "闷", "糊", "刺", "差", "贵", "卡"])
    pain_aspect_terms: List[str] = field(default_factory=lambda: ["驱动", "底噪", "噪声", "麦克风", "佩戴", "音质", "解析", "价格", "连接", "电脑", "做工"])
    pain_negative_descriptors: List[str] = field(default_factory=lambda: ["不稳", "不行", "差", "闷", "糊", "刺耳", "贵", "大", "明显", "严重", "断", "卡", "掉", "延迟", "偏高", "发热", "麻烦", "识别不了", "不足", "不适", "粗糙", "漏音"])
    weak_terms: List[str] = field(default_factory=lambda: ["觉得", "一人", "人", "没有", "出来", "流行", "喜欢", "感觉", "东西", "不错", "还行", "可以", "挺好", "还好", "意思", "实在", "比较", "确实", "没兴趣", "提升很大", "差点", "可能贵"])
    topic_terms: List[str] = field(default_factory=lambda: ["耳放", "解码", "耳机", "电脑", "手机", "设备", "桌面", "便携"])
    phrase_normalization_map: Dict[str, str] = field(
        default_factory=lambda: {
            "驱动力不足": "驱动不足",
            "推不动": "驱动不足",
            "手机驱动": "驱动不足",
            "笔记本驱动": "驱动不足",
            "电脑驱动不足": "驱动不足",
            "手机驱动不足": "驱动不足",
            "笔记本驱动不足": "驱动不足",
            "连接不上": "连接问题",
            "连接延迟": "连接延迟",
            "延迟高": "连接延迟",
            "延迟明显": "连接延迟",
            "连接有延迟": "连接延迟",
            "佩戴不舒服": "佩戴不适",
            "夹头": "佩戴不适",
            "噪声": "噪声明显",
            "底噪": "噪声明显",
            "噪声大": "噪声明显",
            "底噪明显": "噪声明显",
            "价格贵": "价格偏贵",
            "有点贵": "价格偏贵",
            "价格有点高": "价格偏贵",
            "声音差": "素质差",
            "价格贵": "价格高",
            "太贵": "价格高",
            "偏贵": "价格高",
            "有延迟": "连接延迟",
            "连接慢": "连接延迟",
            "蓝牙延迟": "连接延迟",
            "有杂音": "噪声明显",
            "噪音大": "噪声明显",
            "压耳": "佩戴不适",
            "夹头": "佩戴不适",
            "推不动": "驱动不足",
            "带不动": "驱动不足",
            "驱动不了": "驱动不足",
        }
    )
    pain_aspect_mapping: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "驱动类问题": ["驱动", "推", "识别", "解码", "耳放"],
            "连接类问题": ["连接", "蓝牙", "延迟", "卡顿", "断连"],
            "音质类问题": ["噪声", "底噪", "音质", "闷", "糊", "刺耳", "素质"],
            "佩戴类问题": ["佩戴", "夹耳", "硌耳", "舒适"],
            "价格类问题": ["价格", "贵", "溢价", "性价比"],
        }
    )
    incomplete_phrase_terms: List[str] = field(default_factory=lambda: ["有点", "没有", "不会", "应该", "觉得", "试试", "出来"])
    bare_negative_terms: List[str] = field(default_factory=lambda: ["没有", "不足", "不会", "不行", "延迟", "问题"])
    positive_neutral_terms: List[str] = field(default_factory=lambda: ["没问题", "不错", "可以", "喜欢", "满意"])
    pain_point_label_whitelist: List[str] = field(
        default_factory=lambda: [
            "驱动不足",
            "价格偏高",
            "连接延迟",
            "连接不稳",
            "佩戴不适",
            "噪声明显",
            "做工较差",
            "解析较差",
            "音质较差",
            "高音刺耳",
            "低频不足",
        ]
    )


@dataclass
class ExportConfig:
    output_dir: Path = Path("system/outputs")
    artifacts_dir: Path = Path("system/outputs/artifacts")
    chart_dpi: int = 200
    wordcloud_font_path: Path | None = None


@dataclass
class SystemConfig:
    data: DataConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
