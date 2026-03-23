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
    min_text_length: int = 4
    max_text_length: int = 512
    drop_symbol_only: bool = True
    drop_meaningless_english: bool = True
    noise_phrases: List[str] = field(
        default_factory=lambda: [
            "帮顶",
            "同问",
            "围观",
            "马克",
            "mark",
            "支持斑竹",
            "支持版主",
            "路过",
            "蹲个",
            "差评",
            "哈哈",
            "呵呵",
            "看看就很开心",
            "起码我是这样",
        ]
    )
    synonym_map: Dict[str, str] = field(
        default_factory=lambda: {
            "硌耳朵": "佩戴不适",
            "夹耳": "佩戴不适",
            "不舒服": "佩戴不适",
            "戴久了疼": "佩戴不适",
            "底噪": "噪声",
            "电流声": "噪声",
            "杂音": "噪声",
            "推不动": "驱动不足",
            "直推": "驱动不足",
            "太贵": "价格偏高",
            "溢价": "价格偏高",
            "不值": "价格偏高",
            "体积大": "体积过大",
            "太大": "体积过大",
            "笨重": "体积过大",
            "太重": "体积过大",
            "延迟": "连接延迟",
            "卡顿": "连接不稳",
        }
    )


@dataclass
class TrainingConfig:
    test_size: float = 0.2
    random_state: int = 42
    max_features: int = 5000
    batch_size: int = 16
    epochs: int = 3
    learning_rate: float = 2e-5
    max_length: int = 128


@dataclass
class AnalysisConfig:
    negative_labels: List[str] = field(default_factory=lambda: ["negative", "负面", "neg", "-1", -1])
    neutral_labels: List[str] = field(default_factory=lambda: ["neutral", "中性", "0", 0])
    positive_labels: List[str] = field(default_factory=lambda: ["positive", "正面", "1", 1])
    supported_aspects: List[str] = field(
        default_factory=lambda: ["音质", "配置", "价格", "舒适", "功能", "外形", "其他"]
    )
    focus_aspects: List[str] = field(default_factory=lambda: ["音质", "配置", "价格", "舒适", "功能", "外形"])
    aspect_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "价格": ["价格", "便宜", "贵", "性价比", "溢价", "不值"],
            "音质": ["音质", "声音", "低音", "高音", "解析", "底噪", "电流声", "刺耳"],
            "功能": ["功能", "连接", "蓝牙", "操作", "通话", "延迟", "卡顿", "airplay", "usb"],
            "舒适": ["舒适", "佩戴", "耳罩", "压耳", "重量", "夹耳", "硌耳朵"],
            "外形": ["外形", "外观", "颜值", "设计", "做工", "颜色", "体积", "太大", "笨重"],
            "配置": ["配置", "参数", "单元", "芯片", "规格", "直推", "推不动", "搭配", "音源"],
            "其他": [],
        }
    )
    pain_point_top_k: int = 20
    cluster_count: int = 3
    include_other_in_focus_analysis: bool = False
    pain_point_trigger_terms: List[str] = field(
        default_factory=lambda: [
            "太大",
            "太重",
            "硌",
            "夹",
            "漏音",
            "底噪",
            "电流声",
            "噪声",
            "发热",
            "贵",
            "不值",
            "不稳",
            "卡顿",
            "延迟",
            "推不动",
            "刺耳",
            "佩戴不适",
            "驱动不足",
            "价格偏高",
            "体积过大",
        ]
    )
    forum_noise_terms: List[str] = field(
        default_factory=lambda: [
            "帮顶",
            "同问",
            "围观",
            "支持",
            "顶一下",
            "蹲",
            "马克",
            "哈哈",
            "呵呵",
            "楼主",
            "斑竹",
            "版主",
            "咸鱼",
            "二手",
            "路过",
            "香不香",
            "值不值",
        ]
    )
    pain_point_synonyms: Dict[str, str] = field(
        default_factory=lambda: {
            "硌耳朵": "佩戴不适",
            "夹耳": "佩戴不适",
            "不舒服": "佩戴不适",
            "戴久了疼": "佩戴不适",
            "底噪": "噪声",
            "电流声": "噪声",
            "杂音": "噪声",
            "太贵": "价格偏高",
            "溢价": "价格偏高",
            "不值": "价格偏高",
            "太大": "体积过大",
            "笨重": "体积过大",
            "太重": "体积过大",
            "推不动": "驱动不足",
            "直推": "驱动不足",
            "卡顿": "连接不稳",
            "延迟": "连接延迟",
        }
    )


@dataclass
class ExportConfig:
    output_dir: Path = Path("system/outputs")
    chart_dpi: int = 200


@dataclass
class SystemConfig:
    data: DataConfig
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
