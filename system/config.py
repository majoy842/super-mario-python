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
    stopwords_path: Path | None = None
    custom_dict_path: Path | None = None
    min_token_length: int = 1
    min_text_length: int = 4
    max_text_length: int = 512
    drop_symbol_only: bool = True
    drop_meaningless_english: bool = True
    noise_phrases: List[str] = field(default_factory=lambda: ["帮顶", "同问", "围观", "马克", "mark"])


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
            "价格": ["价格", "便宜", "贵", "性价比"],
            "音质": ["音质", "声音", "低音", "高音", "解析"],
            "功能": ["功能", "连接", "蓝牙", "操作", "通话"],
            "舒适": ["舒适", "佩戴", "耳罩", "压耳", "重量"],
            "外形": ["外形", "外观", "颜值", "设计", "做工", "颜色"],
            "配置": ["配置", "参数", "单元", "芯片", "规格"],
            "其他": [],
        }
    )
    pain_point_top_k: int = 15
    cluster_count: int = 3
    include_other_in_focus_analysis: bool = False


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
