from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DataConfig:
    data_path: Path
    text_column: str = "review_text"
    label_column: str = "sentiment_label"
    aspect_column: str = "aspect"
    sentiment_word_column: str = "sentiment_words"
    id_column: str = "review_id"
    encoding: str = "utf-8-sig"


@dataclass
class PreprocessConfig:
    remove_duplicates: bool = True
    lowercase: bool = True
    remove_digits: bool = False
    stopwords_path: Path | None = None
    custom_dict_path: Path | None = None
    min_token_length: int = 1


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
    negative_labels: List[str] = field(default_factory=lambda: ["negative", "负面", "neg", "-1"])
    supported_aspects: List[str] = field(
        default_factory=lambda: ["价格", "音质", "功能", "舒适度", "外观", "续航", "降噪"]
    )
    aspect_keywords: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "价格": ["价格", "便宜", "贵", "性价比"],
            "音质": ["音质", "声音", "低音", "高音", "解析"],
            "功能": ["功能", "连接", "蓝牙", "操作", "降噪"],
            "舒适度": ["舒适", "佩戴", "耳罩", "压耳", "重量"],
            "外观": ["外观", "颜值", "设计", "做工", "颜色"],
            "续航": ["续航", "电量", "充电", "待机"],
            "降噪": ["降噪", "噪音", "环境音", "隔音"],
        }
    )
    pain_point_top_k: int = 15
    cluster_count: int = 3


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
