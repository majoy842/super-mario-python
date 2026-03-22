from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class DatasetSummary:
    rows: int
    columns: list[str]
    label_distribution: dict


class DatasetLoader:
    """Load review data from common tabular formats."""

    SUPPORTED_SUFFIXES = {".csv", ".xlsx", ".xls", ".json"}

    def __init__(self, data_path: Path, encoding: str = "utf-8-sig") -> None:
        self.data_path = Path(data_path)
        self.encoding = encoding

    def load(self):
        suffix = self.data_path.suffix.lower()
        if suffix not in self.SUPPORTED_SUFFIXES:
            raise ValueError(f"Unsupported dataset format: {suffix}")

        import pandas as pd

        if suffix == ".csv":
            return pd.read_csv(self.data_path, encoding=self.encoding)
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(self.data_path)
        return pd.read_json(self.data_path)

    @staticmethod
    def summarize(df, label_column: str) -> DatasetSummary:
        distribution = {}
        if label_column in df.columns:
            distribution = df[label_column].value_counts(dropna=False).to_dict()
        return DatasetSummary(rows=len(df), columns=df.columns.tolist(), label_distribution=distribution)
