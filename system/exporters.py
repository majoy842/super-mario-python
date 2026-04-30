from __future__ import annotations

import json
from pathlib import Path


class ResultExporter:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def to_csv(self, df, filename: str) -> Path:
        path = self.output_dir / filename
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return path

    def to_json(self, payload: dict, filename: str) -> Path:
        path = self.output_dir / filename
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
