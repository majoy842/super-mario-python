from __future__ import annotations

from pathlib import Path


def existing_export_path(output_dir: str | Path, filename: str) -> Path | None:
    path = Path(output_dir) / filename
    return path if path.exists() else None

