from __future__ import annotations

from pathlib import Path

import pandas as pd


class LocalDataRegion:
    """Simple read/write region for raw and processed datasets."""

    def __init__(self, root: str = "data_region"):
        self.root = Path(root)
        self.raw = self.root / "raw"
        self.processed = self.root / "processed"
        self.raw.mkdir(parents=True, exist_ok=True)
        self.processed.mkdir(parents=True, exist_ok=True)

    def write_raw(self, name: str, df: pd.DataFrame) -> Path:
        path = self.raw / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def write_processed(self, name: str, df: pd.DataFrame) -> Path:
        path = self.processed / f"{name}.csv"
        df.to_csv(path, index=False)
        return path

    def read_raw(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.raw / f"{name}.csv")

    def read_processed(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.processed / f"{name}.csv")
