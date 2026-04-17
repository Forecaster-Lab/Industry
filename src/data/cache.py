from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import pandas as pd


@dataclass
class InMemoryFrameCache:
    frames: Dict[str, pd.DataFrame] = field(default_factory=dict)

    def get(self, key: str) -> pd.DataFrame | None:
        return self.frames.get(key)

    def set(self, key: str, df: pd.DataFrame) -> None:
        self.frames[key] = df.copy()
