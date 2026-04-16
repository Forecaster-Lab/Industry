from __future__ import annotations

import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class MacroProvider(BaseDataProvider):
    def fetch(self, context: QueryContext) -> pd.DataFrame:
        rng = np.random.default_rng(11)
        dates = pd.date_range(context.start_date or "2024-01-31", periods=24, freq="ME")
        frame = pd.DataFrame({
            "date": dates,
            "benchmark_return": rng.normal(0.008, 0.03, len(dates)),
            "oil_change_1m": rng.normal(0.0, 0.06, len(dates)),
            "policy_rate_change_3m": rng.normal(0.0, 0.01, len(dates)),
        })
        return frame
