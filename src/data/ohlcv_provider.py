from __future__ import annotations

import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class OHLCVProvider(BaseDataProvider):
    """Placeholder provider. Replace with SQL query when DB is ready."""

    def fetch(self, context: QueryContext) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        dates = pd.date_range(context.start_date or "2024-01-31", periods=24, freq="ME")
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        frame = idx.to_frame(index=False)
        frame["close"] = rng.normal(100, 10, len(frame)).clip(1)
        frame["volume"] = rng.integers(100_000, 1_000_000, len(frame))
        return frame
