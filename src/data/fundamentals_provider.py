from __future__ import annotations

import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class FundamentalsProvider(BaseDataProvider):
    """Placeholder fundamental provider with realistic column names."""

    def fetch(self, context: QueryContext) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        dates = pd.date_range(context.start_date or "2024-01-31", periods=24, freq="ME")
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        frame = idx.to_frame(index=False)
        frame["rev_growth_yoy"] = rng.normal(0.1, 0.2, len(frame))
        frame["gross_margin_ttm"] = rng.normal(0.45, 0.08, len(frame)).clip(0.01, 0.9)
        frame["capex_ratio"] = rng.normal(0.06, 0.02, len(frame)).clip(0.0, 0.3)
        frame["inventory_days"] = rng.normal(80, 15, len(frame)).clip(10, 240)
        frame["pe_fwd"] = rng.normal(20, 8, len(frame)).clip(1, 100)
        frame["ps_fwd"] = rng.normal(5, 2, len(frame)).clip(0.2, 20)
        frame["fcf_yield"] = rng.normal(0.04, 0.03, len(frame))
        frame["debt_to_equity"] = rng.normal(0.7, 0.4, len(frame)).clip(0, 5)
        frame["ev_ebitda"] = rng.normal(12, 4, len(frame)).clip(1, 40)
        frame["production_growth"] = rng.normal(0.03, 0.08, len(frame))
        frame["order_growth"] = rng.normal(0.05, 0.1, len(frame))
        frame["inventory_growth"] = rng.normal(0.03, 0.09, len(frame))
        frame["pe_ttm"] = rng.normal(18, 7, len(frame)).clip(1, 100)
        frame["ps_ttm"] = rng.normal(3, 1.4, len(frame)).clip(0.2, 20)
        return frame
