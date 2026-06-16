from __future__ import annotations

import numpy as np
import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class FundamentalsProvider(BaseDataProvider):
    """Fundamental data provider with FF5-ready field coverage.

    Provides the core accounting fields needed for Fama-French
    five-factor model construction:
        book_equity  — total book equity (BE) for B/M and OP denominator
        operating_profit — revenue - COGS - SG&A - interest expense
        total_assets  — for asset growth rate (CMA investment factor)
    """

    def fetch(self, context: QueryContext) -> pd.DataFrame:
        rng = np.random.default_rng(7)
        dates = pd.date_range(context.start_date or "2024-01-31", periods=24, freq="ME")
        tickers = context.tickers or ["AAA", "BBB", "CCC", "DDD"]
        idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
        frame = idx.to_frame(index=False)

        # --- FF5 core accounting fields ---
        # Book equity (in millions, log-normally distributed around 50B)
        frame["book_equity"] = rng.lognormal(mean=10.8, sigma=0.6, size=len(frame))
        # Operating profit = Rev - COGS - SG&A - Interest
        frame["operating_profit"] = frame["book_equity"] * rng.normal(0.12, 0.05, len(frame)).clip(0.001, 0.5)
        # Total assets (typically 2-5x book equity for non-financials)
        frame["total_assets"] = frame["book_equity"] * rng.normal(3.0, 0.5, len(frame)).clip(1.2, 8.0)
        # Market cap (for size grouping — simulated; in millions)
        frame["market_cap"] = frame["book_equity"] * rng.normal(4.0, 2.0, len(frame)).clip(0.5, 25.0)

        # --- FF5 derived ratios (pre-computed for convenience) ---
        # Operating Profitability = OP / BE
        frame["op_over_be"] = frame["operating_profit"] / frame["book_equity"].clip(lower=1e-6)
        # Book-to-Market = BE / market_cap
        frame["bm_ratio"] = frame["book_equity"] / frame["market_cap"].clip(lower=1e-6)

        # --- Existing fields preserved for backward compatibility ---
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

        # Asset growth rate (for CMA): (total_assets_t - total_assets_t-12) / total_assets_t-12
        frame = frame.sort_values(["ticker", "date"])
        frame["asset_growth"] = frame.groupby("ticker")["total_assets"].pct_change(12)

        return frame
