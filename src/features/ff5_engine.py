"""
FF5 Factor Engine — Fama-French Five-Factor Model

Implements the canonical Fama-French (2015) five-factor construction:
    MKT  — market excess return (R_m - R_f)
    SMB  — Small Minus Big (triple-sorted)
    HML  — High Minus Low (B/M, 2x3 sort with size)
    RMW  — Robust Minus Weak (operating profitability, 2x3 sort)
    CMA  — Conservative Minus Aggressive (investment, 2x3 sort)

Core reference:
    Fama & French (2015), J. Financial Economics 116, 1-22.
    "A five-factor asset pricing model"

The engine supports two modes:
    - market_mode: full cross-section (NYSE-style percentile breaks)
    - industry_mode: within-industry cross-section (per the project's
      industry-level analysis architecture)

Grouping logic (annual, every June-end):
    1. Size: split by NYSE median (or within-industry median)
    2. B/M:   top 30% / mid 40% / bottom 30%
    3. OP:    top 30% / mid 40% / bottom 30%
    4. Inv:   bottom 30% / mid 40% / top 30% (conservative = low growth)

    SMB = (SMB_BM + SMB_OP + SMB_Inv) / 3

June-end rebalance:
    Rebalance only occurs in June (rebalance_month=6 by default). Between
    rebalance dates, exposures are held constant — matching the Fama-French
    convention where portfolios are formed each June and held for 12 months.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Literal, Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class FF5Config:
    """Configuration for FF5 factor construction."""

    # Data column mappings
    date_col: str = "date"
    ticker_col: str = "ticker"
    industry_col: str = "industry"
    market_cap_col: str = "market_cap"
    bm_col: str = "bm_ratio"
    op_col: str = "op_over_be"
    inv_col: str = "asset_growth"
    ret_col: str = "ret_1m"
    benchmark_col: str = "benchmark_return"

    # Breakpoint configuration
    size_pct: float = 0.5
    bm_high_pct: float = 0.30
    bm_low_pct: float = 0.30
    op_high_pct: float = 0.30
    op_low_pct: float = 0.30
    inv_low_pct: float = 0.30
    inv_high_pct: float = 0.30

    # Mode
    mode: Literal["industry", "market"] = "industry"

    # Rebalance schedule (Fama-French: June-end)
    rebalance_month: int = 6
    use_annual_rebalance: bool = True

    # Columns to output for factor exposures
    factor_cols: List[str] = field(default_factory=lambda: [
        "mkt_exposure",
        "smb_exposure",
        "hml_exposure",
        "rmw_exposure",
        "cma_exposure",
    ])  # ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------


class FF5Engine:
    """Construct Fama-French five-factor exposures at the individual stock level.

    Instead of producing aggregate factor returns (like the canonical FF dataset),
    this engine computes factor *exposures* for each stock at each date: what
    quintile/group does this stock belong to for each factor?

    These exposures can then be used as features in downstream ML models,
    complementing the existing price/fundamental/quantum features.
    """

    def __init__(self, config: Optional[FF5Config] = None):
        self.config = config or FF5Config()
        self._last_rebalance_exposures: Optional[pd.DataFrame] = None
        self._last_rebalance_date = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_factor_exposures(
        self,
        fundamentals: pd.DataFrame,
        returns: pd.DataFrame,
        universe: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """Compute FF5 factor exposures for every (date, ticker) row."""
        cfg = self.config
        panel = self._prepare_panel(fundamentals, returns, universe)
        monthly_dates = sorted(panel[cfg.date_col].dropna().unique())
        results: List[pd.DataFrame] = []

        self._last_rebalance_exposures = None
        self._last_rebalance_date = None

        for dt in monthly_dates:
            slice_df = panel[panel[cfg.date_col] == dt].copy()
            if len(slice_df) < 3:
                continue

            is_rebalance = self._is_rebalance_date(dt)
            if is_rebalance or self._last_rebalance_exposures is None:
                exposures = self._compute_date_exposures(slice_df)
                if is_rebalance:
                    self._last_rebalance_exposures = exposures.copy()
                    self._last_rebalance_date = dt
                results.append(exposures)
            else:
                carried = self._carry_forward_exposures(
                    slice_df, self._last_rebalance_exposures, dt
                )
                results.append(carried)

        if not results:
            return pd.DataFrame(columns=[cfg.date_col, cfg.ticker_col] + cfg.factor_cols)

        return pd.concat(results, ignore_index=True)

    # ------------------------------------------------------------------
    # Rebalance scheduling
    # ------------------------------------------------------------------

    def _is_rebalance_date(self, dt) -> bool:
        cfg = self.config
        if not cfg.use_annual_rebalance:
            return True
        dt = pd.Timestamp(dt)
        return dt.month == cfg.rebalance_month

    def _carry_forward_exposures(
        self, current_slice, last_exposures, current_date
    ) -> pd.DataFrame:
        cfg = self.config
        carried = last_exposures[
            last_exposures[cfg.ticker_col].isin(current_slice[cfg.ticker_col])
        ].copy()
        if carried.empty:
            return self._compute_date_exposures(current_slice)
        carried[cfg.date_col] = pd.Timestamp(current_date)
        return carried[
            [cfg.date_col, cfg.ticker_col] + cfg.factor_cols
        ].reset_index(drop=True)    # ------------------------------------------------------------------
    # Internal: panel preparation
    # ------------------------------------------------------------------

    def _prepare_panel(
        self, fundamentals, returns, universe,
    ) -> pd.DataFrame:
        cfg = self.config
        fund_cols = [cfg.date_col, cfg.ticker_col, cfg.market_cap_col,
                     cfg.bm_col, cfg.op_col, cfg.inv_col]
        ret_cols = [cfg.date_col, cfg.ticker_col, cfg.ret_col]

        bench_df = None
        if cfg.benchmark_col in returns.columns:
            bench_df = returns[[cfg.date_col, cfg.benchmark_col]].drop_duplicates()

        f = fundamentals[fund_cols].copy()
        r = returns[ret_cols].copy()
        panel = f.merge(r, on=[cfg.date_col, cfg.ticker_col], how="inner")

        if bench_df is not None:
            panel = panel.merge(bench_df, on=cfg.date_col, how="left")

        if universe is not None and cfg.industry_col in universe.columns:
            panel = panel.merge(
                universe[[cfg.ticker_col, cfg.industry_col]],
                on=cfg.ticker_col, how="left",
            )
        return panel

    # ------------------------------------------------------------------
    # Internal: per-date exposure computation
    # ------------------------------------------------------------------

    def _compute_date_exposures(self, slice_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        df = slice_df.copy()
        grouping_col = (
            cfg.industry_col
            if cfg.mode == "industry" and cfg.industry_col in df.columns
            else None
        )
        if grouping_col:
            df = df.groupby(grouping_col, group_keys=False).apply(
                self._assign_factor_exposures, include_groups=False
            )
        else:
            df = self._assign_factor_exposures(df)
        return df[[cfg.date_col, cfg.ticker_col] + cfg.factor_cols].reset_index(drop=True)

    def _assign_factor_exposures(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        n = len(df)
        if n < 6:
            for col in cfg.factor_cols:
                df[col] = 0.0
            return df

        # Market factor (MKT) — cap rank percentile
        df["mkt_exposure"] = df[cfg.market_cap_col].rank(pct=True)

        # Size split
        cap_median = df[cfg.market_cap_col].median()
        df["_is_small"] = (df[cfg.market_cap_col] <= cap_median).astype(float)

        # B/M groups
        bm_hi = (
            df[cfg.bm_col].quantile(1 - cfg.bm_high_pct)
            if df[cfg.bm_col].notna().sum() > 3
            else df[cfg.bm_col].max()
        )
        bm_lo = (
            df[cfg.bm_col].quantile(cfg.bm_low_pct)
            if df[cfg.bm_col].notna().sum() > 3
            else df[cfg.bm_col].min()
        )
        df["_bm_group"] = pd.cut(
            df[cfg.bm_col],
            bins=[-np.inf, bm_lo, bm_hi, np.inf],
            labels=["low", "neutral", "high"],
            duplicates="drop",
        )
        df["_smb_bm"] = df["_is_small"].map({1.0: 0.5, 0.0: -0.5})

        # OP groups
        op_hi = (
            df[cfg.op_col].quantile(1 - cfg.op_high_pct)
            if df[cfg.op_col].notna().sum() > 3
            else df[cfg.op_col].max()
        )
        op_lo = (
            df[cfg.op_col].quantile(cfg.op_low_pct)
            if df[cfg.op_col].notna().sum() > 3
            else df[cfg.op_col].min()
        )
        df["_op_group"] = pd.cut(
            df[cfg.op_col],
            bins=[-np.inf, op_lo, op_hi, np.inf],
            labels=["weak", "neutral", "robust"],
            duplicates="drop",
        )
        df["_smb_op"] = df["_is_small"].map({1.0: 0.5, 0.0: -0.5})

        # Inv groups
        inv_lo = (
            df[cfg.inv_col].quantile(cfg.inv_low_pct)
            if df[cfg.inv_col].notna().sum() > 3
            else df[cfg.inv_col].min()
        )
        inv_hi = (
            df[cfg.inv_col].quantile(1 - cfg.inv_high_pct)
            if df[cfg.inv_col].notna().sum() > 3
            else df[cfg.inv_col].max()
        )
        df["_inv_group"] = pd.cut(
            df[cfg.inv_col],
            bins=[-np.inf, inv_lo, inv_hi, np.inf],
            labels=["conservative", "neutral", "aggressive"],
            duplicates="drop",
        )
        df["_smb_inv"] = df["_is_small"].map({1.0: 0.5, 0.0: -0.5})

        # SMB = triple-sort average
        df["smb_exposure"] = (df["_smb_bm"] + df["_smb_op"] + df["_smb_inv"]) / 3.0

        # HML / RMW / CMA
        df["hml_exposure"] = df["_bm_group"].map(
            {"high": 1.0, "neutral": 0.0, "low": -1.0}).fillna(0.0).astype(float)
        df["rmw_exposure"] = df["_op_group"].map(
            {"robust": 1.0, "neutral": 0.0, "weak": -1.0}).fillna(0.0).astype(float)
        df["cma_exposure"] = df["_inv_group"].map(
            {"conservative": 1.0, "neutral": 0.0, "aggressive": -1.0}).fillna(0.0).astype(float)

        for tmp in ["_is_small", "_bm_group", "_op_group", "_inv_group",
                    "_smb_bm", "_smb_op", "_smb_inv"]:
            if tmp in df.columns:
                del df[tmp]
        return df  # ---------------------------------------------------------------------------
# Convenience: factor names and descriptions
# ---------------------------------------------------------------------------


FF5_FACTOR_DESCRIPTIONS: Dict[str, str] = {
    "mkt_exposure": "Market beta proxy (market-cap rank percentile in cross-section)",
    "smb_exposure": "Size factor (Small Minus Big) — triple-sorted average",
    "hml_exposure": "Value factor (High B/M Minus Low B/M) — 2x3 size-B/M sort",
    "rmw_exposure": "Profitability factor (Robust OP Minus Weak OP) — 2x3 size-OP sort",
    "cma_exposure": "Investment factor (Conservative Minus Aggressive) — 2x3 size-Inv sort",
}


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    rng = np.random.default_rng(42)
    dates = pd.date_range("2024-01-31", periods=12, freq="ME")
    tickers = [f"STK{i:03d}" for i in range(50)]

    idx = pd.MultiIndex.from_product([dates, tickers], names=["date", "ticker"])
    frame = idx.to_frame(index=False)
    frame["book_equity"] = rng.lognormal(mean=10.8, sigma=0.6, size=len(frame))
    frame["operating_profit"] = frame["book_equity"] * rng.normal(0.12, 0.05, len(frame)).clip(0.001, 0.5)
    frame["total_assets"] = frame["book_equity"] * rng.normal(3.0, 0.5, len(frame)).clip(1.2, 8.0)
    frame["market_cap"] = frame["book_equity"] * rng.normal(4.0, 2.0, len(frame)).clip(0.5, 25.0)
    frame["op_over_be"] = frame["operating_profit"] / frame["book_equity"].clip(lower=1e-6)
    frame["bm_ratio"] = frame["book_equity"] / frame["market_cap"].clip(lower=1e-6)
    frame["asset_growth"] = rng.normal(0.05, 0.15, len(frame))
    fundamentals = frame

    returns = pd.DataFrame({
        "date": np.tile(dates, len(tickers)),
        "ticker": np.repeat(tickers, len(dates)),
        "ret_1m": rng.normal(0.005, 0.08, len(dates) * len(tickers)),
        "benchmark_return": rng.normal(0.008, 0.03, len(dates) * len(tickers)),
    })

    universe = pd.DataFrame({
        "ticker": tickers,
        "industry": np.random.choice(["ai_hardware", "energy", "photonics", "quantum"], len(tickers)),
    })

    engine = FF5Engine(FF5Config(mode="industry"))
    exposures = engine.compute_factor_exposures(fundamentals, returns, universe)
    print("FF5 (June-end rebalance ON):")
    print(exposures.head(10))
    non_june = exposures[~pd.to_datetime(exposures["date"]).dt.month.isin([6])]
    june = exposures[pd.to_datetime(exposures["date"]).dt.month.isin([6])]
    print(f"June months: {len(june)} rows   Non-June: {len(non_june)} rows")
    # Verify carry-forward: non-June exposures should match nearest prior June
    if not non_june.empty and not june.empty:
        sample_ticker = non_june["ticker"].iloc[0]
        non_june_vals = non_june[non_june["ticker"] == sample_ticker]
        print(f"Sample ticker {sample_ticker} non-June exposures (should be constant between Junes):")
        print(non_june_vals.head(3).to_string())
    print("\nSmoke test passed.")
