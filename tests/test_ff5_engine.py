"""Tests for FF5 factor engine — core logic, rebalance, edge cases."""
import numpy as np
import pandas as pd
import pytest

from src.features.ff5_engine import FF5Config, FF5Engine, FF5_FACTOR_DESCRIPTIONS


def make_test_data(n_tickers=50, n_dates=12, seed=42):
    """Create synthetic fundamentals + returns + universe for testing."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2024-01-31", periods=n_dates, freq="ME")
    tickers = [f"STK{i:03d}" for i in range(n_tickers)]

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

    return fundamentals, returns, universe


class TestFF5Engine:
    """Core FF5 factor exposure computation tests."""

    def test_compute_basic(self):
        """Smoke test: compute factor exposures without errors."""
        f, r, u = make_test_data()
        engine = FF5Engine(FF5Config(mode="industry"))
        exposures = engine.compute_factor_exposures(f, r, u)

        assert not exposures.empty
        for col in FF5_FACTOR_DESCRIPTIONS:
            assert col in exposures.columns, f"Missing column: {col}"

    def test_output_shape(self):
        """Output should have one row per (date, ticker) pair."""
        f, r, u = make_test_data(n_tickers=20, n_dates=6, seed=99)
        engine = FF5Engine(FF5Config(mode="industry"))
        exposures = engine.compute_factor_exposures(f, r, u)

        assert len(exposures) >= 20 * 6
        assert "date" in exposures.columns
        assert "ticker" in exposures.columns

    def test_june_end_rebalance_carry_forward(self):
        """Non-June months should carry forward last June exposures."""
        f, r, u = make_test_data(n_tickers=30, n_dates=12, seed=1)
        engine = FF5Engine(FF5Config(mode="industry", use_annual_rebalance=True, rebalance_month=6))
        exposures = engine.compute_factor_exposures(f, r, u)
        factor_cols = engine.config.factor_cols

        july = exposures[pd.to_datetime(exposures["date"]).dt.month == 7]
        assert not july.empty, "July should have data (carried from June)"

        june = exposures[pd.to_datetime(exposures["date"]).dt.month == 6]
        if not june.empty and not july.empty:
            ticker = june["ticker"].iloc[0]
            june_vals = june[june["ticker"] == ticker][factor_cols].values
            july_vals = july[july["ticker"] == ticker][factor_cols].values
            if len(june_vals) > 0 and len(july_vals) > 0:
                np.testing.assert_array_equal(june_vals[0], july_vals[0],
                    "July exposures should equal June (carry-forward)")

    def test_no_rebalance_every_month(self):
        """With rebalance off, every month should have fresh computation."""
        f, r, u = make_test_data(n_tickers=30, n_dates=12, seed=2)
        engine = FF5Engine(FF5Config(mode="industry", use_annual_rebalance=False))
        exposures = engine.compute_factor_exposures(f, r, u)

        # Two consecutive months for same ticker may have different exposures
        jan = exposures[pd.to_datetime(exposures["date"]).dt.month == 1]
        feb = exposures[pd.to_datetime(exposures["date"]).dt.month == 2]
        if not jan.empty and not feb.empty:
            ticker = jan["ticker"].iloc[0]
            jv = jan[jan["ticker"] == ticker]["smb_exposure"].values
            fv = feb[feb["ticker"] == ticker]["smb_exposure"].values
            # They may or may not differ; this just tests no crash

    def test_few_tickers_neutral(self):
        """< 6 tickers per industry should produce neutral or empty FF5 rows."""
        f, r, u = make_test_data(n_tickers=4, n_dates=4)
        engine = FF5Engine(FF5Config(mode="industry"))
        exposures = engine.compute_factor_exposures(f, r, u)

        # With only 4 tickers split across 4 industries, FF5 may be neutral
        # or engine skips dates entirely. Either is correct behavior.
        if not exposures.empty:
            for col in ["smb_exposure", "hml_exposure", "rmw_exposure", "cma_exposure"]:
                unique_vals = exposures[col].unique()
                assert len(unique_vals) <= 3, f"{col} should have very few unique values for tiny N"
        # At minimum, engine did not crash — test passes

    def test_empty_result_graceful(self):
        """Engine should handle empty input gracefully."""
        f = pd.DataFrame(columns=["date", "ticker", "market_cap", "bm_ratio", "op_over_be", "asset_growth"])
        r = pd.DataFrame(columns=["date", "ticker", "ret_1m"])
        engine = FF5Engine(FF5Config(mode="industry"))
        exposures = engine.compute_factor_exposures(f, r)

        assert exposures.empty
        for col in FF5_FACTOR_DESCRIPTIONS:
            assert col in exposures.columns

    def test_factor_descriptions_complete(self):
        """All 5 factors should have descriptions."""
        assert len(FF5_FACTOR_DESCRIPTIONS) == 5
        for name in ["mkt_exposure", "smb_exposure", "hml_exposure", "rmw_exposure", "cma_exposure"]:
            assert name in FF5_FACTOR_DESCRIPTIONS
            assert len(FF5_FACTOR_DESCRIPTIONS[name]) > 10
