"""Tests for data preprocessing: winsorize, neutralize, zscore, and dataset building."""
import numpy as np
import pandas as pd
import pytest

from src.data.base_provider import QueryContext
from src.data.fundamentals_provider import FundamentalsProvider
from src.data.ohlcv_provider import OHLCVProvider
from src.data.universe_provider import UniverseProvider
from src.data.macro_provider import MacroProvider
from src.features.price_features import make_price_features
from src.features.fundamental_features import make_fundamental_features
from src.features.merge import merge_feature_panels
from src.models.industry_low_frequency_models import (
    LowFrequencyModelConfig,
    FactorEngineeringMixin,
    BaseLowFrequencyTradeModel,
)

# Instantiate a mixin for standalone testing of preprocessing methods
_fem = FactorEngineeringMixin()
winsorize = _fem.winsorize_cross_section
neutralize = _fem.simple_neutralize
zscore = _fem.zscore_cross_section


class TestWinsorize:
    """Cross-sectional winsorization tests."""

    def test_clips_extreme_values(self):
        df = pd.DataFrame({
            "date": ["2024-01-31"] * 100,
            "ticker": [f"S{i}" for i in range(100)],
            "factor": list(range(100)),  # 0..99
        })
        result = winsorize(df.copy(), ["factor"], "date", 0.05, 0.95)
        # Should clip: 0..4 -> 5, 95..99 -> 94
        assert result["factor"].min() >= 4
        assert result["factor"].max() <= 95

    def test_all_same_no_crash(self):
        """All values identical should not crash winsorize."""
        df = pd.DataFrame({
            "date": ["2024-01-31"] * 10,
            "ticker": [f"S{i}" for i in range(10)],
            "factor": [5.0] * 10,
        })
        result = winsorize(df.copy(), ["factor"], "date", 0.01, 0.99)
        assert (result["factor"] == 5.0).all()


class TestNeutralize:
    """Industry neutralization tests."""

    def test_removes_industry_mean(self):
        df = pd.DataFrame({
            "date": ["2024-01-31"] * 6,
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "industry": ["tech", "tech", "tech", "oil", "oil", "oil"],
            "factor": [10.0, 12.0, 14.0, 100.0, 104.0, 96.0],
        })
        result = neutralize(df, ["factor"], ["industry"], "date")
        # tech mean=12, oil mean=100
        tech = result[result["industry"] == "tech"]["factor"]
        oil = result[result["industry"] == "oil"]["factor"]
        np.testing.assert_almost_equal(tech.mean(), 0.0, decimal=6)
        np.testing.assert_almost_equal(oil.mean(), 0.0, decimal=6)

    def test_single_industry_no_crash(self):
        """Single industry in cross-section should not crash."""
        df = pd.DataFrame({
            "date": ["2024-01-31"] * 3,
            "ticker": ["A", "B", "C"],
            "industry": ["tech"] * 3,
            "factor": [10.0, 20.0, 30.0],
        })
        result = neutralize(df, ["factor"], ["industry"], "date")
        # Mean should be 0
        np.testing.assert_almost_equal(result["factor"].mean(), 0.0, decimal=6)


class TestZScore:
    """Z-score normalization tests."""

    def test_zero_mean_unit_variance(self):
        df = pd.DataFrame({
            "date": ["2024-01-31"] * 100,
            "ticker": [f"S{i}" for i in range(100)],
            "factor": np.random.randn(100) * 10 + 50,
        })
        result = zscore(df.copy(), ["factor"], "date")
        np.testing.assert_almost_equal(result["factor"].mean(), 0.0, decimal=1)
        np.testing.assert_almost_equal(result["factor"].std(), 1.0, decimal=1)


class TestBuildDataset:
    """Integration: build_dataset with synthetic data."""

    def test_synthetic_output(self):
        from src.pipelines.build_dataset import build_dataset
        panel = build_dataset(
            start_date="2024-01-31",
            end_date="2025-06-30",
            include_ff5_factors=True,
            dataset_source="synthetic",
        )
        assert len(panel) > 50
        assert "future_return" in panel.columns
        # With >=8 tickers, FF5 should be present
        if len(panel["ticker"].unique()) >= 6:
            assert "mkt_exposure" in panel.columns
            assert "smb_exposure" in panel.columns

    def test_synthetic_no_nan_target(self):
        from src.pipelines.build_dataset import build_dataset
        panel = build_dataset(
            start_date="2024-01-31",
            end_date="2025-06-30",
            include_ff5_factors=True,
            dataset_source="synthetic",
        )
        # future_return should not be NaN for trainable rows
        trainable = panel.dropna(subset=["future_return"])
        assert len(trainable) > 0


class TestTrainPipeline:
    """End-to-end: train_industry_model with different model types."""

    def test_ridge_trains(self):
        from src.pipelines.train_industry_model import train_industry_model
        result = train_industry_model(
            industry="ai_hardware",
            model_type="ridge",
            dataset_source="synthetic",
        )
        assert "metrics" in result
        assert "last_cum_return" in result["metrics"]
        assert result["metrics"]["last_cum_return"] is not None

    def test_ensemble_trains(self):
        from src.pipelines.train_industry_model import train_industry_model
        result = train_industry_model(
            industry="ai_hardware",
            model_type="ensemble",
            dataset_source="synthetic",
        )
        assert "metrics" in result
        assert result["metrics"]["last_cum_return"] is not None

    def test_different_models_different_results(self):
        """Ridge and ensemble should produce different cumulative returns."""
        from src.pipelines.train_industry_model import train_industry_model
        r1 = train_industry_model(industry="ai_hardware", model_type="ridge", dataset_source="synthetic")
        r2 = train_industry_model(industry="ai_hardware", model_type="ensemble", dataset_source="synthetic")
        assert r1["metrics"]["last_cum_return"] != r2["metrics"]["last_cum_return"]

    def test_insufficient_data_error(self):
        """Industry with no matching tickers should raise clear error."""
        from src.pipelines.train_industry_model import train_industry_model
        with pytest.raises((ValueError, RuntimeError)):
            train_industry_model(
                industry="ai_hardware",
                model_type="ridge",
                dataset_source="synthetic",
                tickers=["IONQ", "RGTI"],  # only quantum tickers, no ai_hardware
            )
