from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression, Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except ImportError:  # pragma: no cover
    RandomForestRegressor = None
    SimpleImputer = None
    Ridge = None
    LogisticRegression = None
    Pipeline = None
    StandardScaler = None


@dataclass
class LowFrequencyModelConfig:
    date_col: str = "date"
    asset_col: str = "ticker"
    industry_col: str = "industry"
    target_col: str = "future_return"
    benchmark_col: str = "benchmark_return"
    prediction_col: str = "prediction"
    score_col: str = "score"
    min_train_rows: int = 50
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    winsorize_limits: Tuple[float, float] = (0.01, 0.99)
    clip_target: Optional[Tuple[float, float]] = (-0.6, 0.6)
    neutralize_by: List[str] = field(default_factory=list)
    long_only: bool = False
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0
    model_type: str = "ridge"


@dataclass
class EpochAIConfig:
    target_col: str = "label"
    probability_col: str = "probability"
    prediction_col: str = "prediction"
    min_train_rows: int = 80
    use_classifier: bool = True


class DataValidator:
    @staticmethod
    def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")


class FactorEngineeringMixin:
    def winsorize_cross_section(self, df: pd.DataFrame, feature_cols: List[str], date_col: str, lower: float, upper: float) -> pd.DataFrame:
        out = df.copy()
        for col in feature_cols:
            out[col] = out.groupby(date_col)[col].transform(lambda s: s.clip(s.quantile(lower), s.quantile(upper)))
        return out

    def zscore_cross_section(self, df: pd.DataFrame, feature_cols: List[str], date_col: str) -> pd.DataFrame:
        out = df.copy()
        for col in feature_cols:
            out[col] = out.groupby(date_col)[col].transform(
                lambda s: (s - s.mean()) / (s.std() if s.std() and not pd.isna(s.std()) else 1.0)
            )
        return out

    def simple_neutralize(self, df: pd.DataFrame, feature_cols: List[str], neutralize_by: List[str], date_col: str) -> pd.DataFrame:
        if not neutralize_by:
            return df
        out = df.copy()
        grouping = [date_col] + neutralize_by
        for col in feature_cols:
            out[col] = out[col] - out.groupby(grouping)[col].transform("mean")
        return out


class BaseLowFrequencyTradeModel(FactorEngineeringMixin):
    def __init__(self, config: Optional[LowFrequencyModelConfig] = None):
        self.config = config or LowFrequencyModelConfig()
        self.feature_cols: List[str] = []
        self.model: Any = None
        self.is_fitted = False

    def default_feature_columns(self) -> List[str]:
        return []

    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.copy()

    def build_model(self):
        if Pipeline is None:
            raise ImportError("Install scikit-learn to train models.")

        model_type = self.config.model_type.lower()
        estimator = Ridge(alpha=1.0)
        if model_type == "random_forest":
            estimator = RandomForestRegressor(n_estimators=300, max_depth=6, min_samples_leaf=8, random_state=42, n_jobs=-1)
        elif model_type == "lightgbm":
            try:
                from lightgbm import LGBMRegressor

                estimator = LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=31, random_state=42)
            except Exception:
                estimator = Ridge(alpha=1.0)
        elif model_type == "xgboost":
            try:
                from xgboost import XGBRegressor

                estimator = XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                )
            except Exception:
                estimator = Ridge(alpha=1.0)
        elif model_type == "ranker":
            estimator = Ridge(alpha=0.6)
        elif model_type == "ensemble":
            estimator = RandomForestRegressor(n_estimators=150, max_depth=5, random_state=42, n_jobs=-1)

        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        DataValidator.require_columns(df, [cfg.date_col, cfg.asset_col])
        out = df.copy()
        out[cfg.date_col] = pd.to_datetime(out[cfg.date_col])
        out = self.custom_feature_engineering(out)
        if not self.feature_cols:
            self.feature_cols = self.default_feature_columns()
        DataValidator.require_columns(out, self.feature_cols)
        out = self.winsorize_cross_section(out, self.feature_cols, cfg.date_col, *cfg.winsorize_limits)
        out = self.simple_neutralize(out, self.feature_cols, cfg.neutralize_by, cfg.date_col)
        out = self.zscore_cross_section(out, self.feature_cols, cfg.date_col)
        if cfg.target_col in out.columns and cfg.clip_target:
            out[cfg.target_col] = out[cfg.target_col].clip(*cfg.clip_target)
        return out

    def fit(self, train_df: pd.DataFrame):
        cfg = self.config
        DataValidator.require_columns(train_df, [cfg.target_col])
        prepared = self.prepare_features(train_df).dropna(subset=[cfg.target_col])
        if len(prepared) < cfg.min_train_rows:
            raise ValueError(f"Not enough training rows: {len(prepared)} < {cfg.min_train_rows}")
        self.model = self.build_model()
        self.model.fit(prepared[self.feature_cols], prepared[cfg.target_col])
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        cfg = self.config
        out = self.prepare_features(df)
        out[cfg.prediction_col] = self.model.predict(out[self.feature_cols])
        out[cfg.score_col] = out[cfg.prediction_col]
        return out

    def construct_portfolio(self, prediction_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        DataValidator.require_columns(prediction_df, [cfg.date_col, cfg.asset_col, cfg.score_col])

        def per_date(g: pd.DataFrame) -> pd.DataFrame:
            ranked = g.sort_values(cfg.score_col, ascending=False).copy()
            n = len(ranked)
            n_top = max(1, int(np.floor(cfg.top_quantile * n)))
            n_bottom = max(1, int(np.floor(cfg.bottom_quantile * n)))
            ranked["weight"] = 0.0
            ranked["side"] = "flat"
            top = ranked.index[:n_top]
            ranked.loc[top, ["side", "weight"]] = ["long", 1.0 / n_top]
            if not cfg.long_only:
                bottom = ranked.index[-n_bottom:]
                ranked.loc[bottom, ["side", "weight"]] = ["short", -1.0 / n_bottom]
            return ranked

        portfolio = prediction_df.groupby(cfg.date_col, group_keys=False).apply(per_date)
        return self.apply_industry_neutral(portfolio)

    def apply_industry_neutral(self, portfolio_df: pd.DataFrame) -> pd.DataFrame:
        cfg = self.config
        if cfg.industry_col not in portfolio_df.columns:
            return portfolio_df
        out = portfolio_df.copy()
        out["weight"] = out["weight"] - out.groupby([cfg.date_col, cfg.industry_col])["weight"].transform("mean")
        gross = out.groupby(cfg.date_col)["weight"].transform(lambda s: s.abs().sum())
        out["weight"] = np.where(gross > 0, out["weight"] / gross, 0.0)
        return out

    def backtest(self, portfolio_df: pd.DataFrame, realized_return_col: Optional[str] = None) -> pd.DataFrame:
        cfg = self.config
        realized_col = realized_return_col or cfg.target_col
        DataValidator.require_columns(portfolio_df, [cfg.date_col, cfg.asset_col, "weight", realized_col])
        df = portfolio_df.copy().sort_values([cfg.asset_col, cfg.date_col])
        df["prev_weight"] = df.groupby(cfg.asset_col)["weight"].shift(1).fillna(0.0)
        df["turnover_component"] = (df["weight"] - df["prev_weight"]).abs()
        df["holding_continuity_flag"] = ((df["weight"] != 0) & (df["prev_weight"] != 0)).astype(int)
        df["gross_pnl"] = df["weight"] * df[realized_col]
        cost_rate = (cfg.transaction_cost_bps + cfg.slippage_bps) / 10_000
        df["trading_cost"] = df["turnover_component"] * cost_rate
        df["net_pnl"] = df["gross_pnl"] - df["trading_cost"]

        summary = df.groupby(cfg.date_col).agg(
            gross_return=("gross_pnl", "sum"),
            net_return=("net_pnl", "sum"),
            turnover=("turnover_component", "sum"),
            holding_continuity=("holding_continuity_flag", "mean"),
            benchmark_return=(cfg.benchmark_col, "first"),
        ).reset_index()
        summary["excess_return"] = summary["net_return"] - summary["benchmark_return"].fillna(0.0)
        summary["cum_net_return"] = (1 + summary["net_return"]).cumprod() - 1
        summary["cum_excess_return"] = (1 + summary["excess_return"]).cumprod() - 1
        return summary


class PhotonIndustryLowFrequencyTradeModel(BaseLowFrequencyTradeModel):
    def default_feature_columns(self) -> List[str]:
        return ["ret_1m", "ret_6m", "volatility_1m", "order_growth", "inventory_growth", "gross_margin_ttm", "pe_ttm", "ps_ttm"]

    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"order_growth", "inventory_growth"}.issubset(out.columns):
            out["photon_cycle_score"] = out["order_growth"] - out["inventory_growth"]
        return out


class EnergyIndustryLowFrequencyTradeModel(BaseLowFrequencyTradeModel):
    def default_feature_columns(self) -> List[str]:
        return ["ret_1m", "ret_12m", "volatility_3m", "beta_oil", "fcf_yield", "debt_to_equity", "ev_ebitda", "production_growth"]


class AIHardwareIndustryLowFrequencyTradeModel(BaseLowFrequencyTradeModel):
    def default_feature_columns(self) -> List[str]:
        return ["ret_1m", "ret_3m", "ret_6m", "volatility_3m", "rev_growth_yoy", "gross_margin_ttm", "capex_ratio", "inventory_days", "pe_fwd", "ps_fwd"]

    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"rev_growth_yoy", "gross_margin_ttm", "inventory_days"}.issubset(out.columns):
            out["ai_hardware_supply_demand_score"] = out["rev_growth_yoy"] + out["gross_margin_ttm"] - 0.01 * out["inventory_days"]
        return out


class QuantumIndustryLowFrequencyTradeModel(BaseLowFrequencyTradeModel):
    def default_feature_columns(self) -> List[str]:
        return [
            "ret_1m",
            "ret_3m",
            "volatility_3m",
            "upstream_exposure",
            "platform_exposure",
            "application_exposure",
            "pqc_exposure",
            "partnership_count",
            "contract_score",
            "government_dependency_score",
            "commercialization_stage_score",
            "technology_bottleneck_score",
            "capex_cycle_score",
        ]

    def custom_feature_engineering(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if {"upstream_exposure", "technology_bottleneck_score", "commercialization_stage_score"}.issubset(out.columns):
            out["quantum_supply_demand_quality_score"] = (
                out["commercialization_stage_score"]
                + 0.4 * out["upstream_exposure"]
                - 0.5 * out["technology_bottleneck_score"]
            )
        return out


class EpochAITypePredictor:
    def __init__(self, config: Optional[EpochAIConfig] = None):
        self.config = config or EpochAIConfig()
        self.feature_cols: List[str] = []
        self.model = None
        self.is_fitted = False

    def default_feature_columns(self) -> List[str]:
        return [
            "market_return_1m",
            "market_vol_1m",
            "rate_change_3m",
            "credit_spread",
            "semiconductor_momentum",
            "ai_capex_growth",
            "compute_supply_growth",
            "valuation_spread_growth_vs_value",
        ]

    def build_model(self):
        if Pipeline is None:
            raise ImportError("Install scikit-learn to train models.")
        estimator = LogisticRegression(max_iter=1000) if self.config.use_classifier else Ridge(alpha=1.0)
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("model", estimator),
        ])

    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df.copy()
        if not self.feature_cols:
            self.feature_cols = self.default_feature_columns()
        DataValidator.require_columns(out, self.feature_cols)
        return out

    def fit(self, train_df: pd.DataFrame):
        DataValidator.require_columns(train_df, [self.config.target_col])
        prepared = self.prepare_features(train_df).dropna(subset=[self.config.target_col])
        if len(prepared) < self.config.min_train_rows:
            raise ValueError("Not enough training rows.")
        self.model = self.build_model()
        self.model.fit(prepared[self.feature_cols], prepared[self.config.target_col])
        self.is_fitted = True
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before predict.")
        prepared = self.prepare_features(df)
        out = prepared.copy()
        out[self.config.prediction_col] = self.model.predict(prepared[self.feature_cols])
        if self.config.use_classifier and hasattr(self.model, "predict_proba"):
            proba = self.model.predict_proba(prepared[self.feature_cols])
            out[self.config.probability_col] = proba.max(axis=1)
        return out


def industry_model_factory(industry: str, cfg: Optional[LowFrequencyModelConfig] = None) -> BaseLowFrequencyTradeModel:
    mapping = {
        "photonics": PhotonIndustryLowFrequencyTradeModel,
        "energy": EnergyIndustryLowFrequencyTradeModel,
        "ai_hardware": AIHardwareIndustryLowFrequencyTradeModel,
        "quantum": QuantumIndustryLowFrequencyTradeModel,
    }
    model_cls = mapping.get(industry, BaseLowFrequencyTradeModel)
    return model_cls(cfg)
