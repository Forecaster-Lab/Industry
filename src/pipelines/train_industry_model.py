from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

import pandas as pd

from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine
from src.models.industry_low_frequency_models import LowFrequencyModelConfig, industry_model_factory
from src.pipelines.build_dataset import build_dataset


def _apply_factor_weight_score(predicted: pd.DataFrame, factor_weights: Dict[str, float]) -> pd.DataFrame:
    out = predicted.copy()
    selected = {k: float(v) for k, v in factor_weights.items() if k in out.columns}
    if not selected:
        return out

    weighted_signal = pd.Series(0.0, index=out.index)
    for col, weight in selected.items():
        weighted_signal = weighted_signal + out[col].fillna(0.0) * weight

    out["weighted_factor_score"] = weighted_signal
    out["score"] = out["weighted_factor_score"]
    out["prediction"] = out["weighted_factor_score"]
    return out


def train_industry_model(
    industry: str,
    model_type: str,
    feature_columns: List[str] | None = None,
    factor_weights: Dict[str, float] | None = None,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
    dataset_source: str = "alpha_vantage",
    tickers: List[str] | None = None,
) -> Dict:
    panel = build_dataset(include_quantum_business=True, dataset_source=dataset_source, tickers=tickers)
    train_df = panel[panel["industry"] == industry].dropna(subset=["future_return"])

    config = LowFrequencyModelConfig(model_type=model_type, top_quantile=top_quantile, bottom_quantile=bottom_quantile)
    model = industry_model_factory(industry, config)
    if feature_columns:
        model.feature_cols = feature_columns

    model.fit(train_df)
    predicted = model.predict(train_df)

    applied_weights = factor_weights or {}
    if applied_weights:
        predicted = _apply_factor_weight_score(predicted, applied_weights)

    portfolio = model.construct_portfolio(predicted)
    backtest = model.backtest(portfolio)

    graph_payload = {
        "dates": backtest["date"].dt.strftime("%Y-%m-%d").tolist(),
        "cum_net_return": backtest["cum_net_return"].round(8).tolist(),
        "cum_excess_return": backtest["cum_excess_return"].round(8).tolist(),
        "net_return": backtest["net_return"].round(8).tolist(),
        "factor_weights": applied_weights,
    }

    payload = {
        "config": asdict(config),
        "feature_columns": model.feature_cols,
        "metrics": {
            "last_cum_return": float(backtest["cum_net_return"].iloc[-1]),
            "last_cum_excess_return": float(backtest["cum_excess_return"].iloc[-1]),
            "avg_turnover": float(backtest["turnover"].mean()),
            "avg_holding_continuity": float(backtest["holding_continuity"].mean()),
        },
        "model_graph": graph_payload,
        "backtest": backtest.to_dict(orient="records"),
    }

    if industry == "quantum":
        metadata_engine = QuantumIndustryBusinessAnalysisEngine(LLMWorkflowConfig(enabled=False))
        payload["quantum_business_workflow"] = metadata_engine.workflow_metadata()

    return payload
