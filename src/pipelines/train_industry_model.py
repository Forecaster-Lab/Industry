from __future__ import annotations

from dataclasses import asdict
from typing import Dict, List

from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine
from src.models.industry_low_frequency_models import LowFrequencyModelConfig, industry_model_factory
from src.pipelines.build_dataset import build_dataset


def train_industry_model(
    industry: str,
    model_type: str,
    feature_columns: List[str] | None = None,
    top_quantile: float = 0.2,
    bottom_quantile: float = 0.2,
) -> Dict:
    panel = build_dataset(include_quantum_business=True)
    train_df = panel[panel["industry"] == industry].dropna(subset=["future_return"])

    config = LowFrequencyModelConfig(model_type=model_type, top_quantile=top_quantile, bottom_quantile=bottom_quantile)
    model = industry_model_factory(industry, config)
    if feature_columns:
        model.feature_cols = feature_columns

    model.fit(train_df)
    predicted = model.predict(train_df)
    portfolio = model.construct_portfolio(predicted)
    backtest = model.backtest(portfolio)

    payload = {
        "config": asdict(config),
        "feature_columns": model.feature_cols,
        "metrics": {
            "last_cum_return": float(backtest["cum_net_return"].iloc[-1]),
            "last_cum_excess_return": float(backtest["cum_excess_return"].iloc[-1]),
            "avg_turnover": float(backtest["turnover"].mean()),
            "avg_holding_continuity": float(backtest["holding_continuity"].mean()),
        },
        "backtest": backtest.to_dict(orient="records"),
    }

    if industry == "quantum":
        metadata_engine = QuantumIndustryBusinessAnalysisEngine(LLMWorkflowConfig(enabled=False))
        payload["quantum_business_workflow"] = metadata_engine.workflow_metadata()

    return payload
