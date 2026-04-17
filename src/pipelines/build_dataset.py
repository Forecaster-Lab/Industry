from __future__ import annotations

import pandas as pd

from src.data.alpha_vantage_provider import AlphaVantageProvider
from src.data.base_provider import QueryContext
from src.data.fundamentals_provider import FundamentalsProvider
from src.data.macro_provider import MacroProvider
from src.data.ohlcv_provider import OHLCVProvider
from src.data.quantum_signal_provider import demo_quantum_documents, demo_quantum_events, demo_quantum_profiles
from src.data.storage import LocalDataRegion
from src.data.universe_provider import UniverseProvider
from src.features.fundamental_features import make_fundamental_features
from src.features.merge import merge_feature_panels
from src.features.price_features import make_price_features
from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine


DEFAULT_TICKERS = ["NVDA", "AMD", "AVGO", "MSFT"]


def build_dataset(
    start_date: str = "2024-01-31",
    end_date: str = "2025-12-31",
    include_quantum_business: bool = True,
    dataset_source: str = "alpha_vantage",
    tickers: list[str] | None = None,
):
    context = QueryContext(start_date=start_date, end_date=end_date)
    region = LocalDataRegion()

    use_tickers = tickers or DEFAULT_TICKERS
    source = dataset_source.lower().strip()

    if source == "alpha_vantage":
        try:
            ohlcv = AlphaVantageProvider().get_ohlcv_panel(use_tickers, outputsize="full")
        except Exception:
            ohlcv = OHLCVProvider().fetch(context)
    else:
        ohlcv = OHLCVProvider().fetch(context)

    fundamentals = FundamentalsProvider().fetch(context)
    macro = MacroProvider().fetch(context)
    universe = UniverseProvider().fetch(context)

    region.write_raw("ohlcv", ohlcv)
    region.write_raw("fundamentals", fundamentals)
    region.write_raw("macro", macro)
    region.write_raw("universe", universe)

    price_features = make_price_features(ohlcv)
    fundamental_features = make_fundamental_features(fundamentals)

    quantum_panel = None
    if include_quantum_business:
        engine = QuantumIndustryBusinessAnalysisEngine(llm_config=LLMWorkflowConfig(enabled=False))
        for p in demo_quantum_profiles():
            engine.add_company_profile(p)
        for e in demo_quantum_events():
            engine.add_event(e)
        for d in demo_quantum_documents():
            engine.add_document(d)

        monthly_dates = pd.DatetimeIndex(sorted(price_features["date"].dropna().unique()))
        ticker_list = sorted(price_features["ticker"].dropna().unique().tolist())
        quantum_panel = engine.build_monthly_factor_panel(monthly_dates, ticker_list)

    panel = merge_feature_panels(price_features, fundamental_features, macro, universe, quantum_panel)
    region.write_processed("model_panel", panel)
    return panel
