from __future__ import annotations

import pandas as pd

from src.data.base_provider import QueryContext
from src.data.fundamentals_provider import FundamentalsProvider
from src.data.macro_provider import MacroProvider
from src.data.ohlcv_provider import OHLCVProvider
from src.data.quantum_signal_provider import demo_quantum_documents, demo_quantum_events, demo_quantum_profiles
from src.data.universe_provider import UniverseProvider
from src.features.fundamental_features import make_fundamental_features
from src.features.merge import merge_feature_panels
from src.features.price_features import make_price_features
from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine


def build_dataset(start_date: str = "2024-01-31", end_date: str = "2025-12-31", include_quantum_business: bool = True):
    context = QueryContext(start_date=start_date, end_date=end_date)
    ohlcv = OHLCVProvider().fetch(context)
    fundamentals = FundamentalsProvider().fetch(context)
    macro = MacroProvider().fetch(context)
    universe = UniverseProvider().fetch(context)

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
        tickers = sorted(price_features["ticker"].dropna().unique().tolist())
        quantum_panel = engine.build_monthly_factor_panel(monthly_dates, tickers)

    return merge_feature_panels(price_features, fundamental_features, macro, universe, quantum_panel)
