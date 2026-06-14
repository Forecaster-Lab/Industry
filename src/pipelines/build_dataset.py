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
from src.features.ff5_engine import FF5Config, FF5Engine
from src.features.fundamental_features import make_fundamental_features
from src.features.merge import merge_feature_panels
from src.features.price_features import make_price_features
from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine


DEFAULT_TICKERS = ["NVDA", "AMD", "AVGO", "MSFT", "XOM", "SHEL", "IONQ", "RGTI"]
_DEFAULT_INDUSTRIES = ["ai_hardware", "ai_hardware", "photonics", "ai_hardware",
                        "energy", "energy", "quantum", "quantum"]


def build_dataset(
    start_date: str = "2024-01-31",
    end_date: str = "2025-12-31",
    include_quantum_business: bool = True,
    include_ff5_factors: bool = True,
    dataset_source: str = "alpha_vantage",
    tickers: list[str] | None = None,
):
    use_tickers = tickers or DEFAULT_TICKERS
    source = dataset_source.lower().strip()
    context = QueryContext(start_date=start_date, end_date=end_date, tickers=use_tickers)
    region = LocalDataRegion()

    if source == "alpha_vantage":
        try:
            ohlcv = AlphaVantageProvider().get_ohlcv_panel(use_tickers, outputsize="compact")
        except Exception as e:
            raise RuntimeError(
                f"Alpha Vantage API failed: {e}. "
                f"Free tier is limited to 25 calls/day; 8 tickers use 8 calls per run. "
                f"Use dataset_source='synthetic' for development, or wait for the daily reset."
            ) from e
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

    # ---- Fama-French Five Factor exposures ----
    # Skip FF5 when fewer than 6 tickers (cross-sectional sorts need breadth)
    ff5_panel = None
    if include_ff5_factors and len(use_tickers) >= 6:
        ff5_config = FF5Config(mode="industry")
        ff5_engine = FF5Engine(ff5_config)
        ff5_panel = ff5_engine.compute_factor_exposures(fundamentals, price_features, universe)
    elif include_ff5_factors:
        # Not enough tickers for meaningful FF5 — skip silently
        pass

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

    panel = merge_feature_panels(price_features, fundamental_features, macro, universe, quantum_panel, ff5_panel)
    region.write_processed("model_panel", panel)
    return panel
