from __future__ import annotations

import pandas as pd


def merge_feature_panels(
    price: pd.DataFrame,
    fundamental: pd.DataFrame,
    macro: pd.DataFrame,
    universe: pd.DataFrame,
    quantum_business_panel: pd.DataFrame | None = None,
) -> pd.DataFrame:
    panel = price.merge(fundamental, on=["date", "ticker"], how="left")
    panel = panel.merge(universe, on="ticker", how="left")
    panel = panel.merge(macro, on="date", how="left")

    if quantum_business_panel is not None and not quantum_business_panel.empty:
        keep_cols = [
            "date",
            "ticker",
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
            "available_date",
        ]
        safe_quantum = quantum_business_panel[keep_cols].copy()
        panel = panel.merge(safe_quantum, on=["date", "ticker"], how="left")

    panel["future_return"] = panel.groupby("ticker")["ret_1m"].shift(-1)
    return panel
