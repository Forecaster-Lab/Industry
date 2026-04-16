from __future__ import annotations

import pandas as pd


def make_fundamental_features(fundamental: pd.DataFrame) -> pd.DataFrame:
    out = fundamental.copy()
    out["quality_value_blend"] = out["gross_margin_ttm"] + out["fcf_yield"] - 0.3 * out["debt_to_equity"]
    out["growth_capex_cycle"] = out["rev_growth_yoy"] + 0.5 * out["capex_ratio"] - out["inventory_growth"]
    return out
