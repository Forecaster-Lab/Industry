from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass
class PortfolioConstraints:
    max_single_name_weight: float = 0.1
    max_industry_weight: float = 0.3
    gross_exposure_target: float = 1.0


def summarize_exposure(weights_by_asset: Dict[str, float]) -> Dict[str, float]:
    gross = sum(abs(v) for v in weights_by_asset.values())
    net = sum(weights_by_asset.values())
    return {"gross": gross, "net": net}
