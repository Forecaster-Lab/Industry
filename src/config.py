from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class DatabaseConfig:
    """Connection settings for a future production database."""

    enabled: bool = False
    driver: str = "postgresql"
    host: str = "localhost"
    port: int = 5432
    database: str = "industry"
    user: str = "readonly"
    password: str = ""
    pool_size: int = 5


@dataclass
class PlatformDefaults:
    """Global defaults used by APIs and model execution."""

    date_col: str = "date"
    asset_col: str = "ticker"
    industry_col: str = "industry"
    target_col: str = "future_return"
    benchmark_col: str = "benchmark_return"
    default_model_type: str = "ridge"
    default_dataset_source: str = "alpha_vantage"
    available_model_types: List[str] = field(
        default_factory=lambda: ["ridge", "random_forest", "lightgbm", "xgboost", "ranker", "ensemble"]
    )


@dataclass
class AppConfig:
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    defaults: PlatformDefaults = field(default_factory=PlatformDefaults)
    default_feature_map: Dict[str, List[str]] = field(
        default_factory=lambda: {
            "ai_hardware": [
                "ret_1m",
                "ret_3m",
                "ret_6m",
                "volatility_3m",
                "rev_growth_yoy",
                "gross_margin_ttm",
                "capex_ratio",
                "inventory_days",
                "pe_fwd",
                "ps_fwd",
            ],
            "energy": [
                "ret_1m",
                "ret_12m",
                "volatility_3m",
                "beta_oil",
                "fcf_yield",
                "debt_to_equity",
                "ev_ebitda",
                "production_growth",
            ],
            "photonics": [
                "ret_1m",
                "ret_6m",
                "volatility_1m",
                "order_growth",
                "inventory_growth",
                "gross_margin_ttm",
                "pe_ttm",
                "ps_ttm",
            ],
            "quantum": [
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
            ],
        }
    )


APP_CONFIG = AppConfig()
