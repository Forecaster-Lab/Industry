from __future__ import annotations

import os

from src.data.alpha_vantage_provider import DEFAULT_ALPHA_VANTAGE_API_KEY
from src.pipelines.train_industry_model import train_industry_model


if __name__ == "__main__":
    api_key = os.getenv("ALPHA_VANTAGE_API_KEY", DEFAULT_ALPHA_VANTAGE_API_KEY)

    result = train_industry_model(
        industry="ai_hardware",
        model_type="random_forest",
        dataset_source="alpha_vantage",
        tickers=["NVDA", "AMD", "AVGO", "MSFT"],
        top_quantile=0.2,
        bottom_quantile=0.2,
    )

    print("Execution completed.")
    print("Feature columns:", result.get("feature_columns", []))
    print("Metrics:", result.get("metrics", {}))
    print("Graph points:", len(result.get("model_graph", {}).get("dates", [])))
