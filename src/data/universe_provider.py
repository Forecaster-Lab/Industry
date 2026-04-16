from __future__ import annotations

import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class UniverseProvider(BaseDataProvider):
    def fetch(self, context: QueryContext) -> pd.DataFrame:
        tickers = ["AAA", "BBB", "CCC", "DDD"]
        industries = ["ai_hardware", "energy", "photonics", "ai_hardware"]
        return pd.DataFrame({"ticker": tickers, "industry": industries})
