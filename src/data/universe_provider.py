from __future__ import annotations

import pandas as pd

from .base_provider import BaseDataProvider, QueryContext


class UniverseProvider(BaseDataProvider):
    _DEFAULT_INDUSTRIES = {
        "NVDA": "ai_hardware", "AMD": "ai_hardware", "AVGO": "photonics", "MSFT": "ai_hardware",
        "XOM": "energy", "SHEL": "energy", "IONQ": "quantum", "RGTI": "quantum",
    }

    def fetch(self, context: QueryContext) -> pd.DataFrame:
        all_industries = ["ai_hardware", "energy", "photonics", "quantum"]
        tickers = context.tickers or list(self._DEFAULT_INDUSTRIES.keys())
        industries = [self._DEFAULT_INDUSTRIES.get(t, all_industries[i % len(all_industries)])
                       for i, t in enumerate(tickers)]
        return pd.DataFrame({"ticker": tickers, "industry": industries})
