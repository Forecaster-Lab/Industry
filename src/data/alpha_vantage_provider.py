from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import pandas as pd
import requests

DEFAULT_ALPHA_VANTAGE_API_KEY = "H1FT1GT2JTLHD7GK"


@dataclass
class AlphaVantageConfig:
    api_key: str
    base_url: str = "https://www.alphavantage.co/query"
    request_pause_seconds: float = 13.0
    timeout_seconds: int = 60


class AlphaVantageProvider:
    def __init__(self, config: Optional[AlphaVantageConfig] = None):
        api_key = os.getenv("ALPHA_VANTAGE_API_KEY", DEFAULT_ALPHA_VANTAGE_API_KEY)
        self.config = config or AlphaVantageConfig(api_key=api_key)
        self.session = requests.Session()

    def _request_json(self, params: dict) -> dict:
        if not self.config.api_key:
            raise ValueError("ALPHA_VANTAGE_API_KEY is not set.")
        resp = self.session.get(
            self.config.base_url,
            params={**params, "apikey": self.config.api_key},
            timeout=self.config.timeout_seconds,
        )
        resp.raise_for_status()
        data = resp.json()
        if "Error Message" in data:
            raise RuntimeError(data["Error Message"])
        if "Note" in data:
            raise RuntimeError(data["Note"])
        time.sleep(self.config.request_pause_seconds)
        return data

    def get_daily_adjusted(self, symbol: str, outputsize: str = "full") -> pd.DataFrame:
        """Fetch daily OHLCV data using the free TIME_SERIES_DAILY endpoint."""
        key = "Time Series (Daily)"

        data = self._request_json(
            {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": outputsize}
        )

        if key not in data:
            raise RuntimeError(f"Unexpected response for {symbol}: {list(data.keys())}")

        df = pd.DataFrame.from_dict(data[key], orient="index").reset_index()
        df = df.rename(
            columns={
                "index": "date",
                "4. close": "close",
                "5. volume": "volume",
            }
        )
        df["date"] = pd.to_datetime(df["date"])
        for c in ["close", "volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        df["ticker"] = symbol
        return df[["date", "ticker", "close", "volume"]].sort_values("date").reset_index(drop=True)

    def get_ohlcv_panel(self, tickers: Iterable[str], outputsize: str = "full") -> pd.DataFrame:
        frames: List[pd.DataFrame] = []
        for t in tickers:
            frames.append(self.get_daily_adjusted(t, outputsize=outputsize))
        return pd.concat(frames, ignore_index=True)
