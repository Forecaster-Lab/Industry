from __future__ import annotations

import pandas as pd


def make_price_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    df = ohlcv.sort_values(["ticker", "date"]).copy()
    grouped = df.groupby("ticker")
    df["ret_1m"] = grouped["close"].pct_change(1)
    df["ret_3m"] = grouped["close"].pct_change(3)
    df["ret_6m"] = grouped["close"].pct_change(6)
    df["ret_12m"] = grouped["close"].pct_change(12)
    df["volatility_1m"] = grouped["ret_1m"].rolling(1).std().reset_index(level=0, drop=True).fillna(0.0)
    df["volatility_3m"] = grouped["ret_1m"].rolling(3).std().reset_index(level=0, drop=True)
    return df
