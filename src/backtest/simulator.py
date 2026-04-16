from __future__ import annotations

import pandas as pd


def run_backtest(model, panel_df: pd.DataFrame) -> pd.DataFrame:
    model.fit(panel_df)
    pred = model.predict(panel_df)
    portfolio = model.construct_portfolio(pred)
    return model.backtest(portfolio)
