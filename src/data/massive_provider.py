from __future__ import annotations

import logging
import os
import time
from typing import Optional

import pandas as pd
import requests

from .base_provider import BaseDataProvider, QueryContext

logger = logging.getLogger(__name__)


class MassiveAPIError(Exception):
    """Raised when the Massive API returns an error or is unreachable."""


class MassiveProvider(BaseDataProvider):
    """Real market data provider backed by Massive.com (formerly Polygon.io).

    Data sources:
      - OHLCV: daily bars via Massive /v2/aggs (paid plan required)
      - Fundamentals: ticker details (market cap, shares, SIC) via /v3/reference/tickers
        + full financial statements (BS, IS, CF) via /vX/reference/financials (paid plan)

    Requires a Massive API key via:
      - ``MASSIVE_API_KEY`` environment variable, or
      - keyword argument ``api_key``
    """

    BASE_URL = "https://api.massive.com"

    # ---- field mapping: Massive financials → industry platform columns ----

    # /vX/reference/financials returns nested dicts. We flatten key->value.
    # Balance sheet keys
    _BS_KEYS = {
        "assets": "total_assets",
        "noncurrent_assets": "noncurrent_assets",
        "current_assets": "current_assets",
        "fixed_assets": "fixed_assets",
        "inventory": "inventory",
        "equity": "book_equity",
        "liabilities": "total_liabilities",
        "noncurrent_liabilities": "noncurrent_liabilities",
        "current_liabilities": "current_liabilities",
        "accounts_payable": "accounts_payable",
        "long_term_debt": "long_term_debt",
    }
    # Income statement keys
    _IS_KEYS = {
        "revenues": "revenue",
        "cost_of_revenue": "cogs",
        "gross_profit": "gross_profit",
        "operating_income_loss": "operating_profit",
        "operating_expenses": "opex",
        "net_income_loss": "net_income",
        "diluted_earnings_per_share": "eps_diluted",
        "basic_earnings_per_share": "eps_basic",
    }
    # Cash flow keys
    _CF_KEYS = {
        "net_cash_flow_from_operating_activities": "cf_operating",
        "net_cash_flow_from_investing_activities": "cf_investing",
        "net_cash_flow_from_financing_activities": "cf_financing",
        "net_cash_flow": "cf_net",
    }

    def __init__(self, api_key: Optional[str] = None) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("MASSIVE_API_KEY", "")
        if not self._api_key:
            logger.warning(
                "MASSIVE_API_KEY not set — MassiveProvider will fail. "
                "Sign up at https://massive.com to get a free API key."
            )
        self._session = requests.Session()
        self._session.headers.update({"Authorization": f"Bearer {self._api_key}"})

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def fetch(self, context: QueryContext) -> pd.DataFrame:
        """Return the OHLCV + fundamentals merged panel."""
        ohlcv = self.fetch_ohlcv(context)
        fundamentals = self.fetch_fundamentals(context)
        merged = pd.merge(ohlcv, fundamentals, on=["date", "ticker"], how="outer")
        return merged

    # ------------------------------------------------------------------
    # OHLCV — Massive /v2/aggs
    # ------------------------------------------------------------------

    def fetch_ohlcv(self, context: QueryContext) -> pd.DataFrame:
        """Daily OHLCV from Massive aggregates endpoint."""
        tickers = context.tickers or []
        if not tickers:
            raise MassiveAPIError("fetch_ohlcv requires at least one ticker")

        start = context.start_date or "2024-01-01"
        end = context.end_date or "2025-12-31"

        frames: list[pd.DataFrame] = []
        for tkr in tickers:
            try:
                url = (
                    f"{self.BASE_URL}/v2/aggs/ticker/{tkr}/range/1/day/"
                    f"{start}/{end}"
                )
                resp = self._session.get(
                    url, params={"limit": 50000, "sort": "asc"}, timeout=30
                )
                resp.raise_for_status()
                data = resp.json()
                results = data.get("results", [])
            except Exception as exc:
                logger.error("Massive OHLCV failed for %s: %s", tkr, exc)
                continue

            if not results:
                logger.warning("No OHLCV data for %s", tkr)
                continue

            df = pd.DataFrame(results)
            df["ticker"] = tkr
            df.rename(
                columns={
                    "t": "date",
                    "o": "open",
                    "h": "high",
                    "l": "low",
                    "c": "close",
                    "v": "volume",
                },
                inplace=True,
            )
            df["date"] = pd.to_datetime(df["date"], unit="ms").dt.normalize()
            frames.append(
                df[["date", "ticker", "open", "high", "low", "close", "volume"]]
            )
            time.sleep(0.15)  # rate-limit respect

        if not frames:
            raise MassiveAPIError(
                "No OHLCV data returned for any ticker. "
                "Check API key, ticker symbols, and date range."
            )

        result = pd.concat(frames, ignore_index=True)
        logger.info("Massive OHLCV: %d rows across %d tickers", len(result), len(frames))
        return result

    # ------------------------------------------------------------------
    # Fundamentals — ticker details + financial statements
    # ------------------------------------------------------------------

    def fetch_fundamentals(self, context: QueryContext) -> pd.DataFrame:
        """Build fundamental panel from ticker details + financial statements."""
        tickers = context.tickers or []
        if not tickers:
            raise MassiveAPIError("fetch_fundamentals requires at least one ticker")

        rows: list[dict] = []
        for tkr in tickers:
            row = {"ticker": tkr}

            # --- step 1: ticker details ---
            self._fetch_ticker_details(tkr, row)

            # --- step 2: financial statements ---
            self._fetch_financials(tkr, row)

            rows.append(row)
            time.sleep(0.2)  # rate-limit

        result = pd.DataFrame(rows)

        if not result.empty:
            result["date"] = pd.Timestamp(context.end_date or "2025-12-31")

            # Derived fields
            if "operating_profit" in result.columns and "book_equity" in result.columns:
                be = result["book_equity"].clip(lower=1e-6)
                result["op_over_be"] = result["operating_profit"] / be

            if "book_equity" in result.columns and "market_cap" in result.columns:
                mc = result["market_cap"].clip(lower=1e-6)
                result["bm_ratio"] = result["book_equity"] / mc

            if "cogs" in result.columns and "revenue" in result.columns:
                result["gross_margin_ttm"] = (
                    (result["revenue"] - result["cogs"]) / result["revenue"].clip(lower=1e-6)
                )

            if "market_cap" in result.columns and "net_income" in result.columns:
                result["pe_ttm"] = result["market_cap"] / result["net_income"].clip(lower=1e-6)

            if "market_cap" in result.columns and "revenue" in result.columns:
                result["ps_ttm"] = result["market_cap"] / result["revenue"].clip(lower=1e-6)

            if "total_liabilities" in result.columns and "book_equity" in result.columns:
                result["debt_to_equity"] = result["total_liabilities"] / result["book_equity"].clip(lower=1e-6)

            # Fill remaining platform columns with NaN
            for col in [
                "rev_growth_yoy", "capex_ratio", "inventory_days", "fcf_yield",
                "pe_fwd", "ps_fwd", "ev_ebitda", "pb_ratio",
                "production_growth", "order_growth", "inventory_growth",
            ]:
                if col not in result.columns:
                    result[col] = float("nan")

        logger.info("Massive fundamentals: %d tickers", len(result))
        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fetch_ticker_details(self, ticker: str, row: dict) -> None:
        """Fetch market cap, shares outstanding, SIC info from /v3/reference/tickers."""
        try:
            url = f"{self.BASE_URL}/v3/reference/tickers/{ticker}"
            resp = self._session.get(url, timeout=15)
            resp.raise_for_status()
            data = resp.json()
            result = data.get("results", {})
            if result:
                row["market_cap"] = result.get("market_cap")
                row["shares_outstanding"] = result.get("weighted_shares_outstanding")
                row["sic_code"] = result.get("sic_code")
                row["sic_description"] = result.get("sic_description")
                row["homepage_url"] = result.get("homepage_url")
                row["total_employees"] = result.get("total_employees")
        except Exception as exc:
            logger.warning("Ticker details failed for %s: %s", ticker, exc)

    def _fetch_financials(self, ticker: str, row: dict) -> None:
        """Fetch financial statements via /vX/reference/financials.

        The unified endpoint returns BS, IS, and CF in a single response
        (type parameter filters; omitting type returns all three).
        """
        try:
            url = f"{self.BASE_URL}/vX/reference/financials"
            resp = self._session.get(
                url,
                params={
                    "ticker": ticker,
                    "limit": 1,
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            if not results:
                logger.warning("No financials for %s", ticker)
                return

            fin = results[0].get("financials", {})
            self._extract_financial_block(fin.get("balance_sheet", {}), self._BS_KEYS, row)
            self._extract_financial_block(fin.get("income_statement", {}), self._IS_KEYS, row)
            self._extract_financial_block(fin.get("cash_flow_statement", {}), self._CF_KEYS, row)
        except Exception as exc:
            logger.warning("Financials failed for %s: %s", ticker, exc)

    @staticmethod
    def _extract_financial_block(
        block: dict, key_map: dict, row: dict
    ) -> None:
        """Flatten nested financial dict into row using key_map.

        Massive returns values like ``{"value": 123, "unit": "USD", ...}``.
        We extract just the numeric value.
        """
        for src_key, dst_key in key_map.items():
            entry = block.get(src_key)
            if entry is None:
                continue
            if isinstance(entry, dict):
                val = entry.get("value")
            else:
                val = entry
            if val is not None:
                row[dst_key] = float(val)
