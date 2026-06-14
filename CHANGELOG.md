# Changelog

All notable changes to the Industry Analysis Platform.

---

## [0.3.0] — 2026-06-14

### Added
- **Fama-French Five-Factor Engine** (`src/features/ff5_engine.py`)
  - SMB (size), HML (value), RMW (profitability), CMA (investment), MKT exposures
  - Industry-mode and market-mode cross-sectional sorts
  - June-end annual rebalance with carry-forward between rebalance dates
- **FF5 factor API endpoint** (`GET /api/ff5/factors`)
  - Returns factor descriptions, construction methods, and academic reference
- **Training progress bar** (frontend: real backend-polled progress)
  - `POST /api/train` now returns `task_id` immediately, trains in background
  - `GET /api/train/progress?task_id=...` returns real-time progress (pct + stage)
  - Frontend polls every 600ms, shows 7-stage pipeline progress
- **Unit test suite** (`tests/`, 18 tests, pytest)
  - Covers: FF5Engine (rebalance, edge cases), winsorize, neutralize, zscore,
    build_dataset, train pipeline (all model types), error handling
- **User API Key support** in Alpha Vantage provider
- **`PROJECT.md`** — full technical documentation (pipeline logic, UI guide, math framework)

### Changed
- **Data layer** — All Providers now accept dynamic ticker lists via `QueryContext.tickers`
- **FundamentalsProvider** — Added 8 FF5-required fields: `book_equity`, `operating_profit`,
  `total_assets`, `market_cap`, `op_over_be`, `bm_ratio`, `asset_growth`, and derived ratios
- **Alpha Vantage provider** — Switched from premium `TIME_SERIES_DAILY_ADJUSTED` to free
  `TIME_SERIES_DAILY` endpoint; request interval adjusted to 13s for free tier
- **UniverseProvider** — Now covers all 4 industries (added `quantum`); deterministic
  ticker-to-industry mapping for 8 default stocks
- **Industry model classes** — All 4 industry models now include FF5 factor columns in
  their `default_feature_columns()`
- **`merge_feature_panels`** — Accepts optional `ff5_panel` parameter
- **`build_dataset`** — New `include_ff5_factors` parameter; auto-skips FF5 when < 6 tickers
- **Config** — Removed non-existent `beta_oil` factor from energy industry feature map
- **Web API** — `/api/train` returns structured JSON errors instead of 500 HTML pages
- **Web API** — `/api/options` response now includes `ff5_factors` metadata
- **Web UI** — `feature_columns` sent as `null` to backend (model decides its own features)
- **Version** — App version bumped to `0.3.0`

### Fixed
- `construct_portfolio` — `date` column lost after `groupby(group_keys=False)`; manually restored
- `build_dataset` — Alpha Vantage failures no longer silently fall back to synthetic data;
  explicit `RuntimeError` raised with actionable message
- `train_industry_model` — Added minimum data validation (< 10 rows raises clear error)
- Feature column soft-check — Missing optional columns (e.g., FF5 with few tickers) filtered
  automatically instead of crashing
- `pd.cut` duplicate bin edges — Added `duplicates="drop"` to prevent crashes on uniform data
- `min_train_rows` — Lowered from 80 to 20 for small ticker sets (8 stocks × 24 months)

---

## [0.2.0] — 2026-01 (original repository baseline)

### Added
- 4-industry analysis framework: AI Hardware, Energy, Photonics, Quantum Computing
- Multi-model support: Ridge, RandomForest, LightGBM, XGBoost, Ranker, Ensemble
- Alpha Vantage data provider (premium endpoint)
- Synthetic data providers for all layers (OHLCV, fundamentals, macro, universe)
- Quantum Industry Business Analysis Engine (rule-based, LLM workflow reserved)
- `build_dataset` → `train_industry_model` → backtest pipeline
- FastAPI web interface with interactive controls and SVG chart rendering
- Portfolio construction: long-short quantile strategy with industry neutralization
- Backtest with transaction costs and slippage modeling
