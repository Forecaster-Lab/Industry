# Industry Analysis Platform

A web-based **industry analysis and low-frequency trading research platform** with:

- Reserved database integration interfaces (providers can be swapped to SQL later).
- Interactive model controls from the web UI:
  - modify/add factor columns
  - choose model type (Ridge / RandomForest / LightGBM / XGBoost / Ranker / Ensemble)
  - change factor weights payload
  - change top/bottom portfolio quantiles
- Extensible model framework:
  - `BaseLowFrequencyTradeModel`
  - cross-sectional preprocessing
  - `fit / predict / construct_portfolio / backtest`
  - per-industry default factor columns
  - extensible `EpochAITypePredictor`
- Enhanced backtest outputs:
  - turnover
  - holding continuity
  - industry neutralization
  - benchmark excess return
  - transaction cost + slippage impact

## Project structure

```text
src/
  config.py
  data/
    base_provider.py
    ohlcv_provider.py
    fundamentals_provider.py
    macro_provider.py
    universe_provider.py
    cache.py
  features/
    price_features.py
    fundamental_features.py
    merge.py
    quantum_business_engine.py
  models/
    industry_low_frequency_models.py
  backtest/
    portfolio.py
    simulator.py
  pipelines/
    build_dataset.py
    train_industry_model.py
  web/
    app.py
    static/index.html
```

## Run

```bash
pip install fastapi uvicorn pandas numpy scikit-learn
uvicorn src.web.app:app --reload --port 8000
```

Open: `http://127.0.0.1:8000`

## Database integration (reserved)

Current providers generate synthetic/demo data to keep interfaces stable.
When your database is ready, replace each provider `fetch()` implementation with SQL-backed logic.


## Quantum industry business analysis module

This repo now includes `QuantumIndustryBusinessAnalysisEngine` that converts upstream-midstream-downstream and policy/security signals into monthly business factors with `available_date` alignment.
It supports reserved LLM workflow metadata for future agentic extraction and decision loops.


## Render deploy fix

This repository now includes `render.yaml` and `Procfile` so Render detects a Python web service instead of defaulting to Node build commands.

- Build: `pip install -r requirements.txt`
- Start: `uvicorn src.web.app:app --host 0.0.0.0 --port $PORT`


### If your Render dashboard is locked to Node commands

If you must keep:
- Build Command: `npm install && npm run build`
- Start Command: `npm start` (or `npm run start`)

this repo now includes a compatibility `package.json` that creates a local `.venv` during `npm run build`, installs Python deps inside that virtualenv, and starts FastAPI from `.venv/bin/uvicorn` (PEP 668-safe).
