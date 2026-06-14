from __future__ import annotations

import threading
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import APP_CONFIG
from src.features.ff5_engine import FF5_FACTOR_DESCRIPTIONS
from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine
from src.models.industry_low_frequency_models import LowFrequencyModelConfig
from src.pipelines.train_industry_model import train_industry_model

# ---------------------------------------------------------------------------
# Training progress tracker
# ---------------------------------------------------------------------------

_progress: Dict[str, dict] = {}
_progress_lock = threading.Lock()


class TrainRequest(BaseModel):
    industry: str = Field(default="ai_hardware")
    model_type: str = Field(default="ridge")
    feature_columns: Optional[List[str]] = None
    factor_weights: Optional[Dict[str, float]] = None
    dataset_source: str = Field(default="alpha_vantage")
    tickers: Optional[List[str]] = None
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    neutralize_by: List[str] = Field(default_factory=list)
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0


app = FastAPI(title="Industry Analysis Platform", version="0.3.0")

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")


@app.get("/api/options")
def options():
    return {
        "available_model_types": APP_CONFIG.defaults.available_model_types,
        "industries": list(APP_CONFIG.default_feature_map.keys()),
        "default_feature_map": APP_CONFIG.default_feature_map,
        "database_reserved": asdict(APP_CONFIG.database),
        "dataset_sources": ["alpha_vantage", "synthetic"],
        "default_dataset_source": APP_CONFIG.defaults.default_dataset_source,
        "ff5_factors": {
            "enabled": True,
            "factor_names": list(FF5_FACTOR_DESCRIPTIONS.keys()),
            "factor_descriptions": FF5_FACTOR_DESCRIPTIONS,
            "reference": "Fama & French (2015) — A five-factor asset pricing model",
        },
    }


@app.get("/api/ff5/factors")
def ff5_factors():
    return {
        "factor_descriptions": FF5_FACTOR_DESCRIPTIONS,
        "construction_method": {
            "mkt": "Market-cap rank percentile within cross-section (size proxy)",
            "smb": "Triple-sorted Small Minus Big: average of B/M, OP, and Inv sorts",
            "hml": "2x3 size-B/M sort: High (top 30%) minus Low (bottom 30%) B/M",
            "rmw": "2x3 size-OP sort: Robust (top 30% operating profitability) minus Weak (bottom 30%)",
            "cma": "2x3 size-Inv sort: Conservative (bottom 30% asset growth) minus Aggressive (top 30%)",
        },
        "reference": "Fama & French (2015), Journal of Financial Economics 116, 1-22",
    }


@app.get("/api/quantum/workflow")
def quantum_workflow():
    engine = QuantumIndustryBusinessAnalysisEngine(
        llm_config=LLMWorkflowConfig(provider="reserved", model="gpt-reserved", enabled=False)
    )
    return engine.workflow_metadata()


@app.post("/api/train")
def train(req: TrainRequest):
    """Start async training, return task_id immediately."""
    import uuid as _uuid
    task_id = _uuid.uuid4().hex[:12]

    def _set_progress(pct: float, stage: str):
        with _progress_lock:
            _progress[task_id] = {
                "pct": pct, "stage": stage, "task_id": task_id,
                "done": pct >= 100 or pct < 0,
                "error": None,
            }

    _set_progress(0, "Initializing...")

    def _run():
        try:
            _set_progress(5, "Building dataset...")
            cfg = LowFrequencyModelConfig(
                model_type=req.model_type,
                top_quantile=req.top_quantile,
                bottom_quantile=req.bottom_quantile,
                neutralize_by=req.neutralize_by,
                transaction_cost_bps=req.transaction_cost_bps,
                slippage_bps=req.slippage_bps,
            )
            _set_progress(15, "Computing features & FF5 factors...")
            result = train_industry_model(
                industry=req.industry,
                model_type=cfg.model_type,
                feature_columns=req.feature_columns,
                factor_weights=req.factor_weights,
                top_quantile=cfg.top_quantile,
                bottom_quantile=cfg.bottom_quantile,
                dataset_source=req.dataset_source,
                tickers=req.tickers,
            )
            _set_progress(85, "Building response...")
            result["notes"] = [
                "Database interface is reserved.",
                "FF5 (Fama-French five-factor) exposures are included as features for all industries.",
            ]
            with _progress_lock:
                _progress[task_id] = {
                    "pct": 100, "stage": "Done", "task_id": task_id,
                    "done": True, "result": result, "error": None,
                }
        except Exception as exc:
            with _progress_lock:
                _progress[task_id] = {
                    "pct": -1, "stage": str(exc)[:120], "task_id": task_id,
                    "done": True, "result": None, "error": str(exc),
                }

    threading.Thread(target=_run, daemon=True).start()
    return {"task_id": task_id, "status": "started"}


@app.get("/api/train/progress")
def train_progress(task_id: str = ""):
    """Poll training progress by task_id."""
    with _progress_lock:
        info = _progress.get(task_id, {"pct": 0, "stage": "Idle", "task_id": task_id, "done": False})
    return info
