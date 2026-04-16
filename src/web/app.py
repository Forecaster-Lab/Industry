from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.config import APP_CONFIG
from src.features.quantum_business_engine import LLMWorkflowConfig, QuantumIndustryBusinessAnalysisEngine
from src.models.industry_low_frequency_models import LowFrequencyModelConfig
from src.pipelines.train_industry_model import train_industry_model


class TrainRequest(BaseModel):
    industry: str = Field(default="ai_hardware")
    model_type: str = Field(default="ridge")
    feature_columns: Optional[List[str]] = None
    factor_weights: Optional[Dict[str, float]] = None
    top_quantile: float = 0.2
    bottom_quantile: float = 0.2
    neutralize_by: List[str] = Field(default_factory=list)
    transaction_cost_bps: float = 5.0
    slippage_bps: float = 3.0


app = FastAPI(title="Industry Analysis Platform", version="0.2.0")

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
    }


@app.get("/api/quantum/workflow")
def quantum_workflow():
    engine = QuantumIndustryBusinessAnalysisEngine(
        llm_config=LLMWorkflowConfig(provider="reserved", model="gpt-reserved", enabled=False)
    )
    return engine.workflow_metadata()


@app.post("/api/train")
def train(req: TrainRequest):
    cfg = LowFrequencyModelConfig(
        model_type=req.model_type,
        top_quantile=req.top_quantile,
        bottom_quantile=req.bottom_quantile,
        neutralize_by=req.neutralize_by,
        transaction_cost_bps=req.transaction_cost_bps,
        slippage_bps=req.slippage_bps,
    )

    result = train_industry_model(
        industry=req.industry,
        model_type=cfg.model_type,
        feature_columns=req.feature_columns,
        top_quantile=cfg.top_quantile,
        bottom_quantile=cfg.bottom_quantile,
    )

    if req.factor_weights:
        result["applied_factor_weights"] = req.factor_weights

    result["notes"] = [
        "Database interface is reserved. Replace providers with SQL-backed providers when ready.",
        "LLM API key is reserved in workflow metadata for future agentic scoring and decision-making.",
        "Quantum industry business factors are generated with available_date alignment before model ingestion.",
    ]
    return result
