from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


SOURCES = {
    "sec_edgar": {
        "where": [
            "https://www.sec.gov/search-filings/edgar-application-programming-interfaces",
        ],
        "grab": [
            "10-K/10-Q/8-K text",
            "business description",
            "customers/contracts",
            "R&D, capex, revenue mix",
            "government exposure",
            "filing_date / available_date",
        ],
    },
    "company_ir": {
        "where": [
            "company investor relations pages",
            "company press release pages",
            "company product/platform pages",
        ],
        "grab": [
            "partnership announcements",
            "system launches",
            "cloud/platform access",
            "customer case studies",
            "commercial milestones",
        ],
    },
    "nist_pqc": {
        "where": [
            "https://www.nccoe.nist.gov/applied-cryptography/migration-to-pqc",
            "https://csrc.nist.gov/News/2025/pqc-migration-mappings-to-risk-framework-documents",
        ],
        "grab": [
            "PQC migration urgency",
            "security control mapping",
            "procurement/governance signals",
        ],
    },
    "industry_reports": {
        "where": [
            "QED-C reports",
            "CNAS quantum supply chain reports",
        ],
        "grab": [
            "critical upstream bottlenecks",
            "component categories",
            "manufacturing chokepoints",
            "technology readiness signals",
        ],
    },
}

TARGET_FIELDS = [
    "upstream_exposure",
    "platform_exposure",
    "application_exposure",
    "pqc_exposure",
    "partnership_count",
    "contract_score",
    "government_dependency_score",
    "commercialization_stage_score",
    "technology_bottleneck_score",
]


@dataclass
class LLMWorkflowConfig:
    provider: str = "reserved"
    api_key: str = ""
    model: str = "gpt-reserved"
    enabled: bool = False


@dataclass
class QuantumCompanyProfile:
    ticker: str
    tags: Dict[str, float]
    region: str = "US"


@dataclass
class QuantumBusinessEvent:
    ticker: str
    date: pd.Timestamp
    available_date: pd.Timestamp
    event_type: str
    score: float
    metadata: Dict[str, str] = field(default_factory=dict)


@dataclass
class QuantumDocument:
    ticker: str
    source: str
    publish_date: pd.Timestamp
    available_date: pd.Timestamp
    text: str


class QuantumIndustryBusinessAnalysisEngine:
    """Convert quantum industry chain research into trainable monthly factor panel."""

    KEYWORD_MAP = {
        "upstream_exposure": ["cryogenic", "laser", "photonics", "microwave", "rf", "fabrication", "packaging"],
        "platform_exposure": ["qpu", "runtime", "compiler", "cloud quantum", "full-stack", "error correction"],
        "application_exposure": ["optimization", "simulation", "chemistry", "materials", "finance", "manufacturing"],
        "pqc_exposure": ["pqc", "post-quantum", "nist", "migration", "cryptography"],
        "government_dependency_score": ["government", "defense", "dod", "grant", "public contract"],
        "technology_bottleneck_score": ["bottleneck", "yield", "lead time", "shortage", "stability"],
    }

    EVENT_WEIGHT = {
        "partnership": 0.6,
        "contract": 1.0,
        "policy": 0.7,
        "research": 0.4,
        "milestone": 0.9,
    }

    def __init__(self, llm_config: Optional[LLMWorkflowConfig] = None):
        self.company_profiles: Dict[str, QuantumCompanyProfile] = {}
        self.events: List[QuantumBusinessEvent] = []
        self.documents: List[QuantumDocument] = []
        self.llm_config = llm_config or LLMWorkflowConfig()

    def add_company_profile(self, profile: QuantumCompanyProfile):
        self.company_profiles[profile.ticker] = profile

    def add_event(self, event: QuantumBusinessEvent):
        self.events.append(event)

    def add_document(self, doc: QuantumDocument):
        self.documents.append(doc)

    def llm_event_scoring_workflow(self) -> List[Dict[str, str]]:
        """Reserved agentic workflow for future decision automation."""
        return [
            {"step": "ingest", "description": "Collect SEC/IR/NIST/report content and keep available_date metadata."},
            {"step": "extract", "description": "Extract entities, event_type, contract value, policy relevance, and bottleneck tags."},
            {"step": "judge", "description": "LLM + rule ensemble scores event impact and commercialization stage shift."},
            {"step": "evolve", "description": "Agent proposes new factor candidates and validation hypotheses."},
            {"step": "approve", "description": "Human-in-the-loop review updates factor registry and production rules."},
            {"step": "backtest", "description": "Run alpha pipeline and compare uplift vs baseline factors."},
        ]

    def _safe_tag(self, ticker: str, tag: str) -> float:
        profile = self.company_profiles.get(ticker)
        if profile is None:
            return 0.0
        return float(profile.tags.get(tag, 0.0))

    def _keyword_score(self, text: str, keys: List[str]) -> float:
        lowered = text.lower()
        hits = sum(1 for k in keys if k in lowered)
        return min(1.0, hits / max(1, len(keys)))

    def _document_features(self, ticker: str, as_of_date: pd.Timestamp) -> Dict[str, float]:
        docs = [d for d in self.documents if d.ticker == ticker and d.available_date <= as_of_date]
        if not docs:
            return {k: 0.0 for k in self.KEYWORD_MAP}

        merged = "\n".join(d.text for d in docs[-30:])
        out = {k: self._keyword_score(merged, keys) for k, keys in self.KEYWORD_MAP.items()}
        return out

    def _event_aggregates(self, ticker: str, as_of_date: pd.Timestamp) -> Dict[str, float]:
        events = [e for e in self.events if e.ticker == ticker and e.available_date <= as_of_date]
        if not events:
            return {
                "partnership_count": 0.0,
                "contract_score": 0.0,
                "commercialization_stage_score": 0.0,
                "capex_cycle_score": 0.0,
            }

        weighted = np.array([
            e.score * self.EVENT_WEIGHT.get(e.event_type, 0.5)
            for e in events
        ])
        partnership_count = sum(1 for e in events if e.event_type == "partnership")
        contract_score = float(sum(e.score for e in events if e.event_type == "contract"))
        return {
            "partnership_count": float(partnership_count),
            "contract_score": contract_score,
            "commercialization_stage_score": float(np.tanh(weighted.sum() / 4.0)),
            "capex_cycle_score": float(np.tanh(weighted[-6:].sum() / 2.0)),
        }

    def build_monthly_factor_panel(self, dates: pd.DatetimeIndex, tickers: List[str]) -> pd.DataFrame:
        rows = []
        for date in dates:
            for ticker in tickers:
                doc_features = self._document_features(ticker, date)
                event_features = self._event_aggregates(ticker, date)
                upstream_exposure = 0.45 * self._safe_tag(ticker, "upstream") + 0.35 * doc_features["upstream_exposure"]
                platform_exposure = 0.45 * self._safe_tag(ticker, "platform") + 0.35 * doc_features["platform_exposure"]
                application_exposure = 0.45 * self._safe_tag(ticker, "application") + 0.35 * doc_features["application_exposure"]
                pqc_exposure = 0.4 * self._safe_tag(ticker, "security") + 0.5 * doc_features["pqc_exposure"]

                rows.append({
                    "date": pd.to_datetime(date),
                    "ticker": ticker,
                    "upstream_exposure": float(np.clip(upstream_exposure, 0.0, 1.0)),
                    "platform_exposure": float(np.clip(platform_exposure, 0.0, 1.0)),
                    "application_exposure": float(np.clip(application_exposure, 0.0, 1.0)),
                    "pqc_exposure": float(np.clip(pqc_exposure, 0.0, 1.0)),
                    "partnership_count": event_features["partnership_count"],
                    "contract_score": event_features["contract_score"],
                    "government_dependency_score": doc_features["government_dependency_score"],
                    "commercialization_stage_score": event_features["commercialization_stage_score"],
                    "technology_bottleneck_score": doc_features["technology_bottleneck_score"],
                    "capex_cycle_score": event_features["capex_cycle_score"],
                    "available_date": pd.to_datetime(date),
                })

        return pd.DataFrame(rows)

    def workflow_metadata(self) -> Dict[str, object]:
        return {
            "sources": SOURCES,
            "target_fields": TARGET_FIELDS,
            "llm_reserved": asdict(self.llm_config),
            "agentic_workflow": self.llm_event_scoring_workflow(),
        }
