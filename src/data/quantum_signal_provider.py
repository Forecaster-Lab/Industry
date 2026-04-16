from __future__ import annotations

import pandas as pd

from src.features.quantum_business_engine import QuantumBusinessEvent, QuantumCompanyProfile, QuantumDocument


def demo_quantum_profiles() -> list[QuantumCompanyProfile]:
    return [
        QuantumCompanyProfile("AAA", {"upstream": 1, "platform": 0, "application": 0, "security": 0}),
        QuantumCompanyProfile("BBB", {"upstream": 0, "platform": 1, "application": 0, "security": 0}),
        QuantumCompanyProfile("CCC", {"upstream": 0, "platform": 0, "application": 1, "security": 1}),
        QuantumCompanyProfile("DDD", {"upstream": 1, "platform": 1, "application": 0, "security": 0}),
    ]


def demo_quantum_events() -> list[QuantumBusinessEvent]:
    rows = [
        ("AAA", "2024-01-08", "partnership", 0.7),
        ("BBB", "2024-02-11", "contract", 1.2),
        ("CCC", "2024-03-19", "policy", 0.9),
        ("DDD", "2024-04-25", "milestone", 1.0),
        ("BBB", "2024-06-13", "research", 0.5),
    ]
    return [
        QuantumBusinessEvent(
            ticker=ticker,
            date=pd.to_datetime(date),
            available_date=pd.to_datetime(date),
            event_type=event_type,
            score=score,
        )
        for ticker, date, event_type, score in rows
    ]


def demo_quantum_documents() -> list[QuantumDocument]:
    docs = [
        QuantumDocument(
            ticker="AAA",
            source="sec_edgar",
            publish_date=pd.to_datetime("2024-01-31"),
            available_date=pd.to_datetime("2024-01-31"),
            text="Cryogenic systems and photonics packaging expansion; manufacturing lead time bottleneck remains.",
        ),
        QuantumDocument(
            ticker="BBB",
            source="company_ir",
            publish_date=pd.to_datetime("2024-02-29"),
            available_date=pd.to_datetime("2024-02-29"),
            text="QPU cloud quantum runtime launch with full-stack compiler and error correction roadmap.",
        ),
        QuantumDocument(
            ticker="CCC",
            source="nist_pqc",
            publish_date=pd.to_datetime("2024-03-31"),
            available_date=pd.to_datetime("2024-03-31"),
            text="Post-quantum cryptography migration planning with defense procurement and governance mapping.",
        ),
        QuantumDocument(
            ticker="DDD",
            source="industry_reports",
            publish_date=pd.to_datetime("2024-04-30"),
            available_date=pd.to_datetime("2024-04-30"),
            text="RF microwave control, fabrication yield improvement, and application optimization for advanced manufacturing.",
        ),
    ]
    return docs
