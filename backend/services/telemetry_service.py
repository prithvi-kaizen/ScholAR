"""Telemetry & Audit Trace Service for ScholAR.

Records full end-to-end execution traces for every scientific question:
- Reasoning Level (L1 to L5)
- Decomposed subqueries & roles
- Evidence Graph topology
- Deterministic NumericPlan calculations
- 3-Way Atomic Claim verification decisions
- Latency & local hardware profiling
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from pathlib import Path
from typing import Any

from backend.schemas.answer_trace import AnswerTrace
from backend.schemas.claims import VerificationReport
from backend.schemas.evidence_graph import ReasoningPath
from backend.schemas.numeric_plan import NumericExecutionResult
from backend.schemas.reasoning import QuestionAnalysis
from backend.services.pdf_service import write_json

logger = logging.getLogger("scholar.telemetry")

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "backend" / "data" / "traces"
TRACES_DIR.mkdir(parents=True, exist_ok=True)


class TelemetryService:
    """Persists typed answer traces and legacy streaming telemetry records."""

    @classmethod
    def persist_trace(cls, trace: AnswerTrace) -> AnswerTrace:
        """Atomically persist the exact trace returned by the answer pipeline."""
        trace_file = TRACES_DIR / f"{trace.trace_id}.json"
        try:
            trace.persistence_succeeded = True
            write_json(trace_file, trace.model_dump(mode="json"))
            logger.info("Recorded answer trace [%s] for paper [%s]", trace.trace_id, trace.paper_id)
        except Exception as exc:
            trace.persistence_succeeded = False
            logger.warning("Could not persist answer trace: %s", exc)
        return trace

    @classmethod
    def record_trace(
        cls,
        paper_id: str,
        query: str,
        analysis: QuestionAnalysis,
        reasoning_path: ReasoningPath | None = None,
        numeric_result: NumericExecutionResult | None = None,
        verification_report: VerificationReport | None = None,
        latency_ms: float = 0.0,
        hardware_tier: str = "16GB",
    ) -> dict[str, Any]:
        """Serialize and persist a complete execution trace."""
        trace_id = f"trace_{uuid.uuid4().hex[:12]}"

        trace_data: dict[str, Any] = {
            "trace_id": trace_id,
            "paper_id": paper_id,
            "query": query,
            "reasoning_level": analysis.reasoning_level.value,
            "target_modalities": [m.value for m in analysis.target_modalities],
            "subqueries": [sq.model_dump() for sq in analysis.subqueries],
            "reasoning_path": [step.model_dump() for step in (reasoning_path.steps if reasoning_path else [])],
            "numeric_plan": numeric_result.model_dump() if numeric_result else None,
            "verification_report": verification_report.model_dump() if verification_report else None,
            "latency_ms": round(latency_ms, 2),
            "hardware_tier": hardware_tier,
            "timestamp": time.time(),
        }

        trace_file = TRACES_DIR / f"{trace_id}.json"
        try:
            trace_file.write_text(json.dumps(trace_data, indent=2), encoding="utf-8")
            logger.info("Recorded telemetry trace [%s] for paper [%s]", trace_id, paper_id)
        except Exception as exc:
            logger.warning("Could not persist telemetry trace: %s", exc)

        return trace_data
