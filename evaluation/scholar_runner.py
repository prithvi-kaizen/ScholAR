"""Validated adapter for running the exact production ScholAR answer pipeline."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from backend.services.network_policy_service import NetworkPolicyService
from backend.schemas.answer_trace import (
    AnswerTrace,
    DecodingOptions,
    EvaluationContext,
    ExecutionPolicy,
    GenerationMode,
    InterventionControls,
    PipelineStatus,
)


@dataclass(frozen=True)
class ScholarRunResult:
    response: dict[str, Any]
    trace: AnswerTrace

    @property
    def answer(self) -> str:
        return self.trace.final_answer

    @property
    def citations(self) -> list[dict[str, Any]]:
        return [citation.to_api_dict() for citation in self.trace.citations]


def run_scholar_http(
    backend: str,
    paper_id: str,
    query: str,
    model: str,
    *,
    timeout: float = 300.0,
    require_local_model: bool = True,
    history: list[dict[str, str]] | None = None,
    secondary_paper_ids: list[str] | None = None,
    experiment_id: str | None = None,
    generation_seed: int | None = None,
    intervention: InterventionControls | None = None,
    decoding: DecodingOptions | None = None,
    evaluation_context: EvaluationContext | None = None,
    allow_error_trace: bool = False,
    visual_page_backend: str = "configured",
) -> ScholarRunResult:
    """Call the loopback API route that delegates to AnswerPipelineService and validate v1."""
    if not NetworkPolicyService.is_loopback_url(backend):
        raise RuntimeError(f"Evaluation backend must be loopback-only, got {backend!r}")
    body: dict[str, Any] = {
        "message": query,
        "history": history or [],
        "secondary_paper_ids": secondary_paper_ids or [],
        "model": model,
        "execution_policy": (
            ExecutionPolicy.REQUIRE_LOCAL_MODEL.value
            if require_local_model
            else ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK.value
        ),
        "visual_page_backend": visual_page_backend,
    }
    if experiment_id:
        body["experiment_id"] = experiment_id
    if generation_seed is not None:
        body["generation_seed"] = generation_seed
    if intervention is not None:
        body["intervention"] = intervention.model_dump(mode="json")
    if decoding is not None:
        body["decoding"] = decoding.model_dump(mode="json")
    if evaluation_context is not None:
        body["evaluation_context"] = evaluation_context.model_dump(mode="json")
    request = urllib.request.Request(
        f"{backend.rstrip('/')}/api/papers/{paper_id}/chat",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"ScholAR API failed with HTTP {exc.code}: {detail}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("ScholAR API returned a non-object response")
    trace_payload = payload.get("trace")
    if not isinstance(trace_payload, dict):
        raise RuntimeError("ScholAR API response is missing its versioned answer trace")
    trace = AnswerTrace.model_validate(trace_payload)
    if trace.status == PipelineStatus.ERROR and not allow_error_trace:
        raise RuntimeError(trace.generation.error or "ScholAR answer pipeline failed")
    if require_local_model and trace.status != PipelineStatus.ERROR and trace.generation.mode not in {
        GenerationMode.LOCAL_MODEL,
        GenerationMode.VISION_MODEL,
    }:
        raise RuntimeError(
            f"Measured run required a local model but executed {trace.generation.mode.value}"
        )
    return ScholarRunResult(response=payload, trace=trace)
