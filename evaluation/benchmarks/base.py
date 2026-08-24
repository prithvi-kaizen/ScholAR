from __future__ import annotations

from typing import Any, Protocol
from pydantic import BaseModel, Field


class GoldEvidence(BaseModel):
    source_id: str
    page: int | None = None
    section: str | None = None
    text_span: str | None = None
    bbox: list[float] | None = None


class QAExample(BaseModel):
    example_id: str
    dataset: str
    document_id: str
    question: str
    gold_answers: list[str] = Field(default_factory=list)
    gold_evidence: list[GoldEvidence] = Field(default_factory=list)
    answerable: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class BenchmarkAdapter(Protocol):
    def load_examples(self, split: str = "test") -> list[QAExample]: ...
    def compute_metrics(self, predictions: list[dict[str, Any]]) -> dict[str, float]: ...
