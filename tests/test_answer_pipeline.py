"""Regression tests for the shared, versioned answer orchestration contract."""

import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.answer_trace import (
    AnswerPipelineRequest,
    AnswerTrace,
    ExecutionPolicy,
    GenerationMode,
    InterventionControls,
    PipelineStatus,
    RepairMode,
)
from backend.services.answer_pipeline import AnswerPipelineService
from backend.services.ollama_service import LocalGenerationResult
from backend.services.telemetry_service import TelemetryService


class TestAnswerPipeline(unittest.TestCase):
    def _paper(self, root: Path) -> list[dict]:
        chunks = [
            {
                "chunk_id": "chunk_001",
                "evidence_id": "E_001",
                "document_id": "paper",
                "source_paper_id": "paper",
                "page": 1,
                "section": "Results",
                "section_title": "Results",
                "chunk_type": "result",
                "paragraph_text": "We show that grounded retrieval improves scientific search accuracy on the benchmark.",
                "text": "We show that grounded retrieval improves scientific search accuracy on the benchmark.",
                "bm25_score": 2.4,
                "bm25_rank": 1,
                "dense_score": 0.72,
                "dense_rank": 2,
                "modality_score": 0.8,
                "modality_rank": 3,
                "rrf_score": 0.04,
                "rerank_score": 0.9,
            }
        ]
        (root / "metadata.json").write_text(
            json.dumps({"title": "Trace Test", "authors": [], "summary": "Grounded retrieval."}),
            encoding="utf-8",
        )
        (root / "pages.json").write_text(json.dumps([{"page": 1, "text": chunks[0]["text"]}]), encoding="utf-8")
        (root / "chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
        (root / "paper.pdf").write_bytes(b"%PDF-test")
        return chunks

    def _run(self, policy: ExecutionPolicy) -> AnswerTrace:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks = self._paper(root)
            with (
                patch("backend.services.answer_pipeline.paper_dir", return_value=root),
                patch("backend.services.answer_pipeline.retrieve_chunks", return_value=chunks),
                patch("backend.services.answer_pipeline.ollama_available", AsyncMock(return_value=False)),
                patch.object(TelemetryService, "persist_trace", side_effect=lambda trace: trace),
            ):
                return asyncio.run(AnswerPipelineService.answer(AnswerPipelineRequest(
                    paper_id="paper",
                    query="What improves scientific search accuracy?",
                    execution_policy=policy,
                )))

    def test_fallback_trace_is_versioned_and_round_trips(self) -> None:
        trace = self._run(ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK)
        self.assertEqual(trace.schema_version, "1.0")
        self.assertEqual(trace.status, PipelineStatus.SUCCESS)
        self.assertEqual(trace.generation.mode, GenerationMode.EXTRACTIVE_FALLBACK)
        self.assertTrue(trace.prompt_evidence)
        self.assertEqual(trace.prompt_evidence[0].identity.source_id, "paper")
        self.assertTrue(any(hit.shown_to_generator for hit in trace.retrieval_hits))
        hit = trace.retrieval_hits[0]
        self.assertEqual((hit.bm25_rank, hit.dense_rank, hit.modality_rank), (1, 2, 3))
        self.assertGreaterEqual(len(hit.query_channel_results), 1)
        self.assertTrue(all(item.dense_rank == 2 for item in hit.query_channel_results))
        self.assertGreater(trace.latency_ms, 0.0)
        restored = AnswerTrace.model_validate(trace.model_dump(mode="json"))
        self.assertEqual(restored.trace_id, trace.trace_id)
        response = restored.to_chat_response()
        self.assertEqual(response["trace_schema_version"], "1.0")
        self.assertIn("trace", response)

    def test_measured_policy_rejects_extractively_degraded_execution(self) -> None:
        trace = self._run(ExecutionPolicy.REQUIRE_LOCAL_MODEL)
        self.assertEqual(trace.status, PipelineStatus.ERROR)
        self.assertEqual(trace.generation.mode, GenerationMode.NO_GENERATION)
        self.assertIn("unavailable", trace.generation.error or "")

    def test_chat_route_delegates_once_and_returns_the_v1_trace(self) -> None:
        trace = self._run(ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK)
        with patch(
            "backend.main.AnswerPipelineService.answer",
            AsyncMock(return_value=trace),
        ) as answer:
            response = TestClient(app).post(
                "/api/papers/paper/chat",
                json={
                    "message": "What improves accuracy?",
                    "intervention": {
                        "repair_mode": "NONE",
                        "abstain_on_no_supported_claims": False,
                    },
                    "decoding": {
                        "temperature": 0.2,
                        "top_p": 0.8,
                        "num_ctx": 4096,
                        "num_predict": 512,
                    },
                },
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["trace_schema_version"], "1.0")
        self.assertEqual(response.json()["trace"]["trace_id"], trace.trace_id)
        answer.assert_awaited_once()
        delegated = answer.await_args.args[0]
        self.assertEqual(delegated.intervention.repair_mode, RepairMode.NONE)
        self.assertEqual(delegated.decoding.num_ctx, 4096)

    def test_pipeline_returns_mutated_second_pass_verified_answer(self) -> None:
        raw_answer = (
            "Grounded retrieval improves scientific search accuracy on the benchmark [E1]. "
            "It also enables quantum navigation across distant galaxies [E1]."
        )
        generation = LocalGenerationResult(
            response=raw_answer,
            requested_model="test-local",
            resolved_model="test-local",
            options={"temperature": 0.1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks = self._paper(root)
            with (
                patch("backend.services.answer_pipeline.paper_dir", return_value=root),
                patch("backend.services.answer_pipeline.retrieve_chunks", return_value=chunks),
                patch("backend.services.answer_pipeline.ollama_available", AsyncMock(return_value=True)),
                patch("backend.services.answer_pipeline.generate_result", AsyncMock(return_value=generation)),
                patch.object(TelemetryService, "persist_trace", side_effect=lambda trace: trace),
            ):
                trace = asyncio.run(AnswerPipelineService.answer(AnswerPipelineRequest(
                    paper_id="paper",
                    query="What improves scientific search accuracy?",
                    requested_model="test-local",
                    execution_policy=ExecutionPolicy.REQUIRE_LOCAL_MODEL,
                )))

        self.assertEqual(trace.status, PipelineStatus.SUCCESS)
        self.assertEqual(trace.raw_answer, raw_answer)
        self.assertNotEqual(trace.final_answer, trace.normalized_answer)
        self.assertIn("Grounded retrieval improves scientific search accuracy", trace.final_answer)
        self.assertNotIn("quantum navigation", trace.final_answer)
        self.assertTrue(trace.verification.answer_text_changed)
        self.assertTrue(trace.verification.reverified)
        self.assertTrue(trace.verification.edits)
        self.assertTrue(all(
            edit.original_text != edit.replacement_text
            for edit in trace.verification.edits
        ))
        self.assertEqual(trace.verification_report, trace.verification.report)
        self.assertEqual(trace.verification_report.final_verified_response, trace.final_answer)
        self.assertTrue(trace.verification_report.second_pass_completed)
        self.assertEqual(trace.intervention.executed_repair_mode, RepairMode.SELECTIVE)

    def test_no_repair_condition_preserves_unsupported_text_and_is_traced(self) -> None:
        raw_answer = (
            "Grounded retrieval improves scientific search accuracy on the benchmark [E1]. "
            "It also enables quantum navigation across distant galaxies [E1]."
        )
        generation = LocalGenerationResult(
            response=raw_answer,
            requested_model="test-local",
            resolved_model="test-local",
            options={"temperature": 0.1},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            chunks = self._paper(root)
            with (
                patch("backend.services.answer_pipeline.paper_dir", return_value=root),
                patch("backend.services.answer_pipeline.retrieve_chunks", return_value=chunks),
                patch("backend.services.answer_pipeline.ollama_available", AsyncMock(return_value=True)),
                patch("backend.services.answer_pipeline.generate_result", AsyncMock(return_value=generation)),
                patch.object(TelemetryService, "persist_trace", side_effect=lambda trace: trace),
            ):
                trace = asyncio.run(AnswerPipelineService.answer(AnswerPipelineRequest(
                    paper_id="paper",
                    query="What improves scientific search accuracy?",
                    requested_model="test-local",
                    execution_policy=ExecutionPolicy.REQUIRE_LOCAL_MODEL,
                    intervention=InterventionControls(
                        repair_mode=RepairMode.NONE,
                        abstain_on_no_supported_claims=False,
                    ),
                )))
        self.assertEqual(trace.status, PipelineStatus.SUCCESS)
        self.assertIn("quantum navigation", trace.final_answer)
        self.assertEqual(trace.final_answer, trace.normalized_answer)
        self.assertFalse(trace.verification.repair_requested)
        self.assertEqual(trace.verification.edits, [])
        self.assertEqual(trace.intervention.executed_repair_mode, RepairMode.NONE)
        self.assertTrue(trace.intervention.verification_reached)

    def test_trace_persistence_is_atomic_and_validated(self) -> None:
        trace = self._run(ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK)
        with tempfile.TemporaryDirectory() as tmpdir, patch(
            "backend.services.telemetry_service.TRACES_DIR", Path(tmpdir)
        ):
            persisted = TelemetryService.persist_trace(trace)
            payload = json.loads((Path(tmpdir) / f"{trace.trace_id}.json").read_text(encoding="utf-8"))
        self.assertTrue(persisted.persistence_succeeded)
        self.assertEqual(AnswerTrace.model_validate(payload).trace_id, trace.trace_id)


if __name__ == "__main__":
    unittest.main()
