import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.services.answer_pipeline import resolve_conversational_query


class Phase4StreamingTelemetryTest(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_benchmark_summary_endpoint(self):
        response = self.client.get("/api/benchmark/summary")
        self.assertEqual(response.status_code, 200)
        data = response.json()

        self.assertIn("provenance", data)
        self.assertIn("empirical_summary", data)
        self.assertIn("live_telemetry", data)
        self.assertIn("reasoning_levels", data)
        self.assertIn("hardware_tiers", data)

        provenance = data["provenance"]
        self.assertIn("status", provenance)
        self.assertIn("release_provenance", provenance)
        self.assertIn("is_empirical", provenance)

        # Hardware tiers have latency targets declared
        for tier in data["hardware_tiers"]:
            self.assertIn("tier", tier)
            self.assertIn("latency_target", tier)

        # Reasoning levels have target projection status declared
        for lvl in data["reasoning_levels"]:
            self.assertIn("level", lvl)
            self.assertEqual(lvl.get("status"), "target_projection")

    def test_resolve_conversational_query_no_history(self):
        query = "What is the learning rate of Adam?"
        resolved = resolve_conversational_query(query, [])
        self.assertEqual(resolved, query)

    def test_resolve_conversational_query_with_referent(self):
        history = [
            {"role": "user", "content": "What is Adam optimizer?"},
            {"role": "assistant", "content": "Adam is a stochastic optimization method with adaptive moments."},
        ]
        query = "Does it outperform SGD?"
        resolved = resolve_conversational_query(query, history)
        self.assertIn("context:", resolved)
        self.assertIn("Adam", resolved)

    def test_resolve_conversational_query_elliptical(self):
        history = [
            {"role": "user", "content": "What is the accuracy of ResNet-50 on ImageNet?"},
            {"role": "assistant", "content": "ResNet-50 achieves 76.15% top-1 accuracy."},
        ]
        query = "What about for the larger model?"
        resolved = resolve_conversational_query(query, history)
        self.assertIn("context:", resolved)
        self.assertIn("ResNet-50", resolved)

    def test_resolve_conversational_query_independent_not_altered(self):
        history = [
            {"role": "user", "content": "What is Adam optimizer?"},
        ]
        query = "Who are the authors of Attention Is All You Need?"
        resolved = resolve_conversational_query(query, history)
        self.assertEqual(resolved, query)

    def test_chat_stream_emits_stage_events(self):
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            metadata_path = root / "metadata.json"
            pages_path = root / "pages.json"
            chunks_path = root / "chunks.json"
            pdf_path = root / "document.pdf"

            metadata_path.write_text(
                json.dumps({
                    "title": "Retrieval Augmented Benchmarks",
                    "authors": ["A. Author"],
                    "summary": "Dense and sparse retrieval benchmarks.",
                }),
                encoding="utf-8",
            )
            pages_path.write_text("[]", encoding="utf-8")
            chunks_path.write_text(
                json.dumps([
                    {
                        "chunk_id": "chunk_0",
                        "page": 1,
                        "section_title": "Results",
                        "chunk_type": "text",
                        "quote": "Hybrid retrieval improves MRR by 8% over dense retrieval alone.",
                        "text": "Hybrid retrieval improves MRR by 8% over dense retrieval alone.",
                    }
                ]),
                encoding="utf-8",
            )
            pdf_path.write_bytes(b"%PDF-1.4 mock")

            unavailable = unittest.mock.AsyncMock(return_value=False)

            with (
                patch(
                    "backend.services.answer_pipeline._paper_paths",
                    return_value=(metadata_path, pages_path, chunks_path, pdf_path),
                ),
                patch("backend.services.answer_pipeline.paper_dir", return_value=root),
                patch("backend.services.answer_pipeline.ollama_available", unavailable),
                patch(
                    "backend.services.retrieval_service.DenseEmbeddingService.search_dense",
                    side_effect=lambda _paper, _query, chunks, top_k: [(chunks[0], 1.0)],
                ),
                patch("backend.services.telemetry_service.TelemetryService.persist_trace") as trace_mock,
            ):
                response = self.client.post(
                    "/api/papers/paper/chat/stream",
                    json={"message": "What improves retrieval accuracy?"},
                )

        self.assertEqual(response.status_code, 200)
        events = []
        event_payloads = {}
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
            data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
            parsed_data = json.loads(data)
            events.append(event)
            event_payloads.setdefault(event, []).append(parsed_data)

        # Standard API ordering preserved
        self.assertEqual(events[0:2], ["analysis", "evidence_path"])
        self.assertIn("stage", events)
        self.assertIn("token", events)
        self.assertEqual(events[-2:], ["verification", "done"])

        # Real stage events were emitted
        stage_names = [st["stage"] for st in event_payloads["stage"]]
        self.assertIn("load_paper", stage_names)
        self.assertIn("retrieval", stage_names)
        for st in event_payloads["stage"]:
            self.assertIn("duration_ms", st)
            self.assertEqual(st.get("status"), "ok")


if __name__ == "__main__":
    unittest.main()
