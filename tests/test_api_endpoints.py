import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.routing import APIRoute
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.evidence import EvidenceAST, EvidenceBlock


class TestApiEndpoints(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)

    def test_health_endpoint(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data.get("status"), "ok")
        self.assertIn("model", data)

    def test_models_discovery_endpoint(self):
        response = self.client.get("/api/models")
        self.assertEqual(response.status_code, 200)
        models = response.json()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)

        first = models[0]
        self.assertIn("model_id", first)
        self.assertIn("supports_vision", first)
        self.assertIn("supports_text", first)
        self.assertIn("capability_mode", first)

    def test_canonical_ast_route_is_unique_and_validated(self):
        matching = [
            route
            for route in app.routes
            if isinstance(route, APIRoute)
            and route.path == "/api/papers/{paper_id}/ast"
            and "GET" in route.methods
        ]
        self.assertEqual(len(matching), 1)
        self.assertIs(matching[0].response_model, EvidenceAST)

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_path = Path(tmpdir) / "paper"
            paper_path.mkdir()
            ast = EvidenceAST(
                document_id="paper",
                title="Canonical AST",
                page_count=1,
                blocks=[
                    EvidenceBlock(
                        evidence_id="E_001",
                        document_id="paper",
                        page=1,
                        text="Canonical evidence.",
                    )
                ],
            )
            (paper_path / "evidence_ast.json").write_text(
                json.dumps(ast.model_dump()), encoding="utf-8"
            )
            with patch("backend.main.paper_dir", return_value=paper_path):
                response = self.client.get("/api/papers/paper/ast")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["document_id"], "paper")
        self.assertEqual(response.json()["blocks"][0]["evidence_id"], "E_001")

    def test_telemetry_trace_listing_reads_valid_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            traces_dir = Path(tmpdir)
            (traces_dir / "valid.json").write_text(
                json.dumps({"trace_id": "trace_valid", "latency_ms": 12.5}),
                encoding="utf-8",
            )
            (traces_dir / "invalid.json").write_text("{not-json", encoding="utf-8")
            with patch("backend.services.telemetry_service.TRACES_DIR", traces_dir):
                response = self.client.get("/api/telemetry/traces")

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), [{"trace_id": "trace_valid", "latency_ms": 12.5}])

    def test_stream_events_are_json_and_record_measured_trace(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            metadata_path = root / "metadata.json"
            pages_path = root / "pages.json"
            chunks_path = root / "chunks.json"
            pdf_path = root / "paper.pdf"
            metadata_path.write_text(json.dumps({"title": "Stream Test"}), encoding="utf-8")
            pages_path.write_text(json.dumps([{"page": 1, "text": "Evidence"}]), encoding="utf-8")
            chunks_path.write_text(
                json.dumps(
                    [
                        {
                            "chunk_id": "chunk_001",
                            "document_id": "paper",
                            "source_paper_id": "paper",
                            "page": 1,
                            "section_title": "Results",
                            "section_path": ["Results"],
                            "chunk_type": "result",
                            "paragraph_text": "The method improves retrieval accuracy.",
                            "text": "The method improves retrieval accuracy.",
                        }
                    ]
                ),
                encoding="utf-8",
            )
            pdf_path.write_bytes(b"%PDF-test")

            async def unavailable():
                return False

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
                patch(
                    "backend.services.telemetry_service.TelemetryService.persist_trace"
                ) as trace,
            ):
                response = self.client.post(
                    "/api/papers/paper/chat/stream",
                    json={"message": "What improves retrieval accuracy?"},
                )

        self.assertEqual(response.status_code, 200)
        events = []
        for block in response.text.strip().split("\n\n"):
            lines = block.splitlines()
            event = next(line.removeprefix("event: ") for line in lines if line.startswith("event: "))
            data = next(line.removeprefix("data: ") for line in lines if line.startswith("data: "))
            json.loads(data)
            events.append(event)
        self.assertEqual(events[0:2], ["analysis", "evidence_path"])
        self.assertIn("token", events)
        self.assertEqual(events[-2:], ["verification", "done"])
        trace.assert_called_once()
        persisted = trace.call_args.args[0]
        self.assertGreater(persisted.latency_ms, 0.0)
        self.assertEqual(
            persisted.run_identity.retriever_version,
            "hybrid-document-visual-rrf-v3",
        )

    def test_stream_prevalidation_matches_chat_http_errors(self):
        for endpoint in ("chat", "chat/stream"):
            empty = self.client.post(
                f"/api/papers/paper/{endpoint}",
                json={"message": "   "},
            )
            self.assertEqual(empty.status_code, 400)
            self.assertIn("empty", empty.json()["detail"].lower())

        missing = FileNotFoundError("Paper 'missing-paper' has not been completely prepared")
        with patch(
            "backend.services.answer_pipeline._paper_paths",
            side_effect=missing,
        ):
            normal = self.client.post(
                "/api/papers/missing-paper/chat",
                json={"message": "What does the paper report?"},
            )
            streamed = self.client.post(
                "/api/papers/missing-paper/chat/stream",
                json={"message": "What does the paper report?"},
            )
        self.assertEqual(normal.status_code, 404)
        self.assertEqual(streamed.status_code, 404)
        self.assertEqual(normal.json(), streamed.json())


if __name__ == "__main__":
    unittest.main()
