"""Network-boundary regressions for strict-local analysis mode."""

import asyncio
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.capabilities import ModelRegistry
from backend.services.network_policy_service import (
    NetworkMode,
    NetworkPolicyError,
    NetworkPolicyService,
)
from backend.services.pdf_service import download_pdf
from backend.services.visual_embedding_service import VisualEmbeddingService


class TestNetworkPolicy(unittest.TestCase):
    def test_loopback_url_validation_rejects_remote_and_ambiguous_hosts(self) -> None:
        for url in (
            "http://localhost:11434",
            "https://127.0.0.1:8000",
            "http://127.42.1.9:9000",
            "http://[::1]:11434",
        ):
            with self.subTest(url=url):
                self.assertTrue(NetworkPolicyService.is_loopback_url(url))
        for url in (
            "https://api.example.com",
            "http://192.168.1.5:11434",
            "http://user:pass@localhost:11434",
            "file:///tmp/model",
            "not-a-url",
        ):
            with self.subTest(url=url):
                self.assertFalse(NetworkPolicyService.is_loopback_url(url))

    def test_strict_local_blocks_acquisition_and_remote_model_endpoints(self) -> None:
        with patch.dict(os.environ, {"SCHOLAR_NETWORK_MODE": "strict-local"}):
            self.assertEqual(NetworkPolicyService.current_mode(), NetworkMode.STRICT_LOCAL)
            with self.assertRaises(NetworkPolicyError) as acquisition:
                NetworkPolicyService.require_acquisition("download-paper", "https://arxiv.org/a.pdf")
            self.assertEqual(acquisition.exception.code, "NETWORK_REQUIRED")
            NetworkPolicyService.require_local_endpoint("http://127.0.0.1:11434", "Ollama")
            with self.assertRaises(NetworkPolicyError) as endpoint:
                NetworkPolicyService.require_local_endpoint("https://models.example.com", "Ollama")
            self.assertEqual(endpoint.exception.code, "NON_LOOPBACK_ENDPOINT")

    def test_acquisition_enabled_allows_explicit_external_operation(self) -> None:
        with patch.dict(os.environ, {"SCHOLAR_NETWORK_MODE": "acquisition-enabled"}):
            NetworkPolicyService.require_acquisition("search-arxiv", "https://export.arxiv.org")
            NetworkPolicyService.require_local_endpoint("https://explicit-model.example", "model")

    def test_pdf_download_is_denied_before_creating_destination_or_client(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"SCHOLAR_NETWORK_MODE": "strict-local"}
        ):
            destination = Path(tmpdir) / "not-created" / "paper.pdf"
            with self.assertRaises(NetworkPolicyError):
                asyncio.run(download_pdf("https://arxiv.org/pdf/1706.03762", destination))
            self.assertFalse(destination.parent.exists())

    def test_model_discovery_rejects_non_loopback_before_http(self) -> None:
        with patch.dict(os.environ, {"SCHOLAR_NETWORK_MODE": "strict-local"}):
            with self.assertRaises(NetworkPolicyError):
                asyncio.run(ModelRegistry.discover_ollama_models("https://models.example.com"))

    def test_visual_encoder_artifact_digest_changes_with_weights_or_processor(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            weights = root / "model.safetensors"
            processor = root / "preprocessor_config.json"
            weights.write_bytes(b"weights-v1")
            processor.write_text('{"size":224}', encoding="utf-8")
            first, _, count = VisualEmbeddingService._digest_encoder_artifacts(
                str(root), None
            )
            self.assertEqual(count, 2)

            weights.write_bytes(b"weights-v2")
            second, _, _ = VisualEmbeddingService._digest_encoder_artifacts(
                str(root), None
            )
            self.assertNotEqual(first, second)

            processor.write_text('{"size":336}', encoding="utf-8")
            third, _, _ = VisualEmbeddingService._digest_encoder_artifacts(
                str(root), None
            )
            self.assertNotEqual(second, third)

    def test_visual_status_distinguishes_model_load_from_runtime_failure_and_recovery(self) -> None:
        import torch

        class Processor:
            def __call__(self, **_kwargs):
                return {"input_ids": torch.ones((1, 1), dtype=torch.int64)}

        class FailingModel:
            def get_text_features(self, **_kwargs):
                raise RuntimeError("simulated encoding failure")

        class WorkingModel:
            def get_text_features(self, **_kwargs):
                return torch.tensor([[3.0, 4.0]], dtype=torch.float32)

        with patch.multiple(
            VisualEmbeddingService,
            _is_initialized=True,
            _available=True,
            _attempted=True,
            _processor=Processor(),
            _model=FailingModel(),
            _device="cpu",
            _load_failure_reason=None,
            _runtime_failure_reason=None,
            _last_request_succeeded=None,
        ):
            self.assertIsNone(VisualEmbeddingService.encode_texts(["query"]))
            failed = VisualEmbeddingService.status()
            self.assertTrue(failed["model_loaded"])
            self.assertFalse(failed["active"])
            self.assertIn("simulated encoding failure", failed["runtime_failure_reason"])

            VisualEmbeddingService._model = WorkingModel()
            vectors = VisualEmbeddingService.encode_texts(["query"])
            recovered = VisualEmbeddingService.status()
            self.assertIsNotNone(vectors)
            self.assertTrue(recovered["active"])
            self.assertTrue(recovered["last_request_succeeded"])
            self.assertIsNone(recovered["runtime_failure_reason"])

    def test_policy_endpoint_reports_actions_and_missing_assets(self) -> None:
        with patch.dict(os.environ, {"SCHOLAR_NETWORK_MODE": "strict-local"}):
            response = TestClient(app).get("/api/system/network-policy")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["mode"], "strict-local")
        self.assertFalse(payload["external_network_allowed"])
        actions = {item["action"]: item for item in payload["actions"]}
        self.assertTrue(actions["search-arxiv"]["requires_external_network"])
        self.assertFalse(actions["search-arxiv"]["allowed"])
        self.assertTrue(actions["analyze-prepared-paper"]["allowed"])


if __name__ == "__main__":
    unittest.main()
