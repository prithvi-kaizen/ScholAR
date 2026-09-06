"""Clean-fixture strict-local smoke with process-level outbound socket denial."""

import asyncio
import json
import os
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

from backend.schemas.answer_trace import (
    AnswerPipelineRequest,
    ExecutionPolicy,
    GenerationMode,
    PipelineStatus,
)
from backend.services.answer_pipeline import AnswerPipelineService
from backend.services.network_policy_service import NetworkPolicyService
from backend.services.telemetry_service import TelemetryService


class TestStrictOfflineExecution(unittest.TestCase):
    def test_prepared_fixture_answer_completes_while_outbound_sockets_are_blocked(self) -> None:
        denied_attempts: list[str] = []
        real_getaddrinfo = socket.getaddrinfo
        real_connect = socket.socket.connect
        real_create_connection = socket.create_connection

        def is_loopback(host: object) -> bool:
            rendered = str(host)
            return rendered == "localhost" or rendered == "::1" or rendered.startswith("127.")

        def guarded_getaddrinfo(host: object, *args: object, **kwargs: object):
            if not is_loopback(host):
                denied_attempts.append(f"dns:{host}")
                raise AssertionError(f"outbound DNS denied for {host}")
            return real_getaddrinfo(host, *args, **kwargs)

        def guarded_connect(sock: socket.socket, address: object):
            host = address[0] if isinstance(address, tuple) and address else address
            if not is_loopback(host):
                denied_attempts.append(f"connect:{host}")
                raise AssertionError(f"outbound socket denied for {host}")
            return real_connect(sock, address)

        def guarded_create_connection(address: object, *args: object, **kwargs: object):
            host = address[0] if isinstance(address, tuple) and address else address
            if not is_loopback(host):
                denied_attempts.append(f"create:{host}")
                raise AssertionError(f"outbound connection denied for {host}")
            return real_create_connection(address, *args, **kwargs)

        with tempfile.TemporaryDirectory() as tmpdir:
            paper_root = Path(tmpdir) / "strict_fixture"
            paper_root.mkdir()
            chunk = {
                "chunk_id": "chunk_001",
                "evidence_id": "E_001",
                "document_id": "strict_fixture",
                "source_paper_id": "strict_fixture",
                "page": 1,
                "section": "Results",
                "section_title": "Results",
                "chunk_type": "result",
                "paragraph_text": "Grounded retrieval improves scientific search accuracy on the local fixture.",
                "text": "Grounded retrieval improves scientific search accuracy on the local fixture.",
            }
            (paper_root / "metadata.json").write_text(
                json.dumps({"title": "Strict-local fixture", "authors": [], "summary": "Local fixture."}),
                encoding="utf-8",
            )
            (paper_root / "pages.json").write_text(
                json.dumps([{"page": 1, "text": chunk["text"]}]), encoding="utf-8"
            )
            (paper_root / "chunks.json").write_text(json.dumps([chunk]), encoding="utf-8")
            (paper_root / "paper.pdf").write_bytes(b"%PDF-local-fixture")

            strict_env = {
                "SCHOLAR_NETWORK_MODE": "strict-local",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }
            with (
                patch.dict(os.environ, strict_env),
                patch("socket.getaddrinfo", side_effect=guarded_getaddrinfo),
                patch("socket.socket.connect", new=guarded_connect),
                patch("socket.create_connection", side_effect=guarded_create_connection),
                patch("backend.services.answer_pipeline.paper_dir", return_value=paper_root),
                patch("backend.services.answer_pipeline.ollama_available", AsyncMock(return_value=False)),
                patch.object(TelemetryService, "persist_trace", side_effect=lambda trace: trace),
            ):
                self.assertTrue(NetworkPolicyService.is_strict_local())
                trace = asyncio.run(AnswerPipelineService.answer(AnswerPipelineRequest(
                    paper_id="strict_fixture",
                    query="What improves scientific search accuracy?",
                    execution_policy=ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK,
                )))

        self.assertEqual(denied_attempts, [])
        self.assertEqual(trace.status, PipelineStatus.SUCCESS)
        self.assertEqual(trace.generation.mode, GenerationMode.EXTRACTIVE_FALLBACK)
        self.assertEqual(trace.schema_version, "1.0")
        self.assertTrue(trace.prompt_evidence)
        self.assertTrue(trace.verification.reverified)
        self.assertIn("scientific search accuracy", trace.final_answer)


if __name__ == "__main__":
    unittest.main()
