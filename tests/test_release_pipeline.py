from __future__ import annotations

import asyncio
import json
import os
import shutil
import socket
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import fitz
from pydantic import ValidationError

from backend.schemas.answer_trace import (
    AnswerPipelineRequest,
    CitationOrigin,
    GenerationMode,
    PipelineStatus,
)
from backend.services.answer_pipeline import AnswerPipelineService
from backend.services.paper_finalize_service import PaperFinalizeService
from backend.services.telemetry_service import TelemetryService
from evaluation.fixtures.releases.release_v1_minimal.build_fixture import _generate as generate_fixture_row
from evaluation.reproduce_release_fixture import main as reproduce_fixture
from evaluation.release.aggregate import aggregate_rows
from evaluation.release.identity import build_frozen_identity, expected_context
from evaluation.release.io import (
    key_index,
    load_config,
    read_json,
    read_jsonl,
    write_checksums,
    write_json,
    write_jsonl,
)
from evaluation.release.manifest import load_manifest, update_manifest
from evaluation.release.schemas import (
    CanonicalKey,
    CaseRecord,
    ExpectedKeySet,
    RawReleaseRow,
    ReleaseConfig,
    ScoredReleaseRow,
)
from evaluation.release.scoring import score_release
from evaluation.release.validate import validate_release_directory
from evaluation.run_release_suite import run_release
from evaluation.validate_paper import validate_paper


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evaluation/fixtures/releases/release_v1_minimal"


class ReleaseArtifactTests(unittest.TestCase):
    def test_fixture_reproduces_byte_identically_without_generation(self) -> None:
        with patch("evaluation.scholar_runner.run_scholar_http", side_effect=AssertionError("generation called")):
            self.assertEqual(reproduce_fixture(), 0)

    def test_failure_accounting_uses_expected_denominator(self) -> None:
        expected = ExpectedKeySet.model_validate(read_json(FIXTURE / "expected_keys.json"))
        rows = read_jsonl(FIXTURE / "scored/rows.jsonl", ScoredReleaseRow)
        summary = aggregate_rows(rows, expected, ["success_rate", "abstention_rate", "error_rate"])
        group = summary["groups"][0]
        self.assertEqual(group["n_expected"], 4)
        self.assertEqual(group["n_cases"], 2)
        self.assertEqual(group["status_counts"], {"ABSTAINED": 1, "ERROR": 1, "SUCCESS": 2})
        self.assertEqual(group["metrics"]["success_rate"], 0.5)
        self.assertEqual(group["metrics"]["error_rate"], 0.25)
        self.assertEqual(group["metric_case_denominators"]["error_rate"], 2)

    def test_duplicate_and_truncated_jsonl_are_rejected(self) -> None:
        rows = read_jsonl(FIXTURE / "raw/rows.jsonl", RawReleaseRow)
        with self.assertRaisesRegex(ValueError, "duplicate release row key"):
            key_index(rows + [rows[0]])
        with tempfile.TemporaryDirectory() as temporary:
            truncated = Path(temporary) / "rows.jsonl"
            truncated.write_bytes((FIXTURE / "raw/rows.jsonl").read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError, "truncated JSONL"):
                read_jsonl(truncated, RawReleaseRow)

    def test_checksum_tamper_and_template_contamination_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "release"
            shutil.copytree(FIXTURE, candidate)
            scored = candidate / "scored/rows.jsonl"
            scored.write_bytes(scored.read_bytes().replace(b"0.0", b"0.1", 1))
            errors = validate_release_directory(candidate)
            self.assertTrue(any("checksum mismatch: scored/rows.jsonl" in error for error in errors), errors)

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "release"
            shutil.copytree(FIXTURE, candidate)
            write_json(candidate / "raw/template.json", {
                "artifact_type": "template",
                "study_status": "NOT_STARTED",
                "contains_completed_data": False,
                "records": [],
            })
            write_checksums(candidate)
            errors = validate_release_directory(candidate)
            self.assertTrue(any("template-only artifact" in error for error in errors), errors)

    def test_ready_config_rejects_unfrozen_non_disjoint_dataset(self) -> None:
        payload = read_json(FIXTURE / "configs/release_config.json")
        payload["dataset"]["evidence_class"] = "measured"
        payload["dataset"]["claim_status"] = "current"
        payload["dataset"]["paper_disjoint_from_development"] = False
        with self.assertRaisesRegex(ValidationError, "paper-disjoint"):
            ReleaseConfig.model_validate(payload)

    def test_measured_rows_carry_the_frozen_corpus_hash(self) -> None:
        payload = read_json(FIXTURE / "configs/release_config.json")
        payload["dataset"].update(
            evidence_class="measured",
            claim_status="current",
            paper_disjoint_from_development=True,
        )
        with self.assertRaisesRegex(ValidationError, "corpus sha256"):
            ReleaseConfig.model_validate(payload)

        corpus_hash = "a" * 64
        payload["dataset"]["corpus_sha256"] = corpus_hash
        payload["prompt_hashes"] = {"fixture": "sha256:" + "b" * 64}
        config = ReleaseConfig.model_validate(payload)
        key = CanonicalKey(
            system=config.systems[0].name,
            model=config.models[0].tag,
            seed=config.seeds[0],
            case_id="case",
        )
        case = CaseRecord(case_id="case", paper_id="paper", query="query")
        identity = build_frozen_identity(
            config,
            key,
            case,
            git_revision=None,
            git_dirty=None,
        )

        self.assertEqual(identity.corpus_sha256, corpus_hash)
        self.assertEqual(expected_context(config, key).corpus_sha256, corpus_hash)

    def test_resume_skips_all_terminal_rows_including_error(self) -> None:
        fixture_config = load_config(FIXTURE / "configs/release_config.json")
        with tempfile.TemporaryDirectory(prefix="release-resume-", dir=ROOT / "evaluation") as temporary:
            output = Path(temporary) / "release"
            config = fixture_config.model_copy(update={
                "run_id": "resume-test-v1",
                "output_dir": output.relative_to(ROOT).as_posix(),
            })
            config_path = Path(temporary) / "config.json"
            write_json(config_path, config)
            calls: list[tuple[str, str, int, str]] = []

            def generator(cfg: ReleaseConfig, key, case):
                calls.append(key.as_tuple())
                return generate_fixture_row(cfg, key, case)

            run_release(config_path, generator=generator)
            self.assertEqual(len(calls), 4)
            calls.clear()

            def forbidden(*_args):
                raise AssertionError("resume retried an immutable terminal row")

            run_release(config_path, generator=forbidden)
            self.assertEqual(calls, [])
            resumed = read_jsonl(output / "raw/rows.jsonl", RawReleaseRow)
            self.assertEqual(len(resumed), 4)
            self.assertIn("ERROR", {row.status.value for row in resumed})

    def test_resume_and_final_validation_reject_stale_trace_identity(self) -> None:
        fixture_config = load_config(FIXTURE / "configs/release_config.json")
        with tempfile.TemporaryDirectory(prefix="release-identity-", dir=ROOT / "evaluation") as temporary:
            output = Path(temporary) / "release"
            config = fixture_config.model_copy(update={
                "run_id": "identity-test-v1",
                "output_dir": output.relative_to(ROOT).as_posix(),
            })
            config_path = Path(temporary) / "config.json"
            write_json(config_path, config)
            run_release(config_path, generator=generate_fixture_row)
            rows = read_jsonl(output / "raw/rows.jsonl", RawReleaseRow)
            target = next(row for row in rows if row.trace is not None)
            tampered_trace = json.loads(json.dumps(target.trace))
            tampered_trace["request"]["query"] = "stale query from another case"
            tampered = target.model_copy(update={"trace": tampered_trace})
            changed = [tampered if row.key == target.key else row for row in rows]
            write_jsonl(output / "raw/rows.jsonl", changed)
            manifest = load_manifest(output / "manifest.json")
            update_manifest(manifest, output, raw_rows=changed, lifecycle_status="RAW_COMPLETE")

            with self.assertRaisesRegex(ValueError, "immutable raw row identity mismatch"):
                run_release(config_path, generator=lambda *_args: (_ for _ in ()).throw(AssertionError()))

        with tempfile.TemporaryDirectory() as temporary:
            candidate = Path(temporary) / "release"
            shutil.copytree(FIXTURE, candidate)
            rows = read_jsonl(candidate / "raw/rows.jsonl", RawReleaseRow)
            target = next(row for row in rows if row.trace is not None)
            tampered_trace = json.loads(json.dumps(target.trace))
            tampered_trace["request"]["requested_model"] = "stale-model:1"
            changed = [
                target.model_copy(update={"trace": tampered_trace}) if row.key == target.key else row
                for row in rows
            ]
            write_jsonl(candidate / "raw/rows.jsonl", changed)
            manifest = load_manifest(candidate / "manifest.json")
            update_manifest(manifest, candidate, raw_rows=changed, lifecycle_status="VALIDATED")
            errors = validate_release_directory(candidate)
            self.assertTrue(any("requested model differs" in error for error in errors), errors)

    def test_paper_draft_passes_and_submission_is_gate_blocked(self) -> None:
        paper = ROOT / "paper/eacl_industry"
        draft_errors, notices = validate_paper(paper, submission=False)
        self.assertEqual(draft_errors, [])
        self.assertTrue(notices)
        submission_errors, _ = validate_paper(paper, submission=True)
        self.assertTrue(any("required release/study gates" in error for error in submission_errors))
        self.assertTrue(any("pending claims" in error for error in submission_errors))
        self.assertNotIn("pdf: compiled review PDF lacks the anonymous author marker", submission_errors)


class LegalPdfIngestionSmokeTests(unittest.TestCase):
    def test_project_authored_pdf_reaches_canonical_answer_pipeline(self) -> None:
        with tempfile.TemporaryDirectory(prefix="scholar-legal-pdf-") as temporary:
            pdf = Path(temporary) / "project-authored-fixture.pdf"
            document = fitz.open()
            page = document.new_page(width=600, height=800)
            page.insert_text((50, 80), "ScholAR retrieval architecture", fontsize=18)
            page.insert_text(
                (50, 130),
                "BM25 is the primary lexical scoring method used to prevent semantic drift.",
                fontsize=11,
            )
            page.insert_text(
                (50, 165),
                "Dense, modality, crop-image, and full-page visual channels are fused with RRF.",
                fontsize=11,
            )
            document.save(str(pdf))
            document.close()
            bundle = Path(temporary) / "legal_fixture"
            with patch.dict(os.environ, {
                "SCHOLAR_NETWORK_MODE": "strict-local",
                "DOCLING_ARTIFACTS_PATH": "",
                "HF_HUB_OFFLINE": "1",
                "HF_DATASETS_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
            }):
                with patch("backend.services.ingestion_service.is_docling_available", return_value=False):
                    PaperFinalizeService.finalize(
                        pdf,
                        "legal_fixture",
                        {"title": "Project-authored architecture guide", "authors": ["Anonymous"]},
                        target_dir=bundle,
                    )
                self.assertEqual(set(PaperFinalizeService.REQUIRED_FILES), {path.name for path in bundle.iterdir() if path.is_file()})
                self.assertIsNotNone(PaperFinalizeService.load_if_complete("legal_fixture", target_dir=bundle))

                def paths(_paper_id: str):
                    return bundle / "metadata.json", bundle / "pages.json", bundle / "chunks.json", bundle / "paper.pdf"

                with (
                    patch("backend.services.answer_pipeline._paper_paths", side_effect=paths),
                    patch("backend.services.answer_pipeline.paper_dir", return_value=bundle),
                    patch("backend.services.answer_pipeline.ollama_available", new=AsyncMock(return_value=False)),
                    patch(
                        "backend.services.retrieval_service.DenseEmbeddingService.search_dense",
                        return_value=[],
                    ),
                    patch(
                        "backend.services.retrieval_service.RerankerService.rerank",
                        side_effect=lambda _query, candidates, top_k: candidates[:top_k],
                    ),
                    patch.object(socket.socket, "connect", side_effect=AssertionError("network access attempted")),
                    patch.object(TelemetryService, "persist_trace", side_effect=lambda trace: trace),
                ):
                    trace = asyncio.run(AnswerPipelineService.answer(AnswerPipelineRequest(
                        paper_id="legal_fixture",
                        query="Which primary scoring method prevents semantic drift?",
                        requested_model="fixture-text:1",
                    )))
        self.assertEqual(trace.status, PipelineStatus.SUCCESS)
        self.assertFalse(trace.abstention.abstained)
        self.assertEqual(trace.generation.mode, GenerationMode.EXTRACTIVE_FALLBACK)
        self.assertEqual(trace.paper_id, "legal_fixture")
        self.assertTrue(trace.retrieval_hits)
        self.assertTrue(trace.prompt_evidence)
        self.assertIn("BM25", trace.final_answer)
        self.assertTrue(trace.citations)
        self.assertTrue(all(hit.identity.source_id == "legal_fixture" for hit in trace.retrieval_hits))
        self.assertTrue(all(item.identity.source_id == "legal_fixture" for item in trace.prompt_evidence))
        self.assertTrue(all(citation.identity and citation.identity.source_id == "legal_fixture" for citation in trace.citations))
        self.assertTrue(any(
            citation.origin in {CitationOrigin.APPLICATION_IMPUTED, CitationOrigin.REMAPPED}
            for citation in trace.citations
        ))
        self.assertTrue(any("BM25" in citation.quote for citation in trace.citations))


if __name__ == "__main__":
    unittest.main()
