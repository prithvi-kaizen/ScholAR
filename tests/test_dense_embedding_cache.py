"""Tests for deterministic fallback embeddings and validated dense cache reuse."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from backend.services.dense_embedding_service import (
    EMBEDDINGS_FILENAME,
    EMBEDDINGS_MANIFEST_FILENAME,
    FALLBACK_DIMENSION,
    DenseEmbeddingService,
)


class _FailingModel:
    def parameters(self):
        raise RuntimeError("simulated transformer inference failure")


class TestDenseEmbeddingCache(unittest.TestCase):
    _STATE_FIELDS = (
        "_model",
        "_tokenizer",
        "_is_initialized",
        "_fallback_mode",
        "_requested_model",
        "_encoder_mode",
    )

    def setUp(self) -> None:
        self._saved_state = {
            field: getattr(DenseEmbeddingService, field)
            for field in self._STATE_FIELDS
        }
        self.temp_dir = tempfile.TemporaryDirectory()
        self.paper_path = Path(self.temp_dir.name) / "paper"
        self.paper_dir_patch = mock.patch(
            "backend.services.dense_embedding_service.paper_dir",
            return_value=self.paper_path,
        )
        self.paper_dir_patch.start()
        self._set_encoder_state("test-model-a", "fallback")

    def tearDown(self) -> None:
        self.paper_dir_patch.stop()
        self.temp_dir.cleanup()
        for field, value in self._saved_state.items():
            setattr(DenseEmbeddingService, field, value)

    @staticmethod
    def _chunks(first_text: str = "alpha evidence", second_text: str = "beta evidence") -> list[dict[str, str]]:
        return [
            {"section": "Introduction", "text": first_text},
            {"section": "Results", "text": second_text},
        ]

    @staticmethod
    def _fake_encode(texts: list[str]) -> np.ndarray:
        return DenseEmbeddingService._encode_fallback(texts)

    @staticmethod
    def _read_manifest(paper_path: Path) -> dict[str, object]:
        with (paper_path / EMBEDDINGS_MANIFEST_FILENAME).open("r", encoding="utf-8") as handle:
            return json.load(handle)

    @staticmethod
    def _sha256(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    @staticmethod
    def _set_encoder_state(requested_model: str, mode: str) -> None:
        DenseEmbeddingService._is_initialized = True
        DenseEmbeddingService._requested_model = requested_model
        DenseEmbeddingService._encoder_mode = mode
        DenseEmbeddingService._fallback_mode = mode == "fallback"
        DenseEmbeddingService._model = None if mode == "fallback" else object()
        DenseEmbeddingService._tokenizer = None

    def test_same_count_text_change_invalidates_cache(self) -> None:
        with mock.patch.object(DenseEmbeddingService, "encode", side_effect=self._fake_encode) as encode:
            original = DenseEmbeddingService.build_or_load_paper_index("paper", self._chunks())
            cached = DenseEmbeddingService.build_or_load_paper_index("paper", self._chunks())
            changed = DenseEmbeddingService.build_or_load_paper_index(
                "paper",
                self._chunks(first_text="replaced alpha evidence"),
            )

        self.assertEqual(encode.call_count, 2)
        np.testing.assert_array_equal(cached, original)
        self.assertFalse(np.array_equal(changed, original))

    def test_requested_model_and_encoder_mode_invalidate_cache(self) -> None:
        chunks = self._chunks()
        with mock.patch.object(DenseEmbeddingService, "encode", side_effect=self._fake_encode) as encode:
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 1)

            self._set_encoder_state("test-model-b", "fallback")
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 2)
            self.assertEqual(self._read_manifest(self.paper_path)["requested_model"], "test-model-b")

            self._set_encoder_state("test-model-b", "transformer")
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 3)

        manifest = self._read_manifest(self.paper_path)
        self.assertEqual(manifest["encoder_mode"], "transformer")
        self.assertIsInstance(manifest["encoder_fingerprint"], str)

    def test_legacy_and_corrupt_cache_artifacts_are_rebuilt(self) -> None:
        chunks = self._chunks()
        self.paper_path.mkdir(parents=True)
        legacy_vectors = np.ones((len(chunks), FALLBACK_DIMENSION), dtype=np.float32)
        np.save(self.paper_path / EMBEDDINGS_FILENAME, legacy_vectors)

        with mock.patch.object(DenseEmbeddingService, "encode", side_effect=self._fake_encode) as encode:
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 1)
            self.assertTrue((self.paper_path / EMBEDDINGS_MANIFEST_FILENAME).exists())

            (self.paper_path / EMBEDDINGS_FILENAME).write_bytes(b"corrupt-npy")
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 2)

            (self.paper_path / EMBEDDINGS_MANIFEST_FILENAME).write_text("{not-json", encoding="utf-8")
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 3)

            non_finite = self._fake_encode(
                [f"{chunk['section']}: {chunk['text']}" for chunk in chunks]
            )
            non_finite[0, 0] = np.nan
            np.save(self.paper_path / EMBEDDINGS_FILENAME, non_finite)
            manifest = self._read_manifest(self.paper_path)
            manifest["vector_sha256"] = self._sha256(self.paper_path / EMBEDDINGS_FILENAME)
            (self.paper_path / EMBEDDINGS_MANIFEST_FILENAME).write_text(
                json.dumps(manifest),
                encoding="utf-8",
            )
            rebuilt = DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
            self.assertEqual(encode.call_count, 4)

        self.assertTrue(np.isfinite(rebuilt).all())

    def test_manifest_binds_shape_dtype_checksum_and_atomic_publication(self) -> None:
        chunks = self._chunks()
        with mock.patch.object(DenseEmbeddingService, "encode", side_effect=self._fake_encode):
            vectors = DenseEmbeddingService.build_or_load_paper_index("paper", chunks)

        vector_path = self.paper_path / EMBEDDINGS_FILENAME
        manifest = self._read_manifest(self.paper_path)
        self.assertEqual(manifest["vector_shape"], list(vectors.shape))
        self.assertEqual(manifest["vector_dimension"], FALLBACK_DIMENSION)
        self.assertEqual(manifest["vector_dtype"], "float32")
        self.assertEqual(manifest["vector_sha256"], self._sha256(vector_path))
        self.assertEqual(manifest["input_count"], len(chunks))
        self.assertEqual(list(self.paper_path.glob("*.tmp")), [])

    def test_fallback_hashing_matches_versioned_signed_sha256_reference(self) -> None:
        dimension = 32
        vector = DenseEmbeddingService._encode_fallback(["abc"], dim=dimension)
        expected = np.zeros(dimension, dtype=np.float32)
        for feature_kind, feature, weight in (
            ("word", "abc", np.float32(1.0)),
            ("trigram", "abc", np.float32(0.5)),
        ):
            digest = hashlib.sha256(
                f"sha256-signed-feature-hashing-v1\0{feature_kind}\0{feature}".encode("utf-8")
            ).digest()
            bucket = int.from_bytes(digest[:8], "big") % dimension
            sign = np.float32(1.0 if digest[8] & 1 == 0 else -1.0)
            expected[bucket] += weight * sign
        expected /= np.float32(np.linalg.norm(expected))

        self.assertEqual(vector.dtype, np.float32)
        self.assertAlmostEqual(float(np.linalg.norm(vector[0])), 1.0, places=6)
        np.testing.assert_array_equal(vector[0], expected)

    def test_fallback_is_stable_across_python_hash_seeds(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        script = (
            "from backend.services.dense_embedding_service import DenseEmbeddingService; "
            "print(DenseEmbeddingService._encode_fallback(['Stable alpha-123 embedding']).tobytes().hex())"
        )
        outputs = []
        for seed in ("1", "999"):
            environment = os.environ.copy()
            environment["PYTHONHASHSEED"] = seed
            completed = subprocess.run(
                [sys.executable, "-c", script],
                cwd=repository_root,
                env=environment,
                check=True,
                capture_output=True,
                text=True,
            )
            outputs.append(completed.stdout.strip())

        self.assertEqual(outputs[0], outputs[1])
        self.assertTrue(outputs[0])

    def test_transformer_encode_failure_switches_to_sticky_fallback(self) -> None:
        self._set_encoder_state("test-model-a", "transformer")
        DenseEmbeddingService._model = _FailingModel()
        DenseEmbeddingService._tokenizer = object()

        first = DenseEmbeddingService.encode(["alpha"])
        self.assertTrue(DenseEmbeddingService._fallback_mode)
        self.assertEqual(DenseEmbeddingService._encoder_mode, "fallback")
        self.assertIsNone(DenseEmbeddingService._model)
        second = DenseEmbeddingService.encode(["alpha"])

        self.assertEqual(first.dtype, np.float32)
        np.testing.assert_array_equal(first, second)

    def test_query_failure_rebuilds_transformer_cache_in_fallback_space(self) -> None:
        chunks = self._chunks()
        self._set_encoder_state("test-model-a", "transformer")
        with mock.patch.object(DenseEmbeddingService, "encode", side_effect=self._fake_encode):
            DenseEmbeddingService.build_or_load_paper_index("paper", chunks)
        self.assertEqual(self._read_manifest(self.paper_path)["encoder_mode"], "transformer")

        DenseEmbeddingService._model = _FailingModel()
        DenseEmbeddingService._tokenizer = object()
        results = DenseEmbeddingService.search_dense("paper", "alpha query", chunks, top_k=1)

        self.assertEqual(len(results), 1)
        self.assertEqual(DenseEmbeddingService._encoder_mode, "fallback")
        self.assertEqual(self._read_manifest(self.paper_path)["encoder_mode"], "fallback")


if __name__ == "__main__":
    unittest.main()
