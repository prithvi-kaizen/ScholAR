"""Regression tests for always-on, source-scoped full-page visual retrieval."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType
from backend.services.reranker_service import RerankerService
from backend.services.retrieval_service import retrieve_chunks
from backend.services.vision_service import _load_visual_png, _visual_image_url
from backend.services.visual_page_retrieval_service import (
    VisualPageRetrievalService,
    VisualPageSearchResult,
    VisualPageSearchStatus,
)


def _page_unit(page: int, image_sha256: str = "a" * 64) -> VisualDocumentUnit:
    return VisualDocumentUnit(
        visual_id=f"page_{page:04d}",
        document_id="paper",
        source_paper_id="paper",
        page=page,
        unit_type=VisualUnitType.PAGE,
        image_relpath=f"page_images/page_{page:04d}.png",
        image_sha256=image_sha256,
        width_px=100,
        height_px=200,
        bbox_norm=[0.0, 0.0, 1.0, 1.0],
        label=f"Page {page}",
    )


class TestVisualPageRetrieval(unittest.TestCase):
    def test_implicit_query_ranks_page_by_token_patch_maxsim(self) -> None:
        units = [_page_unit(1), _page_unit(2, "b" * 64)]
        vectors = np.asarray(
            [
                [[1.0, 0.0], [1.0, 0.0]],
                [[0.0, 1.0], [0.2, 0.98]],
            ],
            dtype=np.float16,
        )
        query_tokens = np.asarray([[0.0, 1.0]], dtype=np.float32)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "pages.json").write_text(
                json.dumps([
                    {"page": 1, "text": "unrelated prose"},
                    {"page": 2, "text": "the relevant plotted result"},
                ]),
                encoding="utf-8",
            )
            with (
                patch(
                    "backend.services.visual_page_retrieval_service.paper_dir",
                    return_value=root,
                ),
                patch.object(VisualPageRetrievalService, "_page_units", return_value=units),
                patch.object(VisualPageRetrievalService, "_build_or_load_source", return_value=vectors),
                patch.object(VisualPageRetrievalService, "_encode_query", return_value=query_tokens),
                patch(
                    "backend.services.visual_page_retrieval_service.VisualEmbeddingService.encoder_bundle",
                    return_value=(object(), object(), "cpu", "encoder-v1"),
                ),
                patch.dict("os.environ", {"SCHOLAR_VISUAL_PAGE_MIN_SIMILARITY": "0.1"}),
            ):
                result = VisualPageRetrievalService.search(
                    "Which system peaks after the crossover?",
                    ["paper"],
                    top_k=2,
                )

        self.assertEqual(len(result.hits), 1)
        chunk, score = result.hits[0]
        self.assertEqual(chunk["page"], 2)
        self.assertEqual(chunk["chunk_type"], "page_visual")
        self.assertTrue(chunk["is_page_visual_chunk"])
        self.assertGreater(score, 0.9)
        self.assertTrue(result.status.succeeded)
        self.assertEqual(result.status.indexed_pages, 2)

    def test_index_manifest_is_content_and_encoder_bound(self) -> None:
        unit = _page_unit(1)
        identity = VisualPageRetrievalService._manifest_identity(
            "paper", [unit], "encoder-v1"
        )
        vectors = np.asarray([[[1.0, 0.0], [0.0, 1.0]]], dtype=np.float16)
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            index_path = root / "index.npy"
            manifest_path = root / "manifest.json"
            VisualPageRetrievalService._publish_index(
                index_path, manifest_path, identity, vectors
            )
            loaded = VisualPageRetrievalService._load_index(
                index_path, manifest_path, identity
            )
            invalidated = VisualPageRetrievalService._load_index(
                index_path,
                manifest_path,
                {**identity, "encoder_fingerprint": "encoder-v2"},
            )

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded, vectors)
        self.assertIsNone(invalidated)

    def test_page_channel_runs_without_visual_query_words_and_preserves_trace(self) -> None:
        text_chunk = {
            "chunk_id": "text_1",
            "document_id": "paper",
            "source_paper_id": "paper",
            "page": 2,
            "text": "The system reaches its maximum after the crossover point.",
            "is_figure_chunk": False,
        }
        page_chunk = {
            "chunk_id": "visual_page_0002",
            "evidence_id": "page_0002",
            "document_id": "paper",
            "source_paper_id": "paper",
            "page": 2,
            "text": "The system reaches its maximum after the crossover point.",
            "is_figure_chunk": True,
            "is_page_visual_chunk": True,
            "figure_type": "page",
            "figure_id": "page_0002",
            "image_file": "page_0002.png",
            "image_relpath": "page_images/page_0002.png",
        }
        status = VisualPageSearchStatus(
            attempted=True,
            succeeded=True,
            model_loaded=True,
            indexed_pages=2,
            hit_count=1,
            best_score=0.8,
            minimum_score=0.12,
        )
        metadata: dict = {}

        with (
            patch.dict(os.environ, {"SCHOLAR_VISUAL_PAGE_BACKEND": "clip"}),
            patch(
                "backend.services.retrieval_service.DenseEmbeddingService.search_dense",
                return_value=[(text_chunk, 0.8)],
            ),
            patch(
                "backend.services.retrieval_service.VisualEmbeddingService.search_visual",
                return_value=[],
            ),
            patch(
                "backend.services.retrieval_service.VisualPageRetrievalService.search",
                return_value=VisualPageSearchResult([(page_chunk, 0.8)], status),
            ) as page_search,
            patch(
                "backend.services.retrieval_service.RerankerService.rerank",
                side_effect=lambda _query, candidates, top_k: candidates[:top_k],
            ),
        ):
            result = retrieve_chunks(
                "Which system reaches the maximum after the crossover?",
                [text_chunk],
                limit=2,
                paper_id="paper",
                retrieval_metadata=metadata,
            )

        page_search.assert_called_once()
        returned_page = next(item for item in result if item.get("is_page_visual_chunk"))
        self.assertTrue(returned_page["page_image_eligible"])
        self.assertTrue(returned_page["page_image_corroborated"])
        self.assertGreater(RerankerService._page_rank_boost(returned_page), 0.0)
        self.assertEqual(metadata["visual_page_retrieval"]["hit_count"], 1)

    def test_page_image_loader_and_url_are_source_scoped_and_safe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            image = root / "page_images" / "page_0003.png"
            image.parent.mkdir()
            image.write_bytes(b"page-pixels")
            with patch("backend.services.vision_service.paper_dir", return_value=root):
                loaded = _load_visual_png(
                    "paper", "page_0003.png", "page_images/page_0003.png"
                )
                rejected = _load_visual_png(
                    "paper", "secrets.png", "../secrets.png"
                )

        self.assertEqual(loaded, b"page-pixels")
        self.assertIsNone(rejected)
        self.assertEqual(
            _visual_image_url(
                {
                    "source_paper_id": "paper",
                    "page": 3,
                    "figure_type": "page",
                    "is_page_visual_chunk": True,
                },
                "fallback",
            ),
            "/api/papers/paper/page/3.png?zoom=1.6",
        )


if __name__ == "__main__":
    unittest.main()
