"""Regression tests for the document-trained visual retrieval backend."""

from __future__ import annotations

import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType
from backend.services.colqwen_visual_retrieval_service import (
    BACKEND_NAME,
    ColQwenSourceIndex,
    ColQwenVisualRetrievalService,
)
from backend.services.document_visual_retrieval_service import (
    DocumentVisualRetrievalService,
)
from backend.services.vision_service import _crop_visual_png, _retrieval_region_bbox
from backend.services.visual_page_retrieval_service import (
    VisualPageSearchResult,
    VisualPageSearchStatus,
)


def _unit(page: int, checksum: str) -> VisualDocumentUnit:
    return VisualDocumentUnit(
        visual_id=f"page_{page:04d}",
        document_id="paper",
        source_paper_id="paper",
        page=page,
        unit_type=VisualUnitType.PAGE,
        image_relpath=f"page_images/page_{page:04d}.png",
        image_sha256=checksum,
        width_px=1000,
        height_px=1400,
        bbox_norm=[0.0, 0.0, 1.0, 1.0],
    )


class TestColQwenVisualRetrieval(unittest.TestCase):
    def test_variable_length_cache_round_trip_and_identity_invalidation(self) -> None:
        units = [_unit(1, "a" * 64), _unit(2, "b" * 64)]
        identity = ColQwenVisualRetrievalService._manifest_identity(
            "paper", units, "encoder-v1"
        )
        index = ColQwenSourceIndex(
            vectors=np.arange(20, dtype=np.float16).reshape(5, 4),
            offsets=np.asarray([0, 2, 5], dtype=np.int64),
            page_metadata=[
                {"token_count": 2, "image_token_count": 2, "image_token_offset": 0},
                {"token_count": 3, "image_token_count": 3, "image_token_offset": 0},
            ],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            ColQwenVisualRetrievalService._publish_index(directory, identity, index)
            loaded = ColQwenVisualRetrievalService._load_index(directory, identity)
            invalidated = ColQwenVisualRetrievalService._load_index(
                directory,
                {**identity, "encoder_fingerprint": "encoder-v2"},
            )

        self.assertIsNotNone(loaded)
        np.testing.assert_array_equal(loaded.vectors, index.vectors)
        np.testing.assert_array_equal(loaded.offsets, index.offsets)
        self.assertEqual(loaded.page_vectors(1).shape, (3, 4))
        self.assertIsNone(invalidated)

    def test_patch_matches_produce_bounded_candidate_regions(self) -> None:
        page_vectors = np.zeros((16, 3), dtype=np.float16)
        page_vectors[5] = [1.0, 0.0, 0.0]
        page_vectors[6] = [0.0, 1.0, 0.0]
        query_vectors = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
        )
        regions = ColQwenVisualRetrievalService._candidate_regions(
            query_vectors,
            page_vectors,
            {
                "image_token_count": 16,
                "image_token_offset": 0,
                "image_token_positions": None,
                "grid_height": 4,
                "grid_width": 4,
            },
        )

        self.assertEqual(len(regions), 1)
        x0, y0, x1, y1 = regions[0]["bbox_norm"]
        self.assertTrue(0.0 <= x0 < x1 <= 1.0)
        self.assertTrue(0.0 <= y0 < y1 <= 1.0)
        self.assertEqual(regions[0]["method"], "query-token-maxsim-patch-cluster-v1")

    def test_backend_selection_is_configuration_driven_not_query_driven(self) -> None:
        colqwen_result = VisualPageSearchResult(
            [({"chunk_id": "page"}, 7.0)],
            VisualPageSearchStatus(
                attempted=True,
                succeeded=True,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=True,
            ),
        )
        with (
            patch.dict(os.environ, {"SCHOLAR_VISUAL_PAGE_BACKEND": "colqwen2"}),
            patch(
                "backend.services.document_visual_retrieval_service.ColQwenVisualRetrievalService.search",
                return_value=colqwen_result,
            ) as search,
        ):
            result = DocumentVisualRetrievalService.search(
                "Which method peaks after the crossover?", ["paper"], top_k=3
            )

        search.assert_called_once()
        self.assertEqual(result.hits[0][0]["chunk_id"], "page")
        self.assertEqual(result.status.requested_backend, "colqwen2")

    def test_search_keeps_same_page_numbers_source_scoped(self) -> None:
        unit_a = _unit(1, "a" * 64).model_copy(update={
            "document_id": "paper_a", "source_paper_id": "paper_a"
        })
        unit_b = _unit(1, "b" * 64).model_copy(update={
            "document_id": "paper_b", "source_paper_id": "paper_b"
        })
        indexes = {
            "paper_a": ColQwenSourceIndex(
                np.asarray([[1.0, 0.0]], dtype=np.float16),
                np.asarray([0, 1], dtype=np.int64),
                [{
                    "token_count": 1,
                    "image_token_count": 1,
                    "image_token_offset": 0,
                    "image_token_positions": None,
                    "grid_height": 1,
                    "grid_width": 1,
                }],
            ),
            "paper_b": ColQwenSourceIndex(
                np.asarray([[0.8, 0.2]], dtype=np.float16),
                np.asarray([0, 1], dtype=np.int64),
                [{
                    "token_count": 1,
                    "image_token_count": 1,
                    "image_token_offset": 0,
                    "image_token_positions": None,
                    "grid_height": 1,
                    "grid_width": 1,
                }],
            ),
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for source in indexes:
                directory = root / source
                directory.mkdir()
                (directory / "pages.json").write_text("[]", encoding="utf-8")
            with (
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_bundle",
                    return_value=(object(), object(), "cpu", "encoder-v1"),
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_encode_query",
                    return_value=np.asarray([[1.0, 0.0]], dtype=np.float32),
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_page_units",
                    side_effect=lambda source: [unit_a if source == "paper_a" else unit_b],
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_build_or_load_source",
                    side_effect=lambda source, *_args: indexes[source],
                ),
                patch(
                    "backend.services.colqwen_visual_retrieval_service.paper_dir",
                    side_effect=lambda source: root / source,
                ),
            ):
                result = ColQwenVisualRetrievalService.search(
                    "implicit visual question", ["paper_b", "paper_a"], top_k=2
                )

        self.assertEqual(
            [hit[0]["source_paper_id"] for hit in result.hits],
            ["paper_a", "paper_b"],
        )
        self.assertEqual([hit[0]["page"] for hit in result.hits], [1, 1])
        self.assertNotEqual(
            result.hits[0][0]["source_paper_id"],
            result.hits[1][0]["source_paper_id"],
        )

    def test_prebuild_source_does_not_encode_a_query(self) -> None:
        unit = _unit(1, "a" * 64)
        index = ColQwenSourceIndex(
            np.asarray([[1.0, 0.0]], dtype=np.float16),
            np.asarray([0, 1], dtype=np.int64),
            [{"token_count": 1}],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_bundle",
                    return_value=(object(), object(), "cpu", "encoder-v1"),
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_page_units",
                    return_value=[unit],
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_build_or_load_source",
                    return_value=index,
                ),
                patch.object(
                    ColQwenVisualRetrievalService,
                    "_encode_query",
                    side_effect=AssertionError("prebuild must not encode a query"),
                ),
                patch(
                    "backend.services.colqwen_visual_retrieval_service.paper_dir",
                    return_value=Path(tmpdir),
                ),
            ):
                status = ColQwenVisualRetrievalService.prebuild_source("paper")

        self.assertTrue(status.succeeded)
        self.assertEqual(status.indexed_pages, 1)
        self.assertEqual(status.sources_considered, ["paper"])

    def test_auto_backend_records_colqwen_failure_and_clip_fallback(self) -> None:
        unavailable = VisualPageSearchResult(
            [],
            VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=False,
                failure_reason="snapshot missing",
            ),
        )
        clip = VisualPageSearchResult(
            [({"chunk_id": "clip_page"}, 0.5)],
            VisualPageSearchStatus(
                attempted=True,
                succeeded=True,
                model_loaded=True,
                hit_count=1,
            ),
        )
        with (
            patch.dict(os.environ, {"SCHOLAR_VISUAL_PAGE_BACKEND": "auto"}),
            patch(
                "backend.services.document_visual_retrieval_service.ColQwenVisualRetrievalService.search",
                return_value=unavailable,
            ),
            patch(
                "backend.services.document_visual_retrieval_service.VisualPageRetrievalService.search",
                return_value=clip,
            ),
        ):
            result = DocumentVisualRetrievalService.search("implicit question", ["paper"])

        self.assertEqual(result.hits[0][0]["chunk_id"], "clip_page")
        self.assertEqual(result.status.requested_backend, "auto")
        self.assertIn("snapshot missing", result.status.failure_reason)

    def test_retrieval_crop_is_validated_and_created_in_memory(self) -> None:
        figure = {
            "candidate_regions": [
                {"bbox_norm": [0.25, 0.25, 0.75, 0.75], "score": 0.9}
            ]
        }
        bbox = _retrieval_region_bbox(figure)
        image = Image.new("RGB", (100, 80), color="white")
        payload = io.BytesIO()
        image.save(payload, format="PNG")
        cropped = _crop_visual_png(payload.getvalue(), bbox)
        with Image.open(io.BytesIO(cropped)) as opened:
            size = opened.size

        self.assertEqual(bbox, [0.25, 0.25, 0.75, 0.75])
        self.assertEqual(size, (50, 40))


if __name__ == "__main__":
    unittest.main()
