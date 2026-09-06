"""Backend selection for always-on full-page visual retrieval."""

from __future__ import annotations

import os
from dataclasses import replace

from backend.services.colqwen_visual_retrieval_service import (
    ColQwenVisualRetrievalService,
)
from backend.services.visual_page_retrieval_service import (
    VisualPageRetrievalService,
    VisualPageSearchResult,
    VisualPageSearchStatus,
)


VALID_BACKENDS = {"auto", "colqwen2", "clip", "disabled"}


class DocumentVisualRetrievalService:
    """Resolve one configured document-visual retriever without query routing."""

    @staticmethod
    def configured_backend(requested_backend: str | None = None) -> str:
        backend = (
            requested_backend
            or os.getenv("SCHOLAR_VISUAL_PAGE_BACKEND", "auto")
        ).strip().lower()
        if backend not in VALID_BACKENDS:
            allowed = ", ".join(sorted(VALID_BACKENDS))
            raise RuntimeError(
                f"Invalid SCHOLAR_VISUAL_PAGE_BACKEND={backend!r}; expected one of: {allowed}"
            )
        return backend

    @classmethod
    def search(
        cls,
        query: str,
        source_ids: list[str],
        top_k: int = 12,
        backend: str | None = None,
    ) -> VisualPageSearchResult:
        backend = cls.configured_backend(backend)
        if backend == "disabled":
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=False,
                succeeded=None,
                backend="disabled",
                requested_backend="disabled",
                sources_considered=sorted({item for item in source_ids if item}),
                failure_reason="full-page visual retrieval disabled by configuration",
            ))
        if backend == "clip":
            result = VisualPageRetrievalService.search(query, source_ids, top_k=top_k)
            return VisualPageSearchResult(
                result.hits,
                replace(result.status, requested_backend="clip"),
            )
        if backend == "colqwen2":
            return ColQwenVisualRetrievalService.search(query, source_ids, top_k=top_k)

        colqwen = ColQwenVisualRetrievalService.search(query, source_ids, top_k=top_k)
        if colqwen.status.model_loaded:
            return VisualPageSearchResult(
                colqwen.hits,
                replace(colqwen.status, requested_backend="auto"),
            )

        clip = VisualPageRetrievalService.search(query, source_ids, top_k=top_k)
        colqwen_reason = colqwen.status.failure_reason or "ColQwen2 unavailable"
        clip_reason = clip.status.failure_reason
        combined_reason = (
            f"auto fallback: {colqwen_reason}; CLIP: {clip_reason}"
            if clip_reason
            else f"auto fallback from ColQwen2: {colqwen_reason}"
        )
        return VisualPageSearchResult(
            clip.hits,
            replace(
                clip.status,
                requested_backend="auto",
                failure_reason=combined_reason,
            ),
        )

    @classmethod
    def release(cls) -> None:
        ColQwenVisualRetrievalService.release()
        from backend.services.visual_embedding_service import VisualEmbeddingService

        VisualEmbeddingService.release()
