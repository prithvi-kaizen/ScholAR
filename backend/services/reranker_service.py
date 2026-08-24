"""Cross-Encoder Reranking Service for ScholAR.

Scores query-evidence candidate pairs using a local cross-encoder model:
- Takes top-N pooled candidates from Tri-Channel RRF
- Evaluates joint token attention across (query, candidate_evidence)
- Normalizes scores into calibrated relevance probabilities in [0.0, 1.0]
- Provides deterministic lexical-semantic fallback for offline/low-compute environments
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any

logger = logging.getLogger("scholar.reranker")

DEFAULT_RERANKER_MODEL = os.getenv("SCHOLAR_RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")


class RerankerService:
    """Manages cross-encoder scoring and candidate reranking."""

    _model: Any = None
    _tokenizer: Any = None
    _is_initialized: bool = False
    _fallback_mode: bool = False

    @classmethod
    def initialize(cls, model_name: str | None = None) -> None:
        """Initialize the local cross-encoder model."""
        if cls._is_initialized:
            return

        model_name = model_name or DEFAULT_RERANKER_MODEL
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            offline = os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
            cls._model = AutoModelForSequenceClassification.from_pretrained(model_name, local_files_only=offline)
            cls._model.eval()

            if torch.backends.mps.is_available():
                cls._model = cls._model.to("mps")
            elif torch.cuda.is_available():
                cls._model = cls._model.to("cuda")
            else:
                cls._model = cls._model.to("cpu")

            cls._fallback_mode = False
            cls._is_initialized = True
            logger.info("Reranker service initialized with [%s]", model_name)

        except Exception as exc:
            logger.info("Cross-encoder model unavailable (%s). Engaging deterministic heuristic reranker.", exc)
            cls._fallback_mode = True
            cls._is_initialized = True

    @classmethod
    def rerank(
        cls,
        query: str,
        candidates: list[dict[str, Any]],
        top_k: int = 10,
    ) -> list[dict[str, Any]]:
        """Rerank candidates using cross-encoder scores and return top-K."""
        if not candidates:
            return []

        if not cls._is_initialized:
            cls.initialize()

        if cls._fallback_mode or cls._model is None:
            return cls._rerank_fallback(query, candidates, top_k)

        try:
            import torch

            device = next(cls._model.parameters()).device
            pairs = [
                [query, f"{c.get('section', '')}: {c.get('text', '')}"]
                for c in candidates
            ]

            encoded = cls._tokenizer(
                pairs,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = cls._model(**encoded)
                logits = outputs.logits
                if logits.shape[1] == 1:
                    scores = torch.sigmoid(logits.squeeze(-1)).cpu().tolist()
                else:
                    scores = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

            # Attach scores and rank
            scored_candidates = []
            for candidate, score in zip(candidates, scores):
                c_copy = dict(candidate)
                c_copy["rerank_score"] = round(float(score), 4)
                scored_candidates.append(c_copy)

            scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
            return scored_candidates[:top_k]

        except Exception as exc:
            logger.warning("Cross-encoder inference failed (%s). Falling back to heuristic reranking.", exc)
            return cls._rerank_fallback(query, candidates, top_k)

    @classmethod
    def _rerank_fallback(
        cls, query: str, candidates: list[dict[str, Any]], top_k: int
    ) -> list[dict[str, Any]]:
        """Deterministic lexical & positional heuristic reranker."""
        query_words = set(re.findall(r"\w+", query.lower()))
        scored_candidates = []

        for candidate in candidates:
            text = (candidate.get("text", "") + " " + candidate.get("section", "")).lower()
            cand_words = set(re.findall(r"\w+", text))
            overlap = len(query_words & cand_words)
            jaccard = overlap / max(len(query_words | cand_words), 1)

            # Prioritize table and figure if query asks for them
            modality_boost = 0.0
            if "table" in query.lower() and candidate.get("is_table_chunk"):
                modality_boost += 0.2
            if ("figure" in query.lower() or "plot" in query.lower()) and candidate.get("is_figure_chunk"):
                modality_boost += 0.2

            base_rrf = candidate.get("rrf_score", 0.0)
            score = round(min(1.0, 0.5 * jaccard + 0.3 * base_rrf + modality_boost), 4)

            c_copy = dict(candidate)
            c_copy["rerank_score"] = score
            scored_candidates.append(c_copy)

        scored_candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return scored_candidates[:top_k]
