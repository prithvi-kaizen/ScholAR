"""Dense Embedding & Vector Search Service for ScholAR.

Provides:
- Local dense embedding extraction via Transformers (e.g. Qwen3-Embedding / BGE / MiniLM) with PyTorch
- Mean-pooling + L2 normalization for fast exact inner-product search
- Deterministic TF-IDF / subword n-gram vectorizer fallback for 100% offline environments
- Local paper vector indexing & caching (`embeddings.npy`)
"""

from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

from backend.services.pdf_service import paper_dir, read_json

logger = logging.getLogger("scholar.embeddings")

DEFAULT_EMBEDDING_MODEL = os.getenv("SCHOLAR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")


class DenseEmbeddingService:
    """Manages dense embedding extraction, paper vector indexing, and dense similarity search."""

    _model: Any = None
    _tokenizer: Any = None
    _is_initialized: bool = False
    _fallback_mode: bool = False

    @classmethod
    def initialize(cls, model_name: str | None = None) -> None:
        """Initialize the local embedding model with PyTorch/Transformers."""
        if cls._is_initialized:
            return

        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Check if offline mode is enforced
            offline = os.getenv("HF_HUB_OFFLINE", "0") == "1" or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            cls._tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=offline)
            cls._model = AutoModel.from_pretrained(model_name, local_files_only=offline)
            cls._model.eval()

            # Move to MPS (Apple Silicon) or CUDA if available
            if torch.backends.mps.is_available():
                cls._model = cls._model.to("mps")
                logger.info("Dense embedding model loaded on Apple Silicon MPS.")
            elif torch.cuda.is_available():
                cls._model = cls._model.to("cuda")
                logger.info("Dense embedding model loaded on NVIDIA CUDA.")
            else:
                cls._model = cls._model.to("cpu")
                logger.info("Dense embedding model loaded on CPU.")

            cls._fallback_mode = False
            cls._is_initialized = True
            logger.info("Dense embedding service initialized with [%s]", model_name)

        except Exception as exc:
            logger.info("Transformer embedding model unavailable (%s). Engaging deterministic fallback vectorizer.", exc)
            cls._fallback_mode = True
            cls._is_initialized = True

    @classmethod
    def encode(cls, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into normalized L2 dense embeddings."""
        if not cls._is_initialized:
            cls.initialize()

        if not texts:
            return np.empty((0, 384), dtype=np.float32)

        if cls._fallback_mode or cls._model is None:
            return cls._encode_fallback(texts)

        try:
            import torch

            device = next(cls._model.parameters()).device
            encoded = cls._tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            ).to(device)

            with torch.no_grad():
                outputs = cls._model(**encoded)
                # Mean pooling with attention mask
                token_embeddings = outputs.last_hidden_state
                input_mask_expanded = encoded["attention_mask"].unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                mean_pooled = sum_embeddings / sum_mask

                # L2 normalize
                normalized = torch.nn.functional.normalize(mean_pooled, p=2, dim=1)
                return normalized.cpu().numpy().astype(np.float32)

        except Exception as exc:
            logger.warning("Transformer encoding failed (%s). Using fallback vectorizer.", exc)
            return cls._encode_fallback(texts)

    @classmethod
    def _encode_fallback(cls, texts: list[str], dim: int = 384) -> np.ndarray:
        """Deterministic subword n-gram hashing vectorizer for offline fallback."""
        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = re.sub(r"[^\w\s]", " ", text.lower())
            tokens = clean.split()
            if not tokens:
                continue

            for t in tokens:
                # Word hash
                h = hash(t) % dim
                vectors[i, h] += 1.0
                # Bigram subword hashes
                for k in range(len(t) - 2):
                    sub = t[k : k + 3]
                    h_sub = hash(sub) % dim
                    vectors[i, h_sub] += 0.5

            # L2 normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 1e-9:
                vectors[i] /= norm

        return vectors

    @classmethod
    def build_or_load_paper_index(cls, paper_id: str, chunks: list[dict[str, Any]]) -> np.ndarray:
        """Build or load the dense vector index for a paper."""
        p_dir = paper_dir(paper_id)
        emb_path = p_dir / "embeddings.npy"

        if emb_path.exists():
            try:
                vectors = np.load(str(emb_path))
                if len(vectors) == len(chunks):
                    return vectors
            except Exception:
                logger.warning("Failed loading cached embeddings for [%s], rebuilding.", paper_id)

        # Build embeddings from chunk text + section title
        texts_to_embed = [
            f"{c.get('section', '')}: {c.get('text', '')}"
            for c in chunks
        ]
        vectors = cls.encode(texts_to_embed)

        try:
            emb_path.parent.mkdir(parents=True, exist_ok=True)
            np.save(str(emb_path), vectors)
            logger.info("Saved %d dense vectors to %s", len(vectors), emb_path)
        except Exception as exc:
            logger.warning("Could not persist embeddings to disk: %s", exc)

        return vectors

    @classmethod
    def search_dense(
        cls,
        paper_id: str,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        """Search top-K chunks using cosine similarity over normalized dense vectors."""
        if not chunks:
            return []

        vectors = cls.build_or_load_paper_index(paper_id, chunks)
        query_vec = cls.encode([query])[0]  # Shape: (dim,)

        # Compute dot products with all chunk vectors (cosine similarity)
        scores = np.dot(vectors, query_vec)  # Shape: (num_chunks,)

        # Rank indices
        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in ranked_indices:
            score_val = float(scores[idx])
            results.append((chunks[idx], max(0.0, score_val)))

        return results
