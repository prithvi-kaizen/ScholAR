"""Dense Embedding & Vector Search Service for ScholAR.

Provides:
- Local dense embedding extraction via Transformers (e.g. Qwen3-Embedding / BGE / MiniLM) with PyTorch
- Mean-pooling + L2 normalization for fast exact inner-product search
- Deterministic signed feature-hashing fallback for 100% offline environments
- Content- and encoder-aware local paper vector indexing with atomic cache publication
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from backend.services.network_policy_service import NetworkPolicyService
from backend.services.pdf_service import paper_dir, read_json

logger = logging.getLogger("scholar.embeddings")

DEFAULT_EMBEDDING_MODEL = os.getenv("SCHOLAR_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

EMBEDDINGS_FILENAME = "embeddings.npy"
EMBEDDINGS_MANIFEST_FILENAME = "embeddings_manifest.json"
EMBEDDINGS_CACHE_SCHEMA_VERSION = 1
EMBEDDING_ALGORITHM_VERSION = "dense-embedding-cache-v2"
TRANSFORMER_ENCODER_VERSION = "transformers-mean-pooling-l2-v1"
FALLBACK_ENCODER_VERSION = "sha256-signed-feature-hashing-v1"
FALLBACK_DIMENSION = 384


class DenseEmbeddingService:
    """Manages dense embedding extraction, paper vector indexing, and dense similarity search."""

    _model: Any = None
    _tokenizer: Any = None
    _is_initialized: bool = False
    _fallback_mode: bool = False
    _requested_model: str | None = None
    _encoder_mode: str = "uninitialized"

    @classmethod
    def initialize(cls, model_name: str | None = None) -> None:
        """Initialize the local embedding model with PyTorch/Transformers."""
        if cls._is_initialized:
            return

        model_name = model_name or DEFAULT_EMBEDDING_MODEL
        cls._requested_model = model_name
        try:
            import torch
            from transformers import AutoModel, AutoTokenizer

            # Strict-local analysis always uses pre-provisioned model files.
            offline = (
                NetworkPolicyService.enforce_local_model_cache()
                or os.getenv("HF_HUB_OFFLINE", "0") == "1"
                or os.getenv("TRANSFORMERS_OFFLINE", "0") == "1"
            )
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
            cls._encoder_mode = "transformer"
            cls._is_initialized = True
            logger.info("Dense embedding service initialized with [%s]", model_name)

        except Exception as exc:
            logger.info("Transformer embedding model unavailable (%s). Engaging deterministic fallback vectorizer.", exc)
            cls._activate_fallback()
            cls._is_initialized = True

    @classmethod
    def _activate_fallback(cls) -> None:
        """Permanently switch this process to the fallback vector space."""
        cls._fallback_mode = True
        cls._encoder_mode = "fallback"
        cls._model = None
        cls._tokenizer = None

    @classmethod
    def encode(cls, texts: list[str]) -> np.ndarray:
        """Encode a list of text strings into normalized L2 dense embeddings."""
        if not cls._is_initialized:
            cls.initialize()

        if not texts:
            return np.empty((0, FALLBACK_DIMENSION), dtype=np.float32)

        if cls._fallback_mode or cls._model is None:
            cls._activate_fallback()
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
            logger.warning("Transformer encoding failed (%s). Switching consistently to fallback vectorizer.", exc)
            cls._activate_fallback()
            return cls._encode_fallback(texts)

    @staticmethod
    def _fallback_feature_hash(feature_kind: str, feature: str, dim: int) -> tuple[int, np.float32]:
        """Map a namespaced feature to a stable bucket and sign using versioned SHA-256."""
        payload = f"{FALLBACK_ENCODER_VERSION}\0{feature_kind}\0{feature}".encode("utf-8")
        digest = hashlib.sha256(payload).digest()
        bucket = int.from_bytes(digest[:8], byteorder="big", signed=False) % dim
        sign = np.float32(1.0 if digest[8] & 1 == 0 else -1.0)
        return bucket, sign

    @classmethod
    def _encode_fallback(cls, texts: list[str], dim: int = FALLBACK_DIMENSION) -> np.ndarray:
        """Encode text with deterministic, versioned SHA-256 signed feature hashing."""
        if dim <= 0:
            raise ValueError("Fallback embedding dimension must be positive")

        vectors = np.zeros((len(texts), dim), dtype=np.float32)
        for i, text in enumerate(texts):
            clean = re.sub(r"[^\w\s]", " ", text.lower())
            tokens = clean.split()
            if not tokens:
                continue

            for token in tokens:
                word_bucket, word_sign = cls._fallback_feature_hash("word", token, dim)
                vectors[i, word_bucket] += word_sign

                for offset in range(len(token) - 2):
                    trigram = token[offset : offset + 3]
                    subword_bucket, subword_sign = cls._fallback_feature_hash("trigram", trigram, dim)
                    vectors[i, subword_bucket] += np.float32(0.5) * subword_sign

            norm = np.linalg.norm(vectors[i])
            if norm > 1e-9:
                vectors[i] /= np.float32(norm)

        return vectors.astype(np.float32, copy=False)

    @classmethod
    def _encoder_metadata(cls) -> dict[str, Any]:
        """Describe the active vector space and return its stable fingerprint."""
        requested_model = cls._requested_model or DEFAULT_EMBEDDING_MODEL
        mode = "fallback" if cls._fallback_mode else cls._encoder_mode
        if mode not in {"transformer", "fallback"}:
            mode = "fallback" if cls._model is None else "transformer"

        encoder_version = FALLBACK_ENCODER_VERSION if mode == "fallback" else TRANSFORMER_ENCODER_VERSION
        descriptor: dict[str, Any] = {
            "requested_model": requested_model,
            "encoder_mode": mode,
            "encoder_version": encoder_version,
        }
        if mode == "fallback":
            descriptor["dimension"] = FALLBACK_DIMENSION

        encoded_descriptor = json.dumps(
            descriptor,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return {
            **descriptor,
            "encoder_fingerprint": hashlib.sha256(encoded_descriptor).hexdigest(),
        }

    @classmethod
    def _encoder_fingerprint(cls) -> str:
        return str(cls._encoder_metadata()["encoder_fingerprint"])

    @staticmethod
    def _inputs_checksum(texts: list[str]) -> str:
        payload = json.dumps(texts, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _vectors_are_valid(
        vectors: Any,
        expected_rows: int,
        expected_shape: list[int] | None = None,
        expected_dimension: int | None = None,
    ) -> bool:
        if not isinstance(vectors, np.ndarray):
            return False
        if vectors.ndim != 2 or vectors.shape[0] != expected_rows or vectors.shape[1] <= 0:
            return False
        if vectors.dtype != np.dtype(np.float32):
            return False
        if expected_shape is not None and list(vectors.shape) != expected_shape:
            return False
        if expected_dimension is not None and vectors.shape[1] != expected_dimension:
            return False
        return bool(np.isfinite(vectors).all())

    @classmethod
    def _load_cached_vectors(
        cls,
        emb_path: Path,
        manifest_path: Path,
        texts_to_embed: list[str],
    ) -> np.ndarray | None:
        """Load vectors only when the manifest and artifact fully match this request."""
        if not emb_path.exists() or not manifest_path.exists():
            return None

        try:
            manifest = read_json(manifest_path)
            encoder_metadata = cls._encoder_metadata()
            required_metadata = {
                "schema_version": EMBEDDINGS_CACHE_SCHEMA_VERSION,
                "algorithm_version": EMBEDDING_ALGORITHM_VERSION,
                "inputs_sha256": cls._inputs_checksum(texts_to_embed),
                "input_count": len(texts_to_embed),
                "requested_model": encoder_metadata["requested_model"],
                "encoder_mode": encoder_metadata["encoder_mode"],
                "encoder_version": encoder_metadata["encoder_version"],
                "encoder_fingerprint": encoder_metadata["encoder_fingerprint"],
                "vector_dtype": "float32",
            }
            if not isinstance(manifest, dict) or any(manifest.get(key) != value for key, value in required_metadata.items()):
                return None

            shape = manifest.get("vector_shape")
            dimension = manifest.get("vector_dimension")
            checksum = manifest.get("vector_sha256")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(value) is not int for value in shape)
                or shape[0] != len(texts_to_embed)
                or shape[1] <= 0
                or type(dimension) is not int
                or dimension != shape[1]
                or not isinstance(checksum, str)
                or len(checksum) != 64
            ):
                return None
            if encoder_metadata["encoder_mode"] == "fallback" and dimension != FALLBACK_DIMENSION:
                return None
            if cls._file_checksum(emb_path) != checksum:
                return None

            vectors = np.load(str(emb_path), allow_pickle=False)
            if not cls._vectors_are_valid(vectors, len(texts_to_embed), shape, dimension):
                return None
            return vectors
        except Exception as exc:
            logger.warning("Failed loading validated embedding cache [%s], rebuilding: %s", emb_path, exc)
            return None

    @classmethod
    def _publish_cache_atomically(
        cls,
        emb_path: Path,
        manifest_path: Path,
        vectors: np.ndarray,
        texts_to_embed: list[str],
    ) -> None:
        """Publish vectors first and the matching manifest last as the cache commit marker."""
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        vector_temp: Path | None = None
        manifest_temp: Path | None = None

        try:
            vector_fd, vector_temp_name = tempfile.mkstemp(
                prefix=".embeddings-",
                suffix=".npy.tmp",
                dir=str(emb_path.parent),
            )
            vector_temp = Path(vector_temp_name)
            with os.fdopen(vector_fd, "wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())

            encoder_metadata = cls._encoder_metadata()
            manifest = {
                "schema_version": EMBEDDINGS_CACHE_SCHEMA_VERSION,
                "algorithm_version": EMBEDDING_ALGORITHM_VERSION,
                "inputs_sha256": cls._inputs_checksum(texts_to_embed),
                "input_count": len(texts_to_embed),
                "requested_model": encoder_metadata["requested_model"],
                "encoder_mode": encoder_metadata["encoder_mode"],
                "encoder_version": encoder_metadata["encoder_version"],
                "encoder_fingerprint": encoder_metadata["encoder_fingerprint"],
                "vector_shape": list(vectors.shape),
                "vector_dimension": int(vectors.shape[1]),
                "vector_dtype": str(vectors.dtype),
                "vector_sha256": cls._file_checksum(vector_temp),
            }

            manifest_fd, manifest_temp_name = tempfile.mkstemp(
                prefix=".embeddings-manifest-",
                suffix=".json.tmp",
                dir=str(emb_path.parent),
            )
            manifest_temp = Path(manifest_temp_name)
            with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(vector_temp, emb_path)
            vector_temp = None
            os.replace(manifest_temp, manifest_path)
            manifest_temp = None
        finally:
            for temp_path in (vector_temp, manifest_temp):
                if temp_path is not None:
                    try:
                        temp_path.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove temporary embedding cache file [%s]", temp_path)

    @classmethod
    def build_or_load_paper_index(cls, paper_id: str, chunks: list[dict[str, Any]]) -> np.ndarray:
        """Build or load a content- and encoder-matched dense vector index for a paper."""
        if not cls._is_initialized:
            cls.initialize()

        p_dir = paper_dir(paper_id)
        emb_path = p_dir / EMBEDDINGS_FILENAME
        manifest_path = p_dir / EMBEDDINGS_MANIFEST_FILENAME
        texts_to_embed = [f"{chunk.get('section', '')}: {chunk.get('text', '')}" for chunk in chunks]

        cached_vectors = cls._load_cached_vectors(emb_path, manifest_path, texts_to_embed)
        if cached_vectors is not None:
            return cached_vectors

        vectors = cls.encode(texts_to_embed)
        if not cls._vectors_are_valid(vectors, len(texts_to_embed)):
            raise ValueError("Embedding encoder returned non-finite or incompatible vectors")
        if cls._encoder_mode == "fallback" and vectors.shape[1] != FALLBACK_DIMENSION:
            raise ValueError("Fallback encoder returned an unexpected vector dimension")

        try:
            cls._publish_cache_atomically(emb_path, manifest_path, vectors, texts_to_embed)
            logger.info("Saved %d dense vectors and manifest to %s", len(vectors), emb_path)
        except Exception as exc:
            logger.warning("Could not persist embedding cache to disk: %s", exc)

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
        index_fingerprint = cls._encoder_fingerprint()
        query_vec = cls.encode([query])[0]

        # A transformer can fail after a transformer-space index was loaded. The
        # process then remains in fallback mode, so rebuild both operands there.
        if cls._encoder_fingerprint() != index_fingerprint:
            vectors = cls.build_or_load_paper_index(paper_id, chunks)
            query_vec = cls.encode([query])[0]

        # Compute dot products with all chunk vectors (cosine similarity)
        scores = np.dot(vectors, query_vec)  # Shape: (num_chunks,)

        # Rank indices
        ranked_indices = np.argsort(-scores)[:top_k]
        results = []
        for idx in ranked_indices:
            score_val = float(scores[idx])
            results.append((chunks[idx], max(0.0, score_val)))

        return results
