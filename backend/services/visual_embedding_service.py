"""Cache-only paired image/text retrieval for paper figures and tables.

The service deliberately has no synthetic fallback vector space: an image and a
query must be encoded by the same paired model to make their similarity
meaningful.  When the optional model stack or local snapshot is unavailable,
the visual channel reports that state and the existing text channels continue.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from backend.services.network_policy_service import NetworkPolicyService
from backend.services.pdf_service import paper_dir, read_json

logger = logging.getLogger("scholar.visual_embeddings")

DEFAULT_VISUAL_EMBEDDING_MODEL = os.getenv(
    "SCHOLAR_VISUAL_EMBEDDING_MODEL",
    "openai/clip-vit-base-patch32",
)
VISUAL_EMBEDDINGS_FILENAME = "visual_embeddings.npy"
VISUAL_EMBEDDINGS_MANIFEST_FILENAME = "visual_embeddings_manifest.json"
VISUAL_CACHE_SCHEMA_VERSION = 2
VISUAL_ALGORITHM_VERSION = "clip-image-text-retrieval-v2"
VISUAL_ENCODER_VERSION = "transformers-clip-projection-l2-v1"
MAX_IMAGE_BYTES = 18 * 1024 * 1024
MAX_IMAGE_PIXELS = 40_000_000
DEFAULT_BATCH_SIZE = 8
DEFAULT_MIN_SIMILARITY = 0.20

_LOCKS_GUARD = threading.Lock()
_STATE_LOCK = threading.RLock()
_RUNTIME_LOCAL = threading.local()
_INDEX_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class VisualIndex:
    """Vectors and source chunks in identical deterministic row order."""

    vectors: np.ndarray
    chunks: list[dict[str, Any]]


class VisualEmbeddingService:
    """Encode and search local figure images in a paired CLIP vector space."""

    _model: Any = None
    _processor: Any = None
    _is_initialized: bool = False
    _available: bool = False
    _attempted: bool = False
    _requested_model: str = DEFAULT_VISUAL_EMBEDDING_MODEL
    _device: str | None = None
    _load_failure_reason: str | None = None
    _runtime_failure_reason: str | None = None
    _resolved_revision: str | None = None
    _encoder_fingerprint_value: str | None = None
    _encoder_artifact_sha256: str | None = None
    _encoder_artifact_file_count: int | None = None
    _last_request_attempted: bool = False
    _last_request_succeeded: bool | None = None
    _last_request_hit_count: int = 0
    _last_request_best_score: float | None = None

    @staticmethod
    def _digest_encoder_artifacts(
        requested_model: str,
        resolved_revision: str | None,
    ) -> tuple[str, str | None, int]:
        """Hash the exact local weights and processor files used by the encoder."""
        candidate = Path(requested_model).expanduser()
        if candidate.is_dir():
            snapshot_root = candidate.resolve()
        else:
            from huggingface_hub import snapshot_download

            snapshot_root = Path(snapshot_download(
                repo_id=requested_model,
                revision=resolved_revision,
                local_files_only=True,
            )).resolve()

        files = sorted(
            path for path in snapshot_root.rglob("*")
            if path.is_file()
        )
        if not files:
            raise RuntimeError(f"Local visual encoder snapshot is empty: {snapshot_root}")

        digest = hashlib.sha256()
        for path in files:
            relative = path.relative_to(snapshot_root).as_posix()
            digest.update(relative.encode("utf-8"))
            digest.update(b"\0")
            digest.update(str(path.stat().st_size).encode("ascii"))
            digest.update(b"\0")
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(block)
            digest.update(b"\0")

        snapshot_name = snapshot_root.name
        immutable_revision = resolved_revision or (
            snapshot_name if len(snapshot_name) == 40 and all(
                char in "0123456789abcdef" for char in snapshot_name.lower()
            ) else None
        )
        return digest.hexdigest(), immutable_revision, len(files)

    @classmethod
    def initialize(cls, model_name: str | None = None) -> None:
        """Load a pre-provisioned paired encoder without runtime acquisition."""
        requested_model = model_name or DEFAULT_VISUAL_EMBEDDING_MODEL
        with _STATE_LOCK:
            if cls._is_initialized and cls._requested_model == requested_model:
                return

            cls._attempted = True
            cls._requested_model = requested_model
            cls._model = None
            cls._processor = None
            cls._available = False
            cls._device = None
            cls._load_failure_reason = None
            cls._runtime_failure_reason = None
            cls._resolved_revision = None
            cls._encoder_fingerprint_value = None
            cls._encoder_artifact_sha256 = None
            cls._encoder_artifact_file_count = None
            cls._last_request_attempted = False
            cls._last_request_succeeded = None
            cls._last_request_hit_count = 0
            cls._last_request_best_score = None

            try:
                import torch
                from transformers import CLIPModel, CLIPProcessor

                # This retrieval path never acquires assets during a user request.
                NetworkPolicyService.enforce_local_model_cache()
                processor = CLIPProcessor.from_pretrained(
                    requested_model,
                    local_files_only=True,
                )
                model = CLIPModel.from_pretrained(
                    requested_model,
                    local_files_only=True,
                )
                model.eval()

                configured_device = os.getenv(
                    "SCHOLAR_VISUAL_EMBEDDING_DEVICE", "auto"
                ).strip().lower()
                if configured_device not in {"auto", "cpu", "mps", "cuda"}:
                    raise ValueError(
                        "SCHOLAR_VISUAL_EMBEDDING_DEVICE must be one of auto, cpu, mps, or cuda"
                    )
                if configured_device == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("Configured MPS visual embedding device is unavailable")
                if configured_device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("Configured CUDA visual embedding device is unavailable")
                if configured_device == "auto":
                    if torch.backends.mps.is_available():
                        device = "mps"
                    elif torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                else:
                    device = configured_device

                config_dict = model.config.to_dict()
                resolved_revision = getattr(model.config, "_commit_hash", None)
                artifact_sha256, resolved_revision, artifact_file_count = (
                    cls._digest_encoder_artifacts(requested_model, resolved_revision)
                )
                descriptor = {
                    "requested_model": requested_model,
                    "resolved_revision": resolved_revision,
                    "encoder_version": VISUAL_ENCODER_VERSION,
                    "model_config_sha256": hashlib.sha256(
                        json.dumps(
                            config_dict,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                            default=str,
                        ).encode("utf-8")
                    ).hexdigest(),
                    "encoder_artifact_sha256": artifact_sha256,
                    "encoder_artifact_file_count": artifact_file_count,
                    "processor_class": type(processor).__name__,
                    "model_class": type(model).__name__,
                }
                fingerprint = hashlib.sha256(
                    json.dumps(
                        descriptor,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                    ).encode("utf-8")
                ).hexdigest()

                model = model.to(device)
                cls._processor = processor
                cls._model = model
                cls._device = device
                cls._resolved_revision = resolved_revision
                cls._encoder_fingerprint_value = fingerprint
                cls._encoder_artifact_sha256 = artifact_sha256
                cls._encoder_artifact_file_count = artifact_file_count
                cls._available = True
                cls._is_initialized = True
                logger.info(
                    "Visual embedding model loaded from local cache [%s] on %s.",
                    requested_model,
                    device,
                )
            except Exception as exc:
                cls._load_failure_reason = f"{type(exc).__name__}: {exc}"
                cls._is_initialized = True
                cls._available = False
                logger.info(
                    "Visual embedding channel unavailable for [%s]: %s",
                    requested_model,
                    cls._load_failure_reason,
                )

    @classmethod
    def minimum_similarity(cls) -> float:
        """Return the conservative, configurable pre-calibration score floor."""
        raw = os.getenv("SCHOLAR_VISUAL_MIN_SIMILARITY", str(DEFAULT_MIN_SIMILARITY))
        try:
            return max(-1.0, min(float(raw), 1.0))
        except ValueError:
            logger.warning(
                "Invalid SCHOLAR_VISUAL_MIN_SIMILARITY=%r; using %.2f.",
                raw,
                DEFAULT_MIN_SIMILARITY,
            )
            return DEFAULT_MIN_SIMILARITY

    @classmethod
    def _begin_runtime_operation(cls) -> None:
        _RUNTIME_LOCAL.failure_reason = None

    @classmethod
    def _record_runtime_success(cls) -> None:
        _RUNTIME_LOCAL.failure_reason = None
        with _STATE_LOCK:
            # Publish one coherent completed outcome; a later completion wins.
            cls._runtime_failure_reason = None
            cls._last_request_attempted = True
            cls._last_request_succeeded = True
            cls._last_request_hit_count = 0
            cls._last_request_best_score = None

    @classmethod
    def _record_runtime_failure(cls, reason: str) -> None:
        _RUNTIME_LOCAL.failure_reason = reason
        with _STATE_LOCK:
            cls._runtime_failure_reason = reason
            cls._last_request_attempted = True
            cls._last_request_succeeded = False
            cls._last_request_hit_count = 0
            cls._last_request_best_score = None

    @staticmethod
    def _current_operation_failure() -> str | None:
        value = getattr(_RUNTIME_LOCAL, "failure_reason", None)
        return str(value) if value else None

    @classmethod
    def _record_search_outcome(
        cls,
        *,
        attempted: bool,
        succeeded: bool | None,
        hit_count: int,
        best_score: float | None,
        failure_reason: str | None = None,
    ) -> None:
        with _STATE_LOCK:
            cls._runtime_failure_reason = failure_reason
            cls._last_request_attempted = attempted
            cls._last_request_succeeded = succeeded
            cls._last_request_hit_count = hit_count
            cls._last_request_best_score = best_score

    @classmethod
    def status(cls) -> dict[str, Any]:
        """Return separate model-load and most-recent request diagnostics."""
        with _STATE_LOCK:
            fallback_reason = cls._load_failure_reason or cls._runtime_failure_reason
            return {
                "attempted": cls._attempted,
                "active": cls._available and cls._runtime_failure_reason is None,
                "model_loaded": cls._available,
                "requested_model": cls._requested_model,
                "resolved_revision": cls._resolved_revision,
                "encoder_version": VISUAL_ENCODER_VERSION,
                "encoder_fingerprint": cls._encoder_fingerprint_value,
                "encoder_artifact_sha256": cls._encoder_artifact_sha256,
                "encoder_artifact_file_count": cls._encoder_artifact_file_count,
                "device": cls._device,
                "fallback_reason": fallback_reason,
                "load_failure_reason": cls._load_failure_reason,
                "runtime_failure_reason": cls._runtime_failure_reason,
                "last_request_attempted": cls._last_request_attempted,
                "last_request_succeeded": cls._last_request_succeeded,
                "last_request_hit_count": cls._last_request_hit_count,
                "last_request_best_score": cls._last_request_best_score,
                "minimum_similarity": cls.minimum_similarity(),
                "threshold_calibrated": False,
                "cache_only": True,
            }

    @classmethod
    def encoder_bundle(cls) -> tuple[Any, Any, str, str] | None:
        """Expose the loaded cache-only CLIP bundle to compatible retrievers."""
        if not cls._is_initialized:
            cls.initialize()
        with _STATE_LOCK:
            if (
                not cls._available
                or cls._model is None
                or cls._processor is None
                or not cls._device
                or not cls._encoder_fingerprint_value
            ):
                return None
            return (
                cls._model,
                cls._processor,
                cls._device,
                cls._encoder_fingerprint_value,
            )

    @classmethod
    def release(cls) -> None:
        """Release loaded encoder objects, primarily after isolated preflight checks."""
        with _STATE_LOCK:
            cls._model = None
            cls._processor = None
            cls._available = False
            cls._is_initialized = False
            cls._device = None

    @classmethod
    def _encoder_metadata(cls) -> dict[str, Any]:
        if (
            not cls._available
            or not cls._encoder_fingerprint_value
            or not cls._encoder_artifact_sha256
        ):
            raise RuntimeError("Visual embedding encoder is unavailable")
        return {
            "requested_model": cls._requested_model,
            "resolved_revision": cls._resolved_revision,
            "encoder_version": VISUAL_ENCODER_VERSION,
            "encoder_fingerprint": cls._encoder_fingerprint_value,
            "encoder_artifact_sha256": cls._encoder_artifact_sha256,
            "encoder_artifact_file_count": cls._encoder_artifact_file_count,
        }

    @classmethod
    def _batch_size(cls) -> int:
        raw = os.getenv("SCHOLAR_VISUAL_EMBEDDING_BATCH_SIZE", str(DEFAULT_BATCH_SIZE))
        try:
            return max(1, min(int(raw), 32))
        except ValueError:
            logger.warning(
                "Invalid SCHOLAR_VISUAL_EMBEDDING_BATCH_SIZE=%r; using %d.",
                raw,
                DEFAULT_BATCH_SIZE,
            )
            return DEFAULT_BATCH_SIZE

    @staticmethod
    def _move_inputs(inputs: Any, device: str) -> dict[str, Any]:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }

    @staticmethod
    def _as_feature_tensor(output: Any) -> Any:
        if hasattr(output, "pooler_output"):
            return output.pooler_output
        if hasattr(output, "image_embeds"):
            return output.image_embeds
        if hasattr(output, "text_embeds"):
            return output.text_embeds
        return output

    @classmethod
    def _normalize_features(cls, output: Any) -> np.ndarray:
        import torch

        features = cls._as_feature_tensor(output)
        if not isinstance(features, torch.Tensor) or features.ndim != 2:
            raise ValueError("Paired encoder returned an incompatible feature tensor")
        normalized = torch.nn.functional.normalize(features.float(), p=2, dim=1)
        vectors = normalized.detach().cpu().numpy().astype(np.float32)
        if not cls._vectors_are_valid(vectors, vectors.shape[0]):
            raise ValueError("Paired encoder returned non-finite visual features")
        return vectors

    @classmethod
    def encode_texts(cls, texts: list[str]) -> np.ndarray | None:
        """Encode query text in the paired model's shared vector space."""
        if not texts:
            return None
        if not cls._is_initialized:
            cls.initialize()
        if not cls._available or cls._model is None or cls._processor is None or not cls._device:
            return None

        cls._begin_runtime_operation()
        try:
            import torch

            encoded = cls._processor(
                text=texts,
                padding=True,
                truncation=True,
                return_tensors="pt",
            )
            inputs = cls._move_inputs(encoded, cls._device)
            with torch.no_grad():
                output = cls._model.get_text_features(**inputs)
            vectors = cls._normalize_features(output)
            cls._record_runtime_success()
            return vectors
        except Exception as exc:
            reason = f"text encoding failed: {type(exc).__name__}: {exc}"
            cls._record_runtime_failure(reason)
            logger.warning("Visual query encoding failed: %s", exc)
            return None

    @classmethod
    def encode_images(cls, images: list[Any]) -> np.ndarray | None:
        """Encode already-decoded RGB images in bounded batches."""
        if not images:
            return None
        if not cls._is_initialized:
            cls.initialize()
        if not cls._available or cls._model is None or cls._processor is None or not cls._device:
            return None

        cls._begin_runtime_operation()
        batches: list[np.ndarray] = []
        try:
            import torch

            batch_size = cls._batch_size()
            for start in range(0, len(images), batch_size):
                encoded = cls._processor(
                    images=images[start : start + batch_size],
                    return_tensors="pt",
                )
                inputs = cls._move_inputs(encoded, cls._device)
                with torch.no_grad():
                    output = cls._model.get_image_features(**inputs)
                batches.append(cls._normalize_features(output))
            vectors = np.concatenate(batches, axis=0).astype(np.float32, copy=False)
            if not cls._vectors_are_valid(vectors, len(images)):
                raise ValueError("Paired encoder returned incompatible image features")
            cls._record_runtime_success()
            return vectors
        except Exception as exc:
            reason = f"image encoding failed: {type(exc).__name__}: {exc}"
            cls._record_runtime_failure(reason)
            logger.warning("Visual image encoding failed: %s", exc)
            return None

    @staticmethod
    def _file_checksum(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        return hashlib.sha256(
            json.dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest()

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

    @staticmethod
    def _source_id(chunk: dict[str, Any], default_paper_id: str) -> str:
        return str(
            chunk.get("source_paper_id")
            or chunk.get("document_id")
            or default_paper_id
            or ""
        ).strip()

    @staticmethod
    def _candidate_richness(chunk: dict[str, Any]) -> tuple[int, int]:
        text = str(chunk.get("body_text") or chunk.get("caption") or chunk.get("text") or "")
        metadata_count = sum(
            bool(chunk.get(key))
            for key in ("figure_id", "caption", "body_text", "bbox_normalized", "bbox")
        )
        return metadata_count, len(text)

    @classmethod
    def _prepare_rows(
        cls,
        source_paper_id: str,
        chunks: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[Path]]:
        """Resolve, deduplicate, and hash safe image-bearing chunks."""
        by_image: dict[str, dict[str, Any]] = {}
        for chunk in chunks:
            if not chunk.get("is_figure_chunk"):
                continue
            if cls._source_id(chunk, source_paper_id) != source_paper_id:
                continue
            image_file = str(chunk.get("image_file") or "").strip()
            if not image_file or Path(image_file).name != image_file:
                continue
            current = by_image.get(image_file)
            if current is None or cls._candidate_richness(chunk) > cls._candidate_richness(current):
                by_image[image_file] = chunk

        rows: list[dict[str, Any]] = []
        candidates: list[dict[str, Any]] = []
        paths: list[Path] = []
        for image_file, chunk in sorted(by_image.items()):
            path = paper_dir(source_paper_id) / "figures" / image_file
            try:
                if not path.is_file() or path.stat().st_size > MAX_IMAGE_BYTES:
                    continue
                image_sha256 = cls._file_checksum(path)
            except OSError:
                continue

            local_id_kind = "chunk_id" if chunk.get("chunk_id") else (
                "evidence_id" if chunk.get("evidence_id") else "image_file"
            )
            local_id = str(
                chunk.get("chunk_id") or chunk.get("evidence_id") or image_file
            )
            rows.append({
                "row": len(rows),
                "source_id": source_paper_id,
                "local_id_kind": local_id_kind,
                "local_id": local_id,
                "figure_id": str(chunk.get("figure_id") or ""),
                "image_file": image_file,
                "image_sha256": image_sha256,
                "page": int(chunk.get("page") or 0),
            })
            candidates.append(chunk)
            paths.append(path)
        return rows, candidates, paths

    @staticmethod
    def _load_rgb_images(paths: list[Path]) -> tuple[list[Any], list[int]]:
        from PIL import Image

        images: list[Any] = []
        valid_indices: list[int] = []
        for index, path in enumerate(paths):
            try:
                with Image.open(path) as image:
                    width, height = image.size
                    if width <= 0 or height <= 0 or width * height > MAX_IMAGE_PIXELS:
                        raise ValueError(f"unsafe image dimensions: {width}x{height}")
                    image.load()
                    images.append(image.convert("RGB").copy())
                    valid_indices.append(index)
            except Exception as exc:
                logger.warning("Skipping invalid visual retrieval image [%s]: %s", path, exc)
        return images, valid_indices

    @classmethod
    def _load_cached_vectors(
        cls,
        emb_path: Path,
        manifest_path: Path,
        source_paper_id: str,
        rows: list[dict[str, Any]],
    ) -> np.ndarray | None:
        if not emb_path.is_file() or not manifest_path.is_file():
            return None
        try:
            metadata = cls._encoder_metadata()
            manifest = read_json(manifest_path)
            required = {
                "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
                "algorithm_version": VISUAL_ALGORITHM_VERSION,
                "source_paper_id": source_paper_id,
                **metadata,
                "rows": rows,
                "inputs_sha256": cls._canonical_sha256(rows),
                "input_count": len(rows),
                "vector_dtype": "float32",
            }
            if not isinstance(manifest, dict) or any(
                manifest.get(key) != value for key, value in required.items()
            ):
                return None

            shape = manifest.get("vector_shape")
            dimension = manifest.get("vector_dimension")
            checksum = manifest.get("vector_sha256")
            if (
                not isinstance(shape, list)
                or len(shape) != 2
                or any(type(value) is not int for value in shape)
                or shape[0] != len(rows)
                or shape[1] <= 0
                or type(dimension) is not int
                or dimension != shape[1]
                or not isinstance(checksum, str)
                or len(checksum) != 64
                or cls._file_checksum(emb_path) != checksum
            ):
                return None
            vectors = np.load(str(emb_path), allow_pickle=False)
            if not cls._vectors_are_valid(vectors, len(rows), shape, dimension):
                return None
            return vectors
        except Exception as exc:
            logger.warning("Visual embedding cache validation failed [%s]: %s", emb_path, exc)
            return None

    @classmethod
    def _publish_cache_atomically(
        cls,
        emb_path: Path,
        manifest_path: Path,
        source_paper_id: str,
        rows: list[dict[str, Any]],
        vectors: np.ndarray,
    ) -> None:
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        vector_temp: Path | None = None
        manifest_temp: Path | None = None
        try:
            vector_fd, vector_name = tempfile.mkstemp(
                prefix=".visual-embeddings-",
                suffix=".npy.tmp",
                dir=str(emb_path.parent),
            )
            vector_temp = Path(vector_name)
            with os.fdopen(vector_fd, "wb") as handle:
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())

            metadata = cls._encoder_metadata()
            manifest = {
                "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
                "algorithm_version": VISUAL_ALGORITHM_VERSION,
                "source_paper_id": source_paper_id,
                **metadata,
                "rows": rows,
                "inputs_sha256": cls._canonical_sha256(rows),
                "input_count": len(rows),
                "vector_shape": list(vectors.shape),
                "vector_dimension": int(vectors.shape[1]),
                "vector_dtype": str(vectors.dtype),
                "vector_sha256": cls._file_checksum(vector_temp),
            }
            manifest_fd, manifest_name = tempfile.mkstemp(
                prefix=".visual-embeddings-manifest-",
                suffix=".json.tmp",
                dir=str(emb_path.parent),
            )
            manifest_temp = Path(manifest_name)
            with os.fdopen(manifest_fd, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, ensure_ascii=False, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            os.replace(vector_temp, emb_path)
            vector_temp = None
            os.replace(manifest_temp, manifest_path)
            manifest_temp = None
            cls._fsync_directory(emb_path.parent)
        finally:
            for temporary in (vector_temp, manifest_temp):
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        logger.warning("Could not remove visual cache temporary file [%s]", temporary)

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        try:
            descriptor = os.open(str(path), os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(descriptor)
        except OSError:
            pass
        finally:
            os.close(descriptor)

    @staticmethod
    @contextmanager
    def _interprocess_index_lock(path: Path) -> Iterator[None]:
        """Serialize cache validation/publication across backend workers."""
        import fcntl

        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def build_or_load_paper_index(
        cls,
        source_paper_id: str,
        figure_chunks: list[dict[str, Any]],
    ) -> VisualIndex | None:
        """Return a validated source-local visual index, building it when needed."""
        if not cls._is_initialized:
            cls.initialize()
        if not cls._available:
            return None

        rows, candidates, paths = cls._prepare_rows(source_paper_id, figure_chunks)
        if not rows:
            cls._record_runtime_failure(
                f"no safe local figure images were available for source {source_paper_id}"
            )
            return None

        directory = paper_dir(source_paper_id)
        emb_path = directory / VISUAL_EMBEDDINGS_FILENAME
        manifest_path = directory / VISUAL_EMBEDDINGS_MANIFEST_FILENAME
        lock_key = str(directory.resolve())
        with _LOCKS_GUARD:
            build_lock = _INDEX_LOCKS.setdefault(lock_key, threading.Lock())

        with build_lock, cls._interprocess_index_lock(
            directory / ".visual_embeddings.lock"
        ):
            cached = cls._load_cached_vectors(
                emb_path,
                manifest_path,
                source_paper_id,
                rows,
            )
            if cached is not None:
                return VisualIndex(vectors=cached, chunks=candidates)

            images, valid_indices = cls._load_rgb_images(paths)
            if not images:
                cls._record_runtime_failure(
                    f"no decodable local figure images were available for source {source_paper_id}"
                )
                return None
            if len(valid_indices) != len(rows):
                rows = [rows[index] for index in valid_indices]
                candidates = [candidates[index] for index in valid_indices]
                for index, row in enumerate(rows):
                    row["row"] = index
                cached = cls._load_cached_vectors(
                    emb_path,
                    manifest_path,
                    source_paper_id,
                    rows,
                )
                if cached is not None:
                    return VisualIndex(vectors=cached, chunks=candidates)

            vectors = cls.encode_images(images)
            if vectors is None or not cls._vectors_are_valid(vectors, len(rows)):
                return None
            try:
                cls._publish_cache_atomically(
                    emb_path,
                    manifest_path,
                    source_paper_id,
                    rows,
                    vectors,
                )
                logger.info(
                    "Saved %d visual vectors and manifest to %s",
                    len(vectors),
                    emb_path,
                )
            except Exception as exc:
                logger.warning("Could not persist visual embedding cache: %s", exc)
            return VisualIndex(vectors=vectors, chunks=candidates)

    @classmethod
    def search_visual(
        cls,
        default_paper_id: str,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[tuple[dict[str, Any], float]]:
        """Rank image chunks and retain only conservative similarity-floor hits."""
        if not query.strip() or top_k <= 0:
            cls._record_search_outcome(
                attempted=False,
                succeeded=None,
                hit_count=0,
                best_score=None,
            )
            return []

        grouped: dict[str, list[dict[str, Any]]] = {}
        for chunk in chunks:
            if not chunk.get("is_figure_chunk") or not chunk.get("image_file"):
                continue
            source_id = cls._source_id(chunk, default_paper_id)
            if source_id:
                grouped.setdefault(source_id, []).append(chunk)
        if not grouped:
            cls._record_search_outcome(
                attempted=False,
                succeeded=None,
                hit_count=0,
                best_score=None,
            )
            return []

        query_vectors = cls.encode_texts([query])
        if query_vectors is None or not cls._vectors_are_valid(query_vectors, 1):
            failure_reason = cls._current_operation_failure() or "visual query encoding failed"
            cls._record_search_outcome(
                attempted=True,
                succeeded=False,
                hit_count=0,
                best_score=None,
                failure_reason=failure_reason,
            )
            return []
        query_vector = query_vectors[0]

        scored: list[tuple[dict[str, Any], float]] = []
        best_score: float | None = None
        search_failure: str | None = None
        for source_paper_id in sorted(grouped):
            visual_index = cls.build_or_load_paper_index(
                source_paper_id,
                grouped[source_paper_id],
            )
            if visual_index is None:
                search_failure = search_failure or cls._current_operation_failure()
                continue
            if visual_index.vectors.shape[1] != query_vector.shape[0]:
                search_failure = search_failure or "visual query/image dimensions do not match"
                continue
            scores = np.dot(visual_index.vectors, query_vector)
            for chunk, score in zip(visual_index.chunks, scores):
                raw_score = float(score)
                if math.isfinite(raw_score):
                    best_score = raw_score if best_score is None else max(best_score, raw_score)
                    scored.append((chunk, raw_score))

        scored.sort(
            key=lambda item: (
                -item[1],
                cls._source_id(item[0], default_paper_id),
                str(item[0].get("image_file") or ""),
            )
        )
        minimum_similarity = cls.minimum_similarity()
        eligible = [item for item in scored if item[1] >= minimum_similarity][:top_k]
        succeeded = search_failure is None
        cls._record_search_outcome(
            attempted=True,
            succeeded=succeeded,
            hit_count=len(eligible),
            best_score=best_score,
            failure_reason=search_failure,
        )
        return eligible
