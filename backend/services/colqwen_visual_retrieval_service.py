"""Strict-local ColQwen2 late-interaction retrieval over rendered PDF pages.

The service is intentionally optional. It only loads a pre-provisioned local
snapshot and never acquires model files while answering a user query. Page
vectors are stored as one flat float16 matrix plus offsets because ColQwen2
preserves page aspect ratio and therefore emits a variable number of vectors.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover - unavailable on Windows
    fcntl = None  # type: ignore[assignment]

from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType
from backend.services.network_policy_service import NetworkPolicyService
from backend.services.pdf_service import paper_dir, read_json
from backend.services.visual_page_retrieval_service import (
    VisualPageSearchResult,
    VisualPageSearchStatus,
)


logger = logging.getLogger("scholar.colqwen_visual_pages")

DEFAULT_MODEL = "vidore/colqwen2-v1.0-hf"
BACKEND_NAME = "colqwen2-late-interaction-v1"
ENCODER_VERSION = "transformers-colqwen2-retrieval-v1"
INDEX_SCHEMA_VERSION = 1
VECTORS_FILE = "colqwen_page_vectors.npy"
OFFSETS_FILE = "colqwen_page_offsets.npy"
PAGE_METADATA_FILE = "colqwen_page_metadata.json"
MANIFEST_FILE = "colqwen_page_manifest.json"
DEFAULT_MIN_SCORE = -1_000_000_000.0

_STATE_LOCK = threading.RLock()
_LOCKS_GUARD = threading.Lock()
_INDEX_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class ColQwenSourceIndex:
    vectors: np.ndarray
    offsets: np.ndarray
    page_metadata: list[dict[str, Any]]

    def page_vectors(self, index: int) -> np.ndarray:
        start = int(self.offsets[index])
        end = int(self.offsets[index + 1])
        return self.vectors[start:end]


class ColQwenVisualRetrievalService:
    """Encode document pages and queries with a native ColQwen2 retriever."""

    _model: Any = None
    _processor: Any = None
    _initialized = False
    _available = False
    _requested_model = DEFAULT_MODEL
    _resolved_revision: str | None = None
    _device: str | None = None
    _dtype: str | None = None
    _encoder_fingerprint: str | None = None
    _artifact_sha256: str | None = None
    _artifact_file_count: int | None = None
    _load_failure_reason: str | None = None

    @staticmethod
    def minimum_score() -> float:
        raw = os.getenv("SCHOLAR_COLQWEN_MIN_SCORE", str(DEFAULT_MIN_SCORE))
        try:
            value = float(raw)
            return value if math.isfinite(value) else DEFAULT_MIN_SCORE
        except ValueError:
            return DEFAULT_MIN_SCORE

    @staticmethod
    def _sha256_file(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @classmethod
    def _digest_encoder_artifacts(
        cls,
        requested_model: str,
        resolved_revision: str | None,
    ) -> tuple[str, str | None, int]:
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

        files = sorted(path for path in snapshot_root.rglob("*") if path.is_file())
        if not files:
            raise RuntimeError(f"Local ColQwen2 snapshot is empty: {snapshot_root}")
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
            snapshot_name
            if len(snapshot_name) == 40
            and all(char in "0123456789abcdef" for char in snapshot_name.lower())
            else None
        )
        return digest.hexdigest(), immutable_revision, len(files)

    @classmethod
    def initialize(cls, model_name: str | None = None) -> None:
        requested_model = (
            model_name
            or os.getenv("SCHOLAR_DOCUMENT_VISUAL_MODEL", DEFAULT_MODEL).strip()
            or DEFAULT_MODEL
        )
        with _STATE_LOCK:
            if cls._initialized and cls._requested_model == requested_model:
                return
            cls.release()
            cls._initialized = True
            cls._requested_model = requested_model
            try:
                import torch
                from transformers import ColQwen2ForRetrieval, ColQwen2Processor

                NetworkPolicyService.enforce_local_model_cache()
                configured_device = os.getenv(
                    "SCHOLAR_DOCUMENT_VISUAL_DEVICE", "auto"
                ).strip().lower()
                if configured_device not in {"auto", "cpu", "mps", "cuda"}:
                    raise ValueError(
                        "SCHOLAR_DOCUMENT_VISUAL_DEVICE must be auto, cpu, mps, or cuda"
                    )
                if configured_device == "auto":
                    if torch.backends.mps.is_available():
                        device = "mps"
                    elif torch.cuda.is_available():
                        device = "cuda"
                    else:
                        device = "cpu"
                else:
                    device = configured_device
                if device == "mps" and not torch.backends.mps.is_available():
                    raise RuntimeError("Configured MPS device is unavailable")
                if device == "cuda" and not torch.cuda.is_available():
                    raise RuntimeError("Configured CUDA device is unavailable")

                configured_dtype = os.getenv(
                    "SCHOLAR_DOCUMENT_VISUAL_DTYPE", "auto"
                ).strip().lower()
                dtype_options: dict[str, Any] = {
                    "float32": torch.float32,
                    "float16": torch.float16,
                    "bfloat16": torch.bfloat16,
                }
                if configured_dtype not in {"auto", *dtype_options}:
                    raise ValueError(
                        "SCHOLAR_DOCUMENT_VISUAL_DTYPE must be auto, float32, "
                        "float16, or bfloat16"
                    )
                # CPU reduced-precision support varies substantially by host.
                # FP32 is the safe and, on the reference Apple CPU, fastest
                # default; explicit overrides remain fingerprinted for capable
                # deployment hardware.
                load_dtype: Any = (
                    torch.float32
                    if configured_dtype == "auto" and device == "cpu"
                    else dtype_options.get(configured_dtype, torch.float16)
                )
                processor = ColQwen2Processor.from_pretrained(
                    requested_model,
                    local_files_only=True,
                )
                model = ColQwen2ForRetrieval.from_pretrained(
                    requested_model,
                    local_files_only=True,
                    dtype=load_dtype,
                )
                model.eval()
                dtype = next(model.parameters()).dtype
                resolved_revision = getattr(model.config, "_commit_hash", None)
                artifact_sha256, resolved_revision, file_count = (
                    cls._digest_encoder_artifacts(requested_model, resolved_revision)
                )
                descriptor = {
                    "requested_model": requested_model,
                    "resolved_revision": resolved_revision,
                    "encoder_version": ENCODER_VERSION,
                    "artifact_sha256": artifact_sha256,
                    "artifact_file_count": file_count,
                    "model_config": model.config.to_dict(),
                    "processor_class": type(processor).__name__,
                    "model_class": type(model).__name__,
                    "dtype": str(dtype),
                }
                fingerprint = hashlib.sha256(json.dumps(
                    descriptor,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                ).encode("utf-8")).hexdigest()

                cls._model = model.to(device)
                cls._processor = processor
                cls._available = True
                cls._device = device
                cls._dtype = str(dtype).replace("torch.", "")
                cls._resolved_revision = resolved_revision
                cls._encoder_fingerprint = fingerprint
                cls._artifact_sha256 = artifact_sha256
                cls._artifact_file_count = file_count
                cls._load_failure_reason = None
                logger.info("Loaded cache-only ColQwen2 retriever [%s] on %s", requested_model, device)
            except Exception as exc:
                cls._available = False
                cls._load_failure_reason = f"{type(exc).__name__}: {exc}"
                logger.info("ColQwen2 visual retriever unavailable: %s", cls._load_failure_reason)

    @classmethod
    def release(cls) -> None:
        with _STATE_LOCK:
            cls._model = None
            cls._processor = None
            cls._initialized = False
            cls._available = False
            cls._resolved_revision = None
            cls._device = None
            cls._dtype = None
            cls._encoder_fingerprint = None
            cls._artifact_sha256 = None
            cls._artifact_file_count = None
            cls._load_failure_reason = None

    @classmethod
    def status(cls) -> dict[str, Any]:
        if not cls._initialized:
            cls.initialize()
        with _STATE_LOCK:
            return {
                "backend": BACKEND_NAME,
                "model_loaded": cls._available,
                "requested_model": cls._requested_model,
                "resolved_revision": cls._resolved_revision,
                "device": cls._device,
                "dtype": cls._dtype,
                "encoder_fingerprint": cls._encoder_fingerprint,
                "encoder_artifact_sha256": cls._artifact_sha256,
                "encoder_artifact_file_count": cls._artifact_file_count,
                "fallback_reason": cls._load_failure_reason,
                "cache_only": True,
            }

    @classmethod
    def _bundle(cls) -> tuple[Any, Any, str, str] | None:
        if not cls._initialized:
            cls.initialize()
        with _STATE_LOCK:
            if not (
                cls._available
                and cls._model is not None
                and cls._processor is not None
                and cls._device
                and cls._encoder_fingerprint
            ):
                return None
            return cls._model, cls._processor, cls._device, cls._encoder_fingerprint

    @staticmethod
    def _move_inputs(inputs: Any, device: str) -> dict[str, Any]:
        return {
            key: value.to(device) if hasattr(value, "to") else value
            for key, value in dict(inputs).items()
        }

    @classmethod
    def _page_units(cls, source_id: str) -> list[VisualDocumentUnit]:
        path = paper_dir(source_id) / "visual_units.json"
        if not path.is_file():
            return []
        try:
            units = [VisualDocumentUnit.model_validate(item) for item in read_json(path)]
        except Exception as exc:
            logger.warning("Invalid visual units for [%s]: %s", source_id, exc)
            return []
        return sorted(
            (unit for unit in units if unit.unit_type == VisualUnitType.PAGE),
            key=lambda unit: unit.page,
        )

    @classmethod
    def _encode_query(
        cls,
        query: str,
        model: Any,
        processor: Any,
        device: str,
    ) -> np.ndarray:
        import torch

        encoded = processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
        )
        inputs = cls._move_inputs(encoded, device)
        with torch.inference_mode():
            output = model(**inputs, use_cache=False)
        mask = inputs["attention_mask"][0].bool()
        embeddings = output.embeddings[0, mask].float()
        vectors = embeddings.detach().cpu().numpy().astype(np.float32)
        if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
            raise ValueError("ColQwen2 returned invalid query embeddings")
        return vectors

    @classmethod
    def _encode_pages(
        cls,
        image_paths: list[Path],
        model: Any,
        processor: Any,
        device: str,
    ) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
        import torch
        from PIL import Image

        page_vectors: list[np.ndarray] = []
        page_metadata: list[dict[str, Any]] = []
        image_token_id = int(model.config.vlm_config.image_token_id)
        merge_size = int(getattr(processor.image_processor, "merge_size", 1))

        for path in image_paths:
            with Image.open(path) as opened:
                image = opened.convert("RGB").copy()
            encoded = processor(images=[image], return_tensors="pt")
            input_ids_cpu = np.asarray(encoded["input_ids"][0].cpu(), dtype=np.int64)
            attention_cpu = np.asarray(encoded["attention_mask"][0].cpu(), dtype=bool)
            valid_original_positions = np.flatnonzero(attention_cpu)
            original_to_valid = {
                int(original): index
                for index, original in enumerate(valid_original_positions.tolist())
            }
            image_original_positions = np.flatnonzero(
                (input_ids_cpu == image_token_id) & attention_cpu
            )
            image_vector_positions = [
                original_to_valid[int(position)] for position in image_original_positions
            ]
            grid = np.asarray(encoded["image_grid_thw"][0].cpu(), dtype=np.int64)
            grid_h = max(int(grid[1]) // max(merge_size, 1), 1)
            grid_w = max(int(grid[2]) // max(merge_size, 1), 1)

            inputs = cls._move_inputs(encoded, device)
            with torch.inference_mode():
                output = model(**inputs, use_cache=False)
            mask = inputs["attention_mask"][0].bool()
            embeddings = output.embeddings[0, mask].float()
            vectors = embeddings.detach().cpu().numpy().astype(np.float16)
            if vectors.ndim != 2 or not vectors.size or not np.isfinite(vectors).all():
                raise ValueError(f"ColQwen2 returned invalid page embeddings for {path}")
            if any(position >= vectors.shape[0] for position in image_vector_positions):
                raise ValueError(f"Image-token mapping exceeds page vectors for {path}")

            contiguous = bool(image_vector_positions) and image_vector_positions == list(range(
                image_vector_positions[0], image_vector_positions[0] + len(image_vector_positions)
            ))
            metadata: dict[str, Any] = {
                "token_count": int(vectors.shape[0]),
                "image_token_count": len(image_vector_positions),
                "image_token_offset": image_vector_positions[0] if contiguous else None,
                "image_token_positions": None if contiguous else image_vector_positions,
                "grid_height": grid_h,
                "grid_width": grid_w,
                "merge_size": merge_size,
            }
            page_vectors.append(vectors)
            page_metadata.append(metadata)
        return page_vectors, page_metadata

    @staticmethod
    def _canonical_sha256(value: Any) -> str:
        return hashlib.sha256(json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")).hexdigest()

    @classmethod
    def _manifest_identity(
        cls,
        source_id: str,
        units: list[VisualDocumentUnit],
        encoder_fingerprint: str,
    ) -> dict[str, Any]:
        rows = [
            {
                "visual_id": unit.visual_id,
                "page": unit.page,
                "image_relpath": unit.image_relpath,
                "image_sha256": unit.image_sha256,
                "width_px": unit.width_px,
                "height_px": unit.height_px,
            }
            for unit in units
        ]
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "algorithm_version": BACKEND_NAME,
            "source_paper_id": source_id,
            "encoder_fingerprint": encoder_fingerprint,
            "vector_dtype": "float16",
            "offset_dtype": "int64",
            "rows": rows,
            "rows_sha256": cls._canonical_sha256(rows),
        }

    @classmethod
    def _load_index(
        cls,
        directory: Path,
        identity: dict[str, Any],
    ) -> ColQwenSourceIndex | None:
        vectors_path = directory / VECTORS_FILE
        offsets_path = directory / OFFSETS_FILE
        metadata_path = directory / PAGE_METADATA_FILE
        manifest_path = directory / MANIFEST_FILE
        if not all(path.is_file() for path in (
            vectors_path, offsets_path, metadata_path, manifest_path
        )):
            return None
        try:
            manifest = read_json(manifest_path)
            if any(manifest.get(key) != value for key, value in identity.items()):
                return None
            expected_checksums = manifest.get("checksums") or {}
            for name, path in (
                (VECTORS_FILE, vectors_path),
                (OFFSETS_FILE, offsets_path),
                (PAGE_METADATA_FILE, metadata_path),
            ):
                if expected_checksums.get(name) != cls._sha256_file(path):
                    return None
            vectors = np.load(vectors_path, allow_pickle=False)
            offsets = np.load(offsets_path, allow_pickle=False)
            page_metadata = read_json(metadata_path)
            page_count = len(identity["rows"])
            if (
                vectors.dtype != np.float16
                or vectors.ndim != 2
                or vectors.shape[1] <= 0
                or not np.isfinite(vectors).all()
                or offsets.dtype != np.int64
                or offsets.ndim != 1
                or len(offsets) != page_count + 1
                or int(offsets[0]) != 0
                or int(offsets[-1]) != vectors.shape[0]
                or np.any(offsets[1:] < offsets[:-1])
                or not isinstance(page_metadata, list)
                or len(page_metadata) != page_count
            ):
                return None
            for index, metadata in enumerate(page_metadata):
                if int(metadata.get("token_count") or -1) != int(offsets[index + 1] - offsets[index]):
                    return None
            return ColQwenSourceIndex(vectors, offsets, page_metadata)
        except Exception as exc:
            logger.warning("Invalid ColQwen2 page cache [%s]: %s", directory, exc)
            return None

    @classmethod
    def _publish_index(
        cls,
        directory: Path,
        identity: dict[str, Any],
        index: ColQwenSourceIndex,
    ) -> None:
        directory.mkdir(parents=True, exist_ok=True)
        temporary: list[Path] = []
        try:
            payloads: list[tuple[str, str, Any]] = [
                (VECTORS_FILE, "npy", index.vectors),
                (OFFSETS_FILE, "npy", index.offsets),
                (PAGE_METADATA_FILE, "json", index.page_metadata),
            ]
            staged: dict[str, Path] = {}
            for final_name, kind, payload in payloads:
                descriptor, temp_name = tempfile.mkstemp(
                    prefix=f".{final_name}.", suffix=".tmp", dir=directory
                )
                temp_path = Path(temp_name)
                temporary.append(temp_path)
                if kind == "npy":
                    with os.fdopen(descriptor, "wb") as handle:
                        np.save(handle, payload, allow_pickle=False)
                        handle.flush()
                        os.fsync(handle.fileno())
                else:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                        json.dump(payload, handle, indent=2, sort_keys=True)
                        handle.write("\n")
                        handle.flush()
                        os.fsync(handle.fileno())
                staged[final_name] = temp_path

            manifest = {
                **identity,
                "vector_shape": list(index.vectors.shape),
                "offset_shape": list(index.offsets.shape),
                "checksums": {
                    name: cls._sha256_file(path) for name, path in staged.items()
                },
            }
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{MANIFEST_FILE}.", suffix=".tmp", dir=directory
            )
            manifest_temp = Path(temp_name)
            temporary.append(manifest_temp)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(manifest, handle, indent=2, sort_keys=True)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())

            for name, path in staged.items():
                os.replace(path, directory / name)
                temporary.remove(path)
            os.replace(manifest_temp, directory / MANIFEST_FILE)
            temporary.remove(manifest_temp)
        finally:
            for path in temporary:
                with suppress(OSError):
                    path.unlink()

    @staticmethod
    @contextmanager
    def _interprocess_lock(path: Path) -> Iterator[None]:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as handle:
            if fcntl is not None:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if fcntl is not None:
                    with suppress(OSError):
                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    @classmethod
    def _build_or_load_source(
        cls,
        source_id: str,
        units: list[VisualDocumentUnit],
        model: Any,
        processor: Any,
        device: str,
        encoder_fingerprint: str,
    ) -> ColQwenSourceIndex | None:
        directory = paper_dir(source_id)
        identity = cls._manifest_identity(source_id, units, encoder_fingerprint)
        lock_key = str(directory.resolve())
        with _LOCKS_GUARD:
            thread_lock = _INDEX_LOCKS.setdefault(lock_key, threading.Lock())
        with thread_lock, cls._interprocess_lock(directory / ".colqwen_page_embeddings.lock"):
            cached = cls._load_index(directory, identity)
            if cached is not None:
                return cached
            paths = [directory.joinpath(*Path(unit.image_relpath).parts) for unit in units]
            if not all(path.is_file() for path in paths):
                return None
            page_vectors, page_metadata = cls._encode_pages(
                paths, model, processor, device
            )
            if len(page_vectors) != len(units) or not page_vectors:
                return None
            dimension = page_vectors[0].shape[1]
            if any(vector.ndim != 2 or vector.shape[1] != dimension for vector in page_vectors):
                return None
            offsets = np.zeros(len(page_vectors) + 1, dtype=np.int64)
            offsets[1:] = np.cumsum([len(vector) for vector in page_vectors], dtype=np.int64)
            vectors = np.concatenate(page_vectors, axis=0).astype(np.float16, copy=False)
            index = ColQwenSourceIndex(vectors, offsets, page_metadata)
            try:
                cls._publish_index(directory, identity, index)
            except Exception as exc:
                logger.warning("Could not persist ColQwen2 page index for [%s]: %s", source_id, exc)
            return index

    @staticmethod
    def _image_vector_positions(metadata: dict[str, Any]) -> list[int]:
        explicit = metadata.get("image_token_positions")
        if isinstance(explicit, list):
            return [int(value) for value in explicit]
        offset = metadata.get("image_token_offset")
        count = int(metadata.get("image_token_count") or 0)
        if type(offset) is int and count > 0:
            return list(range(offset, offset + count))
        return []

    @classmethod
    def _candidate_regions(
        cls,
        query_vectors: np.ndarray,
        page_vectors: np.ndarray,
        metadata: dict[str, Any],
        max_regions: int = 3,
    ) -> list[dict[str, Any]]:
        image_positions = cls._image_vector_positions(metadata)
        if not image_positions:
            return []
        similarities = np.matmul(
            query_vectors,
            page_vectors.astype(np.float32, copy=False).T,
        )
        best_positions = np.argmax(similarities, axis=1)
        best_scores = np.max(similarities, axis=1)
        position_to_patch = {
            position: patch for patch, position in enumerate(image_positions)
        }
        selected: dict[int, float] = {}
        for query_index in np.argsort(-best_scores):
            position = int(best_positions[query_index])
            patch = position_to_patch.get(position)
            if patch is not None:
                selected[patch] = max(selected.get(patch, -math.inf), float(best_scores[query_index]))
            if len(selected) >= 16:
                break
        if not selected:
            return []

        height = max(int(metadata.get("grid_height") or 1), 1)
        width = max(int(metadata.get("grid_width") or 1), 1)
        remaining = set(selected)
        components: list[set[int]] = []
        while remaining:
            seed = remaining.pop()
            component = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                row, column = divmod(current, width)
                neighbors: set[int] = set()
                for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    neighbor_row = row + row_delta
                    neighbor_column = column + column_delta
                    if 0 <= neighbor_row < height and 0 <= neighbor_column < width:
                        neighbors.add(neighbor_row * width + neighbor_column)
                for neighbor in list(neighbors & remaining):
                    remaining.remove(neighbor)
                    component.add(neighbor)
                    frontier.append(neighbor)
            components.append(component)

        ranked = sorted(
            components,
            key=lambda component: (-max(selected[item] for item in component), -len(component)),
        )
        regions: list[dict[str, Any]] = []
        for component in ranked[:max_regions]:
            rows = [patch // width for patch in component]
            columns = [patch % width for patch in component]
            x0 = max(min(columns) - 1, 0) / width
            y0 = max(min(rows) - 1, 0) / height
            x1 = min(max(columns) + 2, width) / width
            y1 = min(max(rows) + 2, height) / height
            regions.append({
                "bbox_norm": [round(x0, 6), round(y0, 6), round(x1, 6), round(y1, 6)],
                "score": round(max(selected[item] for item in component), 6),
                "patch_count": len(component),
                "method": "query-token-maxsim-patch-cluster-v1",
            })
        return regions

    @classmethod
    def prebuild_source(cls, source_id: str) -> VisualPageSearchStatus:
        """Ensure one source index exists without paying query-encoding cost."""
        started = time.perf_counter()
        minimum = cls.minimum_score()
        bundle = cls._bundle()
        if bundle is None:
            status = cls.status()
            return VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=False,
                sources_considered=[source_id],
                minimum_score=minimum,
                resolved_model=status.get("requested_model"),
                failure_reason=str(
                    status.get("fallback_reason") or "ColQwen2 unavailable"
                ),
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        model, processor, device, fingerprint = bundle
        units = cls._page_units(source_id)
        if not units:
            return VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=True,
                encoder_fingerprint=fingerprint,
                sources_considered=[source_id],
                minimum_score=minimum,
                resolved_model=cls._requested_model,
                device=device,
                failure_reason="source has no valid full-page visual units",
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        index = cls._build_or_load_source(
            source_id, units, model, processor, device, fingerprint
        )
        if index is None:
            return VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=True,
                encoder_fingerprint=fingerprint,
                sources_considered=[source_id],
                minimum_score=minimum,
                resolved_model=cls._requested_model,
                device=device,
                failure_reason="page index build failed",
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            )

        index_bytes = 0
        for name in (VECTORS_FILE, OFFSETS_FILE, PAGE_METADATA_FILE, MANIFEST_FILE):
            path = paper_dir(source_id) / name
            if path.is_file():
                with suppress(OSError):
                    index_bytes += path.stat().st_size
        return VisualPageSearchStatus(
            attempted=True,
            succeeded=True,
            backend=BACKEND_NAME,
            requested_backend="colqwen2",
            model_loaded=True,
            encoder_fingerprint=fingerprint,
            sources_considered=[source_id],
            indexed_pages=len(units),
            index_bytes=index_bytes,
            minimum_score=minimum,
            resolved_model=cls._requested_model,
            device=device,
            query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
        )

    @classmethod
    def search(
        cls,
        query: str,
        source_ids: list[str],
        top_k: int = 12,
    ) -> VisualPageSearchResult:
        started = time.perf_counter()
        # Preserve the caller's deterministic order. Offline prebuilds use this
        # to commit small sources first, while final hit ordering is handled
        # independently below.
        unique_sources = list(dict.fromkeys(
            source_id for source_id in source_ids if source_id
        ))
        minimum = cls.minimum_score()
        if not query.strip() or top_k <= 0 or not unique_sources:
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=False,
                succeeded=None,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                sources_considered=unique_sources,
                minimum_score=minimum,
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ))

        bundle = cls._bundle()
        if bundle is None:
            status = cls.status()
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=False,
                sources_considered=unique_sources,
                minimum_score=minimum,
                resolved_model=status.get("requested_model"),
                failure_reason=str(status.get("fallback_reason") or "ColQwen2 unavailable"),
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ))
        model, processor, device, fingerprint = bundle
        try:
            query_vectors = cls._encode_query(query, model, processor, device)
        except Exception as exc:
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                backend=BACKEND_NAME,
                requested_backend="colqwen2",
                model_loaded=True,
                encoder_fingerprint=fingerprint,
                sources_considered=unique_sources,
                minimum_score=minimum,
                resolved_model=cls._requested_model,
                device=device,
                failure_reason=f"query encoding failed: {type(exc).__name__}: {exc}",
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ))

        scored: list[tuple[dict[str, Any], float]] = []
        indexed_pages = 0
        index_bytes = 0
        failed_sources: list[str] = []
        for source_id in unique_sources:
            units = cls._page_units(source_id)
            if not units:
                continue
            index = cls._build_or_load_source(
                source_id, units, model, processor, device, fingerprint
            )
            if index is None:
                failed_sources.append(source_id)
                continue
            indexed_pages += len(units)
            for name in (VECTORS_FILE, OFFSETS_FILE, PAGE_METADATA_FILE, MANIFEST_FILE):
                path = paper_dir(source_id) / name
                if path.is_file():
                    with suppress(OSError):
                        index_bytes += path.stat().st_size
            pages_path = paper_dir(source_id) / "pages.json"
            pages = read_json(pages_path) if pages_path.is_file() else []
            text_by_page = {
                int(item.get("page") or 0): str(item.get("text") or "")
                for item in pages if isinstance(item, dict)
            }
            for page_index, unit in enumerate(units):
                page_vectors = index.page_vectors(page_index)
                similarities = np.matmul(
                    query_vectors,
                    page_vectors.astype(np.float32, copy=False).T,
                )
                score = float(np.max(similarities, axis=1).sum())
                if not math.isfinite(score) or score < minimum:
                    continue
                candidate_regions = cls._candidate_regions(
                    query_vectors,
                    page_vectors,
                    index.page_metadata[page_index],
                )
                page_text = text_by_page.get(unit.page, "")
                chunk = {
                    "chunk_id": f"visual_page_{unit.page:04d}",
                    "evidence_id": unit.visual_id,
                    "document_id": source_id,
                    "source_paper_id": source_id,
                    "page": unit.page,
                    "section": "Full-page visual",
                    "section_title": "Full-page visual",
                    "section_path": ["Full-page visual"],
                    "modality": "visual",
                    "chunk_type": "page_visual",
                    "text": page_text[:6000],
                    "paragraph_text": page_text[:500],
                    "is_figure_chunk": True,
                    "is_table_chunk": False,
                    "is_page_visual_chunk": True,
                    "figure_id": unit.visual_id,
                    "figure_type": "page",
                    "image_file": Path(unit.image_relpath).name,
                    "image_relpath": unit.image_relpath,
                    "image_sha256": unit.image_sha256,
                    "label": unit.label,
                    "caption": "",
                    "bbox_norm": unit.bbox_norm,
                    "candidate_regions": candidate_regions,
                    "visual_retrieval_backend": BACKEND_NAME,
                    "visual_retrieval_model": cls._requested_model,
                }
                scored.append((chunk, score))

        scored.sort(key=lambda item: (-item[1], item[0]["source_paper_id"], item[0]["page"]))
        hits = scored[:top_k]
        failure = (
            "page index failed for sources: " + ", ".join(failed_sources)
            if failed_sources else None
        )
        return VisualPageSearchResult(hits, VisualPageSearchStatus(
            attempted=indexed_pages > 0 or bool(failed_sources),
            succeeded=False if failed_sources else (True if indexed_pages > 0 else None),
            backend=BACKEND_NAME,
            requested_backend="colqwen2",
            model_loaded=True,
            encoder_fingerprint=fingerprint,
            sources_considered=unique_sources,
            indexed_pages=indexed_pages,
            hit_count=len(hits),
            best_score=scored[0][1] if scored else None,
            minimum_score=minimum,
            threshold_calibrated=False,
            failure_reason=failure,
            resolved_model=cls._requested_model,
            device=device,
            index_bytes=index_bytes,
            query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
        ))
