"""Cache-only patch-level late-interaction retrieval over complete PDF pages."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
import threading
import time
from contextlib import contextmanager, suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[assignment]

from backend.schemas.visual_document import VisualDocumentUnit, VisualUnitType
from backend.services.pdf_service import paper_dir, read_json
from backend.services.visual_embedding_service import VisualEmbeddingService


logger = logging.getLogger("scholar.visual_pages")

INDEX_FILE = "visual_page_embeddings.npy"
MANIFEST_FILE = "visual_page_embeddings_manifest.json"
INDEX_SCHEMA_VERSION = 1
ALGORITHM_VERSION = "clip-tiled-patch-maxsim-v1"
DEFAULT_MIN_SCORE = 0.12
_LOCKS_GUARD = threading.Lock()
_INDEX_LOCKS: dict[str, threading.Lock] = {}


@dataclass(frozen=True)
class VisualPageSearchStatus:
    attempted: bool
    succeeded: bool | None
    backend: str = ALGORITHM_VERSION
    model_loaded: bool = False
    encoder_fingerprint: str | None = None
    sources_considered: list[str] = field(default_factory=list)
    indexed_pages: int = 0
    hit_count: int = 0
    best_score: float | None = None
    minimum_score: float = DEFAULT_MIN_SCORE
    threshold_calibrated: bool = False
    failure_reason: str | None = None
    requested_backend: str = "clip"
    resolved_model: str | None = None
    device: str | None = None
    index_bytes: int = 0
    query_latency_ms: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "succeeded": self.succeeded,
            "backend": self.backend,
            "model_loaded": self.model_loaded,
            "encoder_fingerprint": self.encoder_fingerprint,
            "sources_considered": list(self.sources_considered),
            "indexed_pages": self.indexed_pages,
            "hit_count": self.hit_count,
            "best_score": self.best_score,
            "minimum_score": self.minimum_score,
            "threshold_calibrated": self.threshold_calibrated,
            "failure_reason": self.failure_reason,
            "requested_backend": self.requested_backend,
            "resolved_model": self.resolved_model,
            "device": self.device,
            "index_bytes": self.index_bytes,
            "query_latency_ms": self.query_latency_ms,
        }


@dataclass(frozen=True)
class VisualPageSearchResult:
    hits: list[tuple[dict[str, Any], float]]
    status: VisualPageSearchStatus


class VisualPageRetrievalService:
    """Build immutable page indexes and execute token-to-patch MaxSim search."""

    @staticmethod
    def minimum_score() -> float:
        raw = os.getenv("SCHOLAR_VISUAL_PAGE_MIN_SIMILARITY", str(DEFAULT_MIN_SCORE))
        try:
            return max(-1.0, min(float(raw), 1.0))
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
    def _page_units(cls, source_id: str) -> list[VisualDocumentUnit]:
        path = paper_dir(source_id) / "visual_units.json"
        if not path.is_file():
            return []
        try:
            raw = read_json(path)
            units = [VisualDocumentUnit.model_validate(item) for item in raw]
        except Exception as exc:
            logger.warning("Invalid visual units for [%s]: %s", source_id, exc)
            return []
        return sorted(
            (unit for unit in units if unit.unit_type == VisualUnitType.PAGE),
            key=lambda unit: unit.page,
        )

    @staticmethod
    def _tile_image(image: Any) -> list[Any]:
        """Return a context view plus six overlapping reading-order page tiles."""
        width, height = image.size
        views = [image]
        x_ranges = ((0.0, 0.58), (0.42, 1.0))
        y_ranges = ((0.0, 0.42), (0.29, 0.71), (0.58, 1.0))
        for y0, y1 in y_ranges:
            for x0, x1 in x_ranges:
                views.append(image.crop((
                    int(width * x0),
                    int(height * y0),
                    max(int(width * x1), 1),
                    max(int(height * y1), 1),
                )))
        return views

    @classmethod
    def _encode_pages(
        cls,
        image_paths: list[Path],
        model: Any,
        processor: Any,
        device: str,
    ) -> np.ndarray:
        import torch
        from PIL import Image

        encoded_pages: list[np.ndarray] = []
        for image_path in image_paths:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                views = cls._tile_image(image)
            inputs = processor(images=views, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(device)
            with torch.no_grad():
                output = model.vision_model(pixel_values=pixel_values)
                patch_tokens = output.last_hidden_state[:, 1:, :]
                projected = model.visual_projection(patch_tokens)
                projected = torch.nn.functional.normalize(projected, p=2, dim=-1)
            flattened = projected.reshape(-1, projected.shape[-1])
            encoded_pages.append(flattened.detach().cpu().numpy().astype(np.float16))
        if not encoded_pages:
            return np.empty((0, 0, 0), dtype=np.float16)
        return np.stack(encoded_pages, axis=0)

    @staticmethod
    def _encode_query(query: str, model: Any, processor: Any, device: str) -> np.ndarray:
        import torch

        inputs = processor(
            text=[query],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77,
        )
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            output = model.text_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            projected = model.text_projection(output.last_hidden_state)
            projected = torch.nn.functional.normalize(projected, p=2, dim=-1)

        mask = attention_mask[0].bool()
        positions = torch.nonzero(mask, as_tuple=False).flatten()
        # CLIP adds start/end tokens. Excluding them makes MaxSim attributable to
        # the user's semantic query tokens rather than prompt framing tokens.
        if len(positions) > 2:
            positions = positions[1:-1]
        tokens = projected[0, positions]
        if tokens.numel() == 0:
            tokens = projected[0, mask]
        return tokens.detach().cpu().numpy().astype(np.float32)

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
            }
            for unit in units
        ]
        return {
            "schema_version": INDEX_SCHEMA_VERSION,
            "algorithm_version": ALGORITHM_VERSION,
            "source_paper_id": source_id,
            "encoder_fingerprint": encoder_fingerprint,
            "rows": rows,
            "rows_sha256": hashlib.sha256(
                json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
            ).hexdigest(),
        }

    @classmethod
    def _load_index(
        cls,
        index_path: Path,
        manifest_path: Path,
        identity: dict[str, Any],
    ) -> np.ndarray | None:
        if not index_path.is_file() or not manifest_path.is_file():
            return None
        try:
            manifest = read_json(manifest_path)
            for key, value in identity.items():
                if manifest.get(key) != value:
                    return None
            if manifest.get("vector_sha256") != cls._sha256_file(index_path):
                return None
            vectors = np.load(index_path, allow_pickle=False)
            expected_shape = manifest.get("vector_shape")
            if (
                vectors.dtype != np.float16
                or vectors.ndim != 3
                or vectors.shape[0] != len(identity["rows"])
                or list(vectors.shape) != expected_shape
                or not np.isfinite(vectors).all()
            ):
                return None
            return vectors
        except Exception as exc:
            logger.warning("Invalid page visual cache [%s]: %s", index_path, exc)
            return None

    @classmethod
    def _publish_index(
        cls,
        index_path: Path,
        manifest_path: Path,
        identity: dict[str, Any],
        vectors: np.ndarray,
    ) -> None:
        index_path.parent.mkdir(parents=True, exist_ok=True)
        descriptor: int | None = None
        temp_name: str | None = None
        temp_manifest: Path | None = None
        try:
            descriptor, temp_name = tempfile.mkstemp(
                prefix=f".{index_path.name}.", suffix=".tmp", dir=index_path.parent
            )
            with os.fdopen(descriptor, "wb") as handle:
                descriptor = None
                np.save(handle, vectors, allow_pickle=False)
                handle.flush()
                os.fsync(handle.fileno())
            temp_path = Path(temp_name)
            manifest = {
                **identity,
                "vector_shape": list(vectors.shape),
                "vector_dtype": "float16",
                "vector_sha256": cls._sha256_file(temp_path),
            }
            temp_manifest = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
            temp_manifest.write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            with temp_manifest.open("rb") as handle:
                os.fsync(handle.fileno())
            os.replace(temp_path, index_path)
            temp_name = None
            os.replace(temp_manifest, manifest_path)
            temp_manifest = None
        finally:
            if descriptor is not None:
                os.close(descriptor)
            if temp_name:
                with suppress(OSError):
                    Path(temp_name).unlink()
            if temp_manifest is not None:
                with suppress(OSError):
                    temp_manifest.unlink()

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
    ) -> np.ndarray | None:
        directory = paper_dir(source_id)
        index_path = directory / INDEX_FILE
        manifest_path = directory / MANIFEST_FILE
        identity = cls._manifest_identity(source_id, units, encoder_fingerprint)
        lock_key = str(directory.resolve())
        with _LOCKS_GUARD:
            thread_lock = _INDEX_LOCKS.setdefault(lock_key, threading.Lock())
        with thread_lock, cls._interprocess_lock(directory / ".visual_page_embeddings.lock"):
            cached = cls._load_index(index_path, manifest_path, identity)
            if cached is not None:
                return cached
            paths = [directory.joinpath(*Path(unit.image_relpath).parts) for unit in units]
            if not all(path.is_file() for path in paths):
                return None
            vectors = cls._encode_pages(paths, model, processor, device)
            if vectors.ndim != 3 or vectors.shape[0] != len(units):
                return None
            try:
                cls._publish_index(index_path, manifest_path, identity, vectors)
            except Exception as exc:
                logger.warning("Could not persist page visual index for [%s]: %s", source_id, exc)
            return vectors

    @classmethod
    def search(
        cls,
        query: str,
        source_ids: list[str],
        top_k: int = 12,
    ) -> VisualPageSearchResult:
        started = time.perf_counter()
        unique_sources = sorted({source_id for source_id in source_ids if source_id})
        minimum = cls.minimum_score()
        if not query.strip() or top_k <= 0 or not unique_sources:
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=False,
                succeeded=None,
                sources_considered=unique_sources,
                minimum_score=minimum,
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ))

        bundle = VisualEmbeddingService.encoder_bundle()
        if bundle is None:
            status = VisualEmbeddingService.status()
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                model_loaded=False,
                sources_considered=unique_sources,
                minimum_score=minimum,
                failure_reason=str(status.get("fallback_reason") or "visual encoder unavailable"),
                resolved_model=status.get("requested_model"),
                device=status.get("device"),
                query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
            ))
        model, processor, device, fingerprint = bundle
        try:
            query_tokens = cls._encode_query(query, model, processor, device)
        except Exception as exc:
            return VisualPageSearchResult([], VisualPageSearchStatus(
                attempted=True,
                succeeded=False,
                model_loaded=True,
                encoder_fingerprint=fingerprint,
                sources_considered=unique_sources,
                minimum_score=minimum,
                failure_reason=f"query encoding failed: {type(exc).__name__}: {exc}",
                resolved_model=VisualEmbeddingService.status().get("requested_model"),
                device=device,
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
            vectors = cls._build_or_load_source(
                source_id, units, model, processor, device, fingerprint
            )
            if vectors is None:
                failed_sources.append(source_id)
                continue
            indexed_pages += len(units)
            index_path = paper_dir(source_id) / INDEX_FILE
            if index_path.is_file():
                with suppress(OSError):
                    index_bytes += index_path.stat().st_size
            pages_path = paper_dir(source_id) / "pages.json"
            pages = read_json(pages_path) if pages_path.is_file() else []
            text_by_page = {
                int(item.get("page") or 0): str(item.get("text") or "")
                for item in pages if isinstance(item, dict)
            }
            for unit, page_vectors in zip(units, vectors):
                similarities = np.matmul(
                    query_tokens,
                    page_vectors.astype(np.float32, copy=False).T,
                )
                score = float(np.max(similarities, axis=1).mean())
                if not np.isfinite(score):
                    continue
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
                    "text": text_by_page.get(unit.page, "")[:6000],
                    "paragraph_text": text_by_page.get(unit.page, "")[:500],
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
                }
                scored.append((chunk, score))

        scored.sort(key=lambda item: (-item[1], item[0]["source_paper_id"], item[0]["page"]))
        hits = [item for item in scored if item[1] >= minimum][:top_k]
        failure = (
            "page index failed for sources: " + ", ".join(failed_sources)
            if failed_sources else None
        )
        return VisualPageSearchResult(hits, VisualPageSearchStatus(
            attempted=indexed_pages > 0 or bool(failed_sources),
            succeeded=False if failed_sources else (True if indexed_pages > 0 else None),
            model_loaded=True,
            encoder_fingerprint=fingerprint,
            sources_considered=unique_sources,
            indexed_pages=indexed_pages,
            hit_count=len(hits),
            best_score=scored[0][1] if scored else None,
            minimum_score=minimum,
            failure_reason=failure,
            resolved_model=VisualEmbeddingService.status().get("requested_model"),
            device=device,
            index_bytes=index_bytes,
            query_latency_ms=round((time.perf_counter() - started) * 1000, 3),
        ))
