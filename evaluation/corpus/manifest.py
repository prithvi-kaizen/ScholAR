"""Deterministic, fail-closed identities for an experimental paper corpus."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.services.paper_finalize_service import PaperFinalizeService
from backend.services.pdf_service import PAPERS_DIR, read_json, safe_paper_id, write_json


CORPUS_SCHEMA_VERSION = "1.0"
ROOT = Path(__file__).resolve().parents[2]
DERIVED_INDEX_MANIFESTS = (
    "embeddings_manifest.json",
    "visual_embeddings_manifest.json",
    "visual_page_embeddings_manifest.json",
    "colqwen_page_manifest.json",
)
COLQWEN_COMPANION_FILES = (
    "colqwen_page_metadata.json",
    "colqwen_page_offsets.npy",
    "colqwen_page_vectors.npy",
)


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusSelection(StrictModel):
    """Paper-disjoint split selected before measured annotation and evaluation."""

    schema_version: Literal["1.0"] = CORPUS_SCHEMA_VERSION
    selection_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    development_paper_ids: list[str] = Field(min_length=1)
    test_paper_ids: list[str] = Field(min_length=1)
    source_evaluation_files: list[str] = Field(default_factory=list)
    selection_rule: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_splits(self) -> "CorpusSelection":
        development = self.development_paper_ids
        test = self.test_paper_ids
        if development != sorted(development) or test != sorted(test):
            raise ValueError("paper IDs in each split must use canonical sorted order")
        if len(development) != len(set(development)):
            raise ValueError("development split contains duplicate paper IDs")
        if len(test) != len(set(test)):
            raise ValueError("test split contains duplicate paper IDs")
        overlap = sorted(set(development) & set(test))
        if overlap:
            raise ValueError(f"development and test papers overlap: {overlap}")
        unsafe = [
            paper_id
            for paper_id in development + test
            if safe_paper_id(paper_id) != paper_id
        ]
        if unsafe:
            raise ValueError(f"selection contains unsafe paper IDs: {unsafe}")
        if self.source_evaluation_files != sorted(set(self.source_evaluation_files)):
            raise ValueError("source evaluation files must be unique and sorted")
        return self

    def split_for(self, paper_id: str) -> Literal["development", "test"]:
        if paper_id in self.development_paper_ids:
            return "development"
        if paper_id in self.test_paper_ids:
            return "test"
        raise KeyError(paper_id)

    @property
    def all_paper_ids(self) -> list[str]:
        return sorted(self.development_paper_ids + self.test_paper_ids)


class FileArtifact(StrictModel):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    bytes: int = Field(ge=0)


class CorpusPaperRecord(StrictModel):
    paper_id: str
    split: Literal["development", "test"]
    title: str = ""
    generation_id: str
    ingestion_schema_version: str
    parser_engine: str
    degraded_mode: bool
    pdf_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    chunks_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    page_count: int = Field(ge=1)
    chunk_count: int = Field(ge=1)
    figure_count: int = Field(ge=0)
    visual_unit_count: int = Field(ge=1)
    source_artifacts: list[FileArtifact] = Field(min_length=1)
    visual_artifacts: list[FileArtifact] = Field(min_length=1)
    derived_index_manifests: list[FileArtifact] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_artifact_order(self) -> "CorpusPaperRecord":
        for label, artifacts in (
            ("source", self.source_artifacts),
            ("visual", self.visual_artifacts),
            ("derived-index", self.derived_index_manifests),
        ):
            paths = [item.path for item in artifacts]
            if paths != sorted(paths) or len(paths) != len(set(paths)):
                raise ValueError(f"{label} artifact paths must be unique and sorted")
        return self


class CorpusManifest(StrictModel):
    schema_version: Literal["1.0"] = CORPUS_SCHEMA_VERSION
    corpus_id: str = Field(pattern=r"^[a-z0-9][a-z0-9_.-]*$")
    selection_id: str
    selection_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    papers_root: str = "backend/data/papers"
    paper_disjoint: Literal[True] = True
    development_paper_count: int = Field(ge=1)
    test_paper_count: int = Field(ge=1)
    paper_count: int = Field(ge=2)
    total_pages: int = Field(ge=1)
    total_chunks: int = Field(ge=1)
    total_visual_units: int = Field(ge=1)
    required_index_manifests: list[str] = Field(default_factory=list)
    selection_source_artifacts: list[FileArtifact] = Field(default_factory=list)
    papers: list[CorpusPaperRecord] = Field(min_length=2)
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_counts_and_order(self) -> "CorpusManifest":
        paper_ids = [paper.paper_id for paper in self.papers]
        if paper_ids != sorted(paper_ids) or len(paper_ids) != len(set(paper_ids)):
            raise ValueError("corpus paper records must be unique and sorted")
        development = sum(paper.split == "development" for paper in self.papers)
        test = sum(paper.split == "test" for paper in self.papers)
        expected = (
            development,
            test,
            len(self.papers),
            sum(paper.page_count for paper in self.papers),
            sum(paper.chunk_count for paper in self.papers),
            sum(paper.visual_unit_count for paper in self.papers),
        )
        actual = (
            self.development_paper_count,
            self.test_paper_count,
            self.paper_count,
            self.total_pages,
            self.total_chunks,
            self.total_visual_units,
        )
        if actual != expected:
            raise ValueError("corpus aggregate counts differ from paper records")
        if self.required_index_manifests != sorted(set(self.required_index_manifests)):
            raise ValueError("required index manifests must be unique and sorted")
        source_paths = [item.path for item in self.selection_source_artifacts]
        if source_paths != sorted(source_paths) or len(source_paths) != len(set(source_paths)):
            raise ValueError("selection source artifacts must be unique and sorted")
        return self


class CorpusDataCard(StrictModel):
    """Compact disclosure of split composition and parser quality."""

    schema_version: Literal["1.0"] = CORPUS_SCHEMA_VERSION
    corpus_id: str
    corpus_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    selection_id: str
    development_paper_count: int = Field(ge=1)
    test_paper_count: int = Field(ge=1)
    parser_engine_counts: dict[str, int]
    degraded_mode_counts: dict[str, int]
    clean_mode_counts: dict[str, int]
    degraded_paper_ids: list[str]
    clean_paper_ids: list[str]
    intended_use: str
    split_policy: str


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact(path: Path, relative_to: Path) -> FileArtifact:
    resolved_root = relative_to.resolve()
    resolved = path.resolve()
    try:
        relative = resolved.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError(f"artifact escapes paper directory: {path}") from exc
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"artifact is missing or symlinked: {path}")
    return FileArtifact(
        path=relative.as_posix(),
        sha256=sha256_file(path),
        bytes=path.stat().st_size,
    )


def load_selection(path: Path) -> CorpusSelection:
    return CorpusSelection.model_validate(json.loads(Path(path).read_text(encoding="utf-8")))


def selection_sha256(selection: CorpusSelection) -> str:
    return sha256_bytes(canonical_json_bytes(selection.model_dump(mode="json")))


def _referenced_paper_ids(value: Any, *, parent_key: str = "") -> set[str]:
    paper_ids: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            paper_ids.update(_referenced_paper_ids(item, parent_key=str(key)))
    elif isinstance(value, list):
        for item in value:
            paper_ids.update(_referenced_paper_ids(item, parent_key=parent_key))
    elif isinstance(value, str):
        candidate = value.strip()
        if re.fullmatch(r"\d{4}\.\d{4,5}", candidate):
            paper_ids.add(candidate)
        elif parent_key in {
            "paper_id",
            "source_paper_id",
            "secondary_paper_ids",
            "paper_ids",
        }:
            paper_ids.add(candidate)
    return paper_ids


def _selection_source_artifacts(selection: CorpusSelection) -> list[FileArtifact]:
    artifacts: list[FileArtifact] = []
    development = set(selection.development_paper_ids)
    for name in selection.source_evaluation_files:
        path = ROOT.joinpath(*Path(name).parts)
        source_artifact = artifact(path, ROOT)
        value = json.loads(path.read_text(encoding="utf-8"))
        leaked_or_unknown = sorted(_referenced_paper_ids(value) - development)
        if leaked_or_unknown:
            raise ValueError(
                f"selection source {name} references non-development papers: "
                f"{leaked_or_unknown}"
            )
        artifacts.append(source_artifact)
    return sorted(artifacts, key=lambda item: item.path)


def _validate_required_page_index(
    directory: Path,
    paper_id: str,
    manifest_name: str,
    visual_units: list[Any],
) -> None:
    """Verify a required page-index manifest and every bound payload."""
    if manifest_name not in {
        "visual_page_embeddings_manifest.json",
        "colqwen_page_manifest.json",
    }:
        return
    manifest = read_json(directory / manifest_name)
    if not isinstance(manifest, dict):
        raise ValueError(f"invalid required index manifest: {paper_id}/{manifest_name}")
    if manifest.get("source_paper_id") != paper_id:
        raise ValueError(f"required index source identity differs: {paper_id}/{manifest_name}")

    page_units = [
        unit for unit in visual_units
        if isinstance(unit, dict) and unit.get("unit_type") == "page"
    ]
    row_keys = ["visual_id", "page", "image_relpath", "image_sha256"]
    if manifest_name == "colqwen_page_manifest.json":
        row_keys.extend(["width_px", "height_px"])
    expected_rows = [
        {key: unit.get(key) for key in row_keys}
        for unit in page_units
    ]
    if not expected_rows or manifest.get("rows") != expected_rows:
        raise ValueError(f"required index page coverage differs: {paper_id}/{manifest_name}")
    if manifest.get("rows_sha256") != sha256_bytes(canonical_json_bytes(expected_rows)):
        raise ValueError(f"required index row identity differs: {paper_id}/{manifest_name}")

    if manifest_name == "visual_page_embeddings_manifest.json":
        vector_path = directory / "visual_page_embeddings.npy"
        if (
            not vector_path.is_file()
            or manifest.get("vector_sha256") != sha256_file(vector_path)
        ):
            raise ValueError(f"required CLIP vector checksum differs: {paper_id}")
        return

    checksums = manifest.get("checksums")
    if not isinstance(checksums, dict) or set(checksums) != set(COLQWEN_COMPANION_FILES):
        raise ValueError(f"required ColQwen payload set differs: {paper_id}")
    for name in COLQWEN_COMPANION_FILES:
        payload_path = directory / name
        if not payload_path.is_file() or checksums.get(name) != sha256_file(payload_path):
            raise ValueError(f"required ColQwen checksum differs: {paper_id}/{name}")


def _paper_record(
    directory: Path,
    paper_id: str,
    split: Literal["development", "test"],
    required_index_manifests: tuple[str, ...],
) -> CorpusPaperRecord:
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError(f"paper directory is missing or symlinked: {paper_id}")
    if PaperFinalizeService.load_if_complete(paper_id, target_dir=directory) is None:
        raise ValueError(f"paper bundle does not satisfy the current ingestion contract: {paper_id}")

    metadata = read_json(directory / "metadata.json")
    ingestion = read_json(directory / PaperFinalizeService.MANIFEST_NAME)
    visual_units = read_json(directory / "visual_units.json")
    if not isinstance(metadata, dict) or not isinstance(ingestion, dict):
        raise ValueError(f"invalid metadata or ingestion manifest: {paper_id}")
    if not isinstance(visual_units, list):
        raise ValueError(f"invalid visual units: {paper_id}")

    source_artifacts = [
        artifact(directory / name, directory)
        for name in PaperFinalizeService.REQUIRED_FILES
    ]
    source_artifacts.sort(key=lambda item: item.path)

    visual_paths: set[Path] = set()
    for unit in visual_units:
        if not isinstance(unit, dict) or not isinstance(unit.get("image_relpath"), str):
            raise ValueError(f"visual unit lacks an image path: {paper_id}")
        visual_paths.add(directory.joinpath(*Path(unit["image_relpath"]).parts))
    visual_artifacts = sorted(
        (artifact(path, directory) for path in visual_paths),
        key=lambda item: item.path,
    )

    missing_indexes = [
        name for name in required_index_manifests if not (directory / name).is_file()
    ]
    if missing_indexes:
        raise ValueError(
            f"paper {paper_id} lacks required index manifests: {', '.join(missing_indexes)}"
        )
    for name in required_index_manifests:
        _validate_required_page_index(directory, paper_id, name, visual_units)
    derived = sorted(
        (
            artifact(directory / name, directory)
            for name in DERIVED_INDEX_MANIFESTS
            if (directory / name).is_file()
        ),
        key=lambda item: item.path,
    )

    return CorpusPaperRecord(
        paper_id=paper_id,
        split=split,
        title=str(metadata.get("title") or ""),
        generation_id=str(ingestion.get("generation_id") or ""),
        ingestion_schema_version=str(ingestion.get("schema_version") or ""),
        parser_engine=str(metadata.get("parser_engine") or ""),
        degraded_mode=bool(metadata.get("degraded_mode")),
        pdf_sha256=str(ingestion.get("pdf_sha256") or ""),
        chunks_sha256=str(ingestion.get("chunks_sha256") or ""),
        page_count=int(ingestion.get("page_count") or 0),
        chunk_count=int(ingestion.get("chunk_count") or 0),
        figure_count=int(ingestion.get("figure_count") or 0),
        visual_unit_count=int(ingestion.get("visual_unit_count") or 0),
        source_artifacts=source_artifacts,
        visual_artifacts=visual_artifacts,
        derived_index_manifests=derived,
    )


def _manifest_digest(payload: dict[str, Any]) -> str:
    unhashed = dict(payload)
    unhashed.pop("corpus_sha256", None)
    return sha256_bytes(canonical_json_bytes(unhashed))


def build_corpus_manifest(
    selection: CorpusSelection,
    *,
    papers_dir: Path = PAPERS_DIR,
    corpus_id: str | None = None,
    required_index_manifests: tuple[str, ...] = (),
) -> CorpusManifest:
    """Build a deterministic manifest after revalidating every selected bundle."""
    unknown_indexes = sorted(set(required_index_manifests) - set(DERIVED_INDEX_MANIFESTS))
    if unknown_indexes:
        raise ValueError(f"unknown required index manifests: {unknown_indexes}")
    required_indexes = tuple(sorted(set(required_index_manifests)))
    selection_artifacts = _selection_source_artifacts(selection)
    records = [
        _paper_record(
            Path(papers_dir) / paper_id,
            paper_id,
            selection.split_for(paper_id),
            required_indexes,
        )
        for paper_id in selection.all_paper_ids
    ]
    payload: dict[str, Any] = {
        "schema_version": CORPUS_SCHEMA_VERSION,
        "corpus_id": corpus_id or selection.selection_id,
        "selection_id": selection.selection_id,
        "selection_sha256": selection_sha256(selection),
        "papers_root": "backend/data/papers",
        "paper_disjoint": True,
        "development_paper_count": len(selection.development_paper_ids),
        "test_paper_count": len(selection.test_paper_ids),
        "paper_count": len(records),
        "total_pages": sum(record.page_count for record in records),
        "total_chunks": sum(record.chunk_count for record in records),
        "total_visual_units": sum(record.visual_unit_count for record in records),
        "required_index_manifests": list(required_indexes),
        "selection_source_artifacts": [
            item.model_dump(mode="json") for item in selection_artifacts
        ],
        "papers": [record.model_dump(mode="json") for record in records],
    }
    payload["corpus_sha256"] = _manifest_digest(payload)
    return CorpusManifest.model_validate(payload)


def write_corpus_manifest(path: Path, manifest: CorpusManifest) -> None:
    write_json(Path(path), manifest.model_dump(mode="json"))


def build_corpus_data_card(manifest: CorpusManifest) -> CorpusDataCard:
    parser_counts: dict[str, int] = {}
    degraded_counts = {"development": 0, "test": 0}
    clean_counts = {"development": 0, "test": 0}
    degraded_ids: list[str] = []
    clean_ids: list[str] = []
    for paper in manifest.papers:
        parser = paper.parser_engine or "unknown"
        parser_counts[parser] = parser_counts.get(parser, 0) + 1
        if paper.degraded_mode:
            degraded_counts[paper.split] += 1
            degraded_ids.append(paper.paper_id)
        else:
            clean_counts[paper.split] += 1
            clean_ids.append(paper.paper_id)
    return CorpusDataCard(
        corpus_id=manifest.corpus_id,
        corpus_sha256=manifest.corpus_sha256,
        selection_id=manifest.selection_id,
        development_paper_count=manifest.development_paper_count,
        test_paper_count=manifest.test_paper_count,
        parser_engine_counts=dict(sorted(parser_counts.items())),
        degraded_mode_counts=degraded_counts,
        clean_mode_counts=clean_counts,
        degraded_paper_ids=sorted(degraded_ids),
        clean_paper_ids=sorted(clean_ids),
        intended_use=(
            "Development papers support system construction and ablations; test papers "
            "are reserved for held-out measured evaluation."
        ),
        split_policy=(
            "Paper-disjoint. Existing executable evaluation papers form development; "
            "the held-out split is a deterministic predeclared subset of unreferenced "
            "locally available papers."
        ),
    )


def write_corpus_data_card(path: Path, data_card: CorpusDataCard) -> None:
    write_json(Path(path), data_card.model_dump(mode="json"))


def validate_corpus_manifest(
    manifest: CorpusManifest,
    selection: CorpusSelection,
    *,
    papers_dir: Path = PAPERS_DIR,
) -> list[str]:
    """Rebuild from source and report every frozen-identity difference."""
    errors: list[str] = []
    if manifest.selection_id != selection.selection_id:
        errors.append("manifest selection ID differs from the supplied selection")
    if manifest.selection_sha256 != selection_sha256(selection):
        errors.append("manifest selection hash differs from the supplied selection")
    if manifest.corpus_sha256 != _manifest_digest(manifest.model_dump(mode="json")):
        errors.append("manifest corpus hash is invalid")
    try:
        rebuilt = build_corpus_manifest(
            selection,
            papers_dir=papers_dir,
            corpus_id=manifest.corpus_id,
            required_index_manifests=tuple(manifest.required_index_manifests),
        )
    except Exception as exc:
        errors.append(f"source corpus validation failed: {type(exc).__name__}: {exc}")
        return errors
    if rebuilt != manifest:
        errors.append("source corpus differs from the frozen manifest")
    return errors
