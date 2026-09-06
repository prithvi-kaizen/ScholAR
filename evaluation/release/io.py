"""Canonical JSON/JSONL, hashing, path, and expected-key helpers."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Iterable, TypeVar

from pydantic import BaseModel

from evaluation.release.schemas import (
    CanonicalKey,
    CaseRecord,
    ExpectedKeySet,
    ReleaseConfig,
)


ROOT = Path(__file__).resolve().parents[2]
T = TypeVar("T", bound=BaseModel)


def canonical_json_bytes(value: Any) -> bytes:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    text = json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
    return (text + "\n").encode("utf-8")


def canonical_jsonl_bytes(values: Iterable[Any]) -> bytes:
    lines: list[str] = []
    for value in values:
        if isinstance(value, BaseModel):
            value = value.model_dump(mode="json")
        lines.append(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))
    return (("\n".join(lines) + "\n") if lines else "").encode("utf-8")


def atomic_write(path: Path, payload: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_json(path: Path, value: Any) -> None:
    atomic_write(path, canonical_json_bytes(value))


def write_jsonl(path: Path, values: Iterable[Any]) -> None:
    atomic_write(path, canonical_jsonl_bytes(values))


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path, model: type[T]) -> list[T]:
    raw = Path(path).read_bytes()
    if raw and not raw.endswith(b"\n"):
        raise ValueError(f"truncated JSONL file (missing final newline): {path}")
    rows: list[T] = []
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL row at {path}:{line_number}")
        try:
            rows.append(model.model_validate_json(line))
        except Exception as exc:
            raise ValueError(f"invalid JSONL row at {path}:{line_number}: {exc}") from exc
    return rows


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def resolve_repo_path(value: str) -> Path:
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError as exc:
        raise ValueError(f"release path escapes the repository: {value!r}") from exc
    return candidate


def load_config(path: Path) -> ReleaseConfig:
    return ReleaseConfig.model_validate(read_json(path))


def load_cases(config: ReleaseConfig) -> list[CaseRecord]:
    path = resolve_repo_path(config.dataset.cases_path)
    if not path.is_file():
        raise ValueError(f"frozen case file is missing: {config.dataset.cases_path}")
    actual_hash = sha256_file(path)
    if config.dataset.sha256 != actual_hash:
        raise ValueError(
            f"dataset hash mismatch for {config.dataset.cases_path}: expected {config.dataset.sha256}, got {actual_hash}"
        )
    payload = read_json(path)
    if not isinstance(payload, list):
        raise ValueError("release case file must contain a JSON array")
    cases = [CaseRecord.model_validate(item) for item in payload]
    identities = [item.case_id for item in cases]
    if len(identities) != len(set(identities)):
        raise ValueError("release case IDs must be unique")
    return sorted(cases, key=lambda item: item.case_id)


def build_expected_keys(config: ReleaseConfig, cases: list[CaseRecord]) -> ExpectedKeySet:
    keys = sorted(
        (
            CanonicalKey(system=system.name, model=model.tag, seed=seed, case_id=case.case_id)
            for system in config.systems
            for model in config.models
            for seed in config.seeds
            for case in cases
        ),
        key=lambda item: item.as_tuple(),
    )
    return ExpectedKeySet(
        release_id=config.release_id,
        run_id=config.run_id,
        dataset_sha256=config.dataset.sha256 or "",
        n_expected=len(keys),
        keys=keys,
    )


def key_index(rows: Iterable[Any]) -> dict[tuple[str, str, int, str], Any]:
    indexed: dict[tuple[str, str, int, str], Any] = {}
    for row in rows:
        key = row.key.as_tuple()
        if key in indexed:
            raise ValueError(f"duplicate release row key: {row.key.as_string()}")
        indexed[key] = row
    return indexed


def write_checksums(release_dir: Path) -> Path:
    release_dir = Path(release_dir)
    checksum_path = release_dir / "checksums.sha256"
    paths = sorted(
        path for path in release_dir.rglob("*")
        if path.is_file()
        and path != checksum_path
        and path.name not in {".gitkeep", ".DS_Store"}
        and path.suffix != ".pyc"
        and "__pycache__" not in path.parts
    )
    lines = [f"{sha256_file(path)}  {path.relative_to(release_dir).as_posix()}" for path in paths]
    atomic_write(checksum_path, (("\n".join(lines) + "\n") if lines else "").encode("utf-8"))
    return checksum_path


def read_checksums(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line:
            raise ValueError(f"blank checksum line at {line_number}")
        try:
            digest, name = line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"malformed checksum line at {line_number}") from exc
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError(f"invalid SHA-256 at checksum line {line_number}")
        if name in checksums:
            raise ValueError(f"duplicate checksum path: {name}")
        checksums[name] = digest
    return checksums
