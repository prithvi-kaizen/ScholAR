"""Release manifest creation and lifecycle updates."""

from __future__ import annotations

import os
import platform
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psutil

from evaluation.release.io import read_json, sha256_file, write_checksums, write_json
from evaluation.release.schemas import ArtifactHash, RawReleaseRow, ReleaseConfig, ReleaseManifest


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def git_identity(root: Path) -> tuple[str | None, bool | None]:
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"], cwd=root, check=True, capture_output=True, text=True
        ).stdout.strip())
        return revision or None, dirty
    except (OSError, subprocess.CalledProcessError):
        return None, None


def runtime_identity() -> tuple[dict[str, Any], dict[str, Any]]:
    hardware = {
        "os": platform.system(),
        "os_release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or None,
        "logical_cpu_count": os.cpu_count(),
        "memory_bytes": int(psutil.virtual_memory().total),
    }
    software = {
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }
    return hardware, software


def artifact_records(release_dir: Path) -> list[ArtifactHash]:
    names = (
        "expected_keys.json",
        "configs/release_config.json",
        "raw/rows.jsonl",
        "scored/rows.jsonl",
        "aggregates/summary.json",
        "tables/summary.csv",
        "tables/summary.tex",
        "tables/provenance.json",
    )
    records: list[ArtifactHash] = []
    for name in names:
        path = release_dir / name
        if path.is_file():
            records.append(ArtifactHash(path=name, sha256=sha256_file(path), bytes=path.stat().st_size))
    return records


def create_manifest(config: ReleaseConfig, release_dir: Path, n_expected: int, root: Path) -> ReleaseManifest:
    revision, dirty = git_identity(root)
    hardware, software = runtime_identity()
    now = utc_now()
    return ReleaseManifest(
        release_id=config.release_id,
        run_id=config.run_id,
        evidence_class=config.dataset.evidence_class,
        claim_status=config.dataset.claim_status,
        lifecycle_status="GENERATING",
        git_revision=revision,
        git_dirty=dirty,
        exact_command=config.command or [sys.executable, "evaluation/run_release_suite.py", "--config", "<config>"],
        dataset=config.dataset.model_dump(mode="json"),
        systems=[item.model_dump(mode="json") for item in config.systems],
        models=[item.model_dump(mode="json") for item in config.models],
        prompt_hashes=config.prompt_hashes,
        seeds=config.seeds,
        hardware=hardware,
        software=software,
        started_at=now,
        updated_at=now,
        n_expected=n_expected,
        status_counts={},
        artifacts=artifact_records(release_dir),
    )


def load_manifest(path: Path) -> ReleaseManifest:
    return ReleaseManifest.model_validate(read_json(path))


def update_manifest(
    manifest: ReleaseManifest,
    release_dir: Path,
    *,
    raw_rows: list[RawReleaseRow] | None = None,
    lifecycle_status: str | None = None,
    completed: bool = False,
) -> ReleaseManifest:
    if lifecycle_status is not None:
        manifest.lifecycle_status = lifecycle_status  # type: ignore[assignment]
    if raw_rows is not None:
        manifest.status_counts = dict(sorted(Counter(row.status.value for row in raw_rows).items()))
    manifest.updated_at = utc_now()
    if completed:
        manifest.completed_at = manifest.updated_at
    manifest.artifacts = artifact_records(release_dir)
    write_json(release_dir / "manifest.json", manifest)
    write_checksums(release_dir)
    return manifest
