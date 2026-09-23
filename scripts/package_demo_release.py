#!/usr/bin/env python3
"""Build and verify the fail-closed EACL 2027 Demonstration installable release package.

Assembles the standalone demo package for ScholAR:
- Single archive root: ScholAR-EACL2027-Demo-v1.0.0/
- Explicit allowlist of runtime, documentation, license, example, and test files
- Rejects secrets, private data, model weights, and absolute user paths
- Deterministic archive timestamps and order for reproducible SHA-256 digests
- Produces DEMO_PACKAGE_MANIFEST.json and a .sha256 sidecar
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Sequence

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_OUTPUT = ROOT / "release" / "ScholAR_EACL2027_Demo_v1.0.0.zip"
ARCHIVE_ROOT = "ScholAR-EACL2027-Demo-v1.0.0"
MANIFEST_NAME = "DEMO_PACKAGE_MANIFEST.json"
FIXED_ZIP_TIME = (2026, 1, 1, 0, 0, 0)
MAX_FILE_BYTES = 25 * 1024 * 1024

TOP_LEVEL_FILES = {
    ".gitignore",
    "DEMO_INSTALL.md",
    "LICENSE",
    "Makefile",
    "README.md",
    "THIRD_PARTY_NOTICES.md",
    "pytest.ini",
    "requirements.txt",
    "run_demo.py",
}

ALLOW_ROOTS = (
    "backend",
    "docs",
    "examples/eacl_demo",
    "frontend",
    "requirements",
    "scripts",
    "tests",
)

BLOCKED_PARTS = {
    ".agents",
    ".git",
    ".idea",
    ".next",
    ".pytest_cache",
    ".tox",
    ".venv",
    ".venv312",
    ".vscode",
    "__pycache__",
    "node_modules",
    "private",
    "scratch",
}

BLOCKED_SUFFIXES = {
    ".aux",
    ".bbl",
    ".blg",
    ".db",
    ".log",
    ".npy",
    ".npz",
    ".out",
    ".pickle",
    ".pkl",
    ".pyc",
    ".pyo",
    ".sqlite",
    ".sqlite3",
    ".synctex",
    ".tsbuildinfo",
}

BLOCKED_PREFIXES = (
    "backend/data",
    "evaluation",
    "paper",
    "release",
)

BLOCKED_EXACT = {
    ".env",
    ".env.local",
    "LICENSE-ANONYMOUS",
    "ANONYMOUS_ARTIFACT.md",
}

SECRET_PATTERNS = (
    ("private key", re.compile(br"-----BEGIN [A-Z ]*PRIVATE KEY-----")),
    ("OpenAI-style secret", re.compile(br"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("GitHub token", re.compile(br"\bgh[pousr]_[A-Za-z0-9]{20,}\b")),
    ("AWS access key", re.compile(br"\bAKIA[A-Z0-9]{16}\b")),
)

USER_PATH_PATTERNS = (
    ("macOS user home", re.compile(rb"/Users/[A-Za-z0-9._-]+/(?!Downloads/ScholAR|Library)")),
    ("Linux user home", re.compile(rb"/home/[A-Za-z0-9._-]+/")),
    ("Windows user home", re.compile(rb"[A-Za-z]:\\Users\\[^\\\s]+\\", re.I)),
)


@dataclass(frozen=True)
class ArtifactFile:
    source: Path
    archive_name: str


class DemoPackagePolicyError(ValueError):
    """Raised when candidate package files violate release policy."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _under_prefix(relative: str, prefix: str) -> bool:
    return relative == prefix or relative.startswith(prefix + "/")


def is_allowed_source(path: Path) -> bool:
    if not path.is_file() or path.is_symlink():
        return False
    relative = path.relative_to(ROOT).as_posix()
    posix = PurePosixPath(relative)

    if relative in BLOCKED_EXACT:
        return False
    if any(part in BLOCKED_PARTS for part in posix.parts):
        return False
    if any(_under_prefix(relative, prefix) for prefix in BLOCKED_PREFIXES):
        return False
    if path.suffix.lower() in BLOCKED_SUFFIXES:
        return False
    if path.name in {".DS_Store", ".env", ".env.local"}:
        return False

    if relative in TOP_LEVEL_FILES:
        return True
    if any(_under_prefix(relative, prefix) for prefix in ALLOW_ROOTS):
        # Additional filter for tests: only include demo smoke, package, and submission tests
        if relative.startswith("tests/"):
            demo_tests = {
                "tests/test_demo_smoke.py",
                "tests/test_demo_release_package.py",
                "tests/test_demo_submission_integrity.py",
            }
            return relative in demo_tests
        return True

    return False


def collect_demo_files(root: Path = ROOT) -> list[ArtifactFile]:
    items: list[ArtifactFile] = []

    for top_file in sorted(TOP_LEVEL_FILES):
        src = root / top_file
        if src.is_file():
            items.append(ArtifactFile(source=src, archive_name=top_file))

    for prefix in ALLOW_ROOTS:
        base_dir = root / prefix
        if not base_dir.exists():
            continue
        for child in sorted(base_dir.rglob("*")):
            if is_allowed_source(child):
                rel = child.relative_to(root).as_posix()
                items.append(ArtifactFile(source=child, archive_name=rel))

    # Sort deterministically by archive name
    items.sort(key=lambda item: item.archive_name)
    return items


def scan_bytes(data: bytes, archive_name: str) -> list[str]:
    issues: list[str] = []
    for label, pattern in SECRET_PATTERNS:
        if pattern.search(data):
            issues.append(f"{archive_name}: found prohibited secret ({label})")
    # Only scan text files for home paths
    text_exts = {".py", ".ts", ".tsx", ".js", ".mjs", ".json", ".md", ".sh", ".txt", ".yml", ".yaml"}
    if Path(archive_name).suffix.lower() in text_exts:
        for label, pattern in USER_PATH_PATTERNS:
            if pattern.search(data):
                issues.append(f"{archive_name}: found absolute user home path ({label})")
    return issues


def validate_demo_sources(files: Sequence[ArtifactFile]) -> list[str]:
    issues: list[str] = []
    names = {item.archive_name for item in files}

    # Verify essential deliverables
    required_files = {
        "DEMO_INSTALL.md",
        "LICENSE",
        "Makefile",
        "README.md",
        "THIRD_PARTY_NOTICES.md",
        "requirements/locks/base-py312.txt",
        "requirements/locks/test-py312.txt",
        "scripts/quickstart.sh",
        "scripts/doctor.py",
        "scripts/setup_models.py",
        "examples/eacl_demo/sample_paper.pdf",
        "examples/eacl_demo/sample_paper.tex",
        "examples/eacl_demo/sample_figure.png",
        "examples/eacl_demo/QUESTIONS.md",
        "examples/eacl_demo/LICENSE.md",
        "examples/eacl_demo/EXPECTED_EVIDENCE.json",
        "tests/test_demo_smoke.py",
        "frontend/package.json",
        "frontend/package-lock.json",
        "backend/main.py",
    }
    missing = required_files - names
    if missing:
        issues.append(f"Missing required release files: {sorted(missing)}")

    for item in files:
        if item.source.stat().st_size > MAX_FILE_BYTES:
            issues.append(f"{item.archive_name} exceeds {MAX_FILE_BYTES} bytes")
        try:
            raw = item.source.read_bytes()
        except OSError as exc:
            issues.append(f"Cannot read {item.source}: {exc}")
            continue
        issues.extend(scan_bytes(raw, item.archive_name))

    return issues


def build_manifest(files: Sequence[ArtifactFile]) -> dict:
    file_records = []
    for item in files:
        data = item.source.read_bytes()
        file_records.append({
            "path": item.archive_name,
            "sha256": sha256_bytes(data),
            "size_bytes": len(data),
        })
    return {
        "schema_version": "scholar_demo_package_manifest_v1",
        "release_name": "ScholAR EACL 2027 Demo v1.0.0",
        "git_tag": "eacl-demo-2027-v1.0.0",
        "license": "MIT",
        "total_files": len(file_records),
        "created_at_utc": "2026-09-22T00:00:00Z",
        "required_tools": {
            "python": ">=3.12.0",
            "node": ">=18.0.0",
            "npm": ">=9.0.0",
            "ollama": ">=0.5.0",
        },
        "supported_environments": [
            "macOS on Apple Silicon (tested: M3 Pro 18GB)",
            "Linux Ubuntu 22.04+ (16GB RAM or 8GB VRAM GPU)",
        ],
        "notes": "Model weights are downloaded separately via 'make demo-model' under their community licenses.",
        "files": file_records,
    }


def build_demo_archive(output_path: Path = DEFAULT_OUTPUT) -> Path:
    files = collect_demo_files()
    issues = validate_demo_sources(files)
    if issues:
        raise DemoPackagePolicyError(f"Package policy violated:\n" + "\n".join(issues))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(files)
    manifest_bytes = (json.dumps(manifest, indent=2) + "\n").encode("utf-8")

    # Build zip file deterministically
    with zipfile.ZipFile(output_path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        # 1. Manifest at archive root
        manifest_info = zipfile.ZipInfo(filename=f"{ARCHIVE_ROOT}/{MANIFEST_NAME}", date_time=FIXED_ZIP_TIME)
        manifest_info.external_attr = 0o644 << 16
        manifest_info.compress_type = zipfile.ZIP_DEFLATED
        archive.writestr(manifest_info, manifest_bytes)

        # 2. Included files
        for item in files:
            arcname = f"{ARCHIVE_ROOT}/{item.archive_name}"
            data = item.source.read_bytes()
            mode = item.source.stat().st_mode & 0o777
            zinfo = zipfile.ZipInfo(filename=arcname, date_time=FIXED_ZIP_TIME)
            zinfo.external_attr = (mode << 16) | (0o100000 << 16)
            zinfo.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(zinfo, data)

    # Write .sha256 sidecar
    digest = sha256_file(output_path)
    sidecar = output_path.with_suffix(".zip.sha256")
    sidecar.write_text(f"{digest}  {output_path.name}\n", encoding="utf-8")

    return output_path


def verify_demo_archive(archive_path: Path, expected_files: Sequence[ArtifactFile] | None = None) -> list[str]:
    issues: list[str] = []
    if not archive_path.is_file():
        return [f"Archive does not exist: {archive_path}"]

    if expected_files is None:
        expected_files = collect_demo_files()

    prefix = f"{ARCHIVE_ROOT}/"
    with zipfile.ZipFile(archive_path, "r") as archive:
        namelist = archive.namelist()
        for name in namelist:
            if not name.startswith(prefix):
                issues.append(f"Archive entry does not start with root {prefix}: {name}")
            if ".." in name or name.startswith("/"):
                issues.append(f"Prohibited path traversal: {name}")

        manifest_entry = f"{prefix}{MANIFEST_NAME}"
        if manifest_entry not in namelist:
            issues.append(f"Missing embedded manifest: {manifest_entry}")
            return issues

        manifest_data = json.loads(archive.read(manifest_entry).decode("utf-8"))
        declared_files = {f["path"]: f for f in manifest_data.get("files", [])}

        actual_files = {name[len(prefix):]: name for name in namelist if name != manifest_entry}
        if set(declared_files.keys()) != set(actual_files.keys()):
            diff = set(declared_files.keys()) ^ set(actual_files.keys())
            issues.append(f"Manifest declared files mismatch actual archive files: {diff}")

        for path_str, zname in actual_files.items():
            content = archive.read(zname)
            calc_hash = sha256_bytes(content)
            decl_hash = declared_files[path_str]["sha256"]
            if calc_hash != decl_hash:
                issues.append(f"Hash mismatch for {path_str}: calculated {calc_hash} != manifest {decl_hash}")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Package and verify ScholAR EACL 2027 Demo release.")
    parser.add_argument("--validate-only", action="store_true", help="Validate candidate files without writing zip.")
    parser.add_argument("--verify-archive", type=Path, default=None, help="Verify an existing archive against policy.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output path for release archive.")
    args = parser.parse_args()

    if args.verify_archive:
        issues = verify_demo_archive(args.verify_archive)
        if issues:
            print("Archive verification FAILED:", file=sys.stderr)
            for issue in issues:
                print(f"  - {issue}", file=sys.stderr)
            sys.exit(1)
        print(f"Archive verified successfully: {args.verify_archive}")
        return

    files = collect_demo_files()
    issues = validate_demo_sources(files)
    if issues:
        print("Package policy validation FAILED:", file=sys.stderr)
        for issue in issues:
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(1)

    if args.validate_only:
        print(f"Allowlist validated successfully ({len(files)} files eligible).")
        return

    out = build_demo_archive(args.output)
    verify_issues = verify_demo_archive(out, files)
    if verify_issues:
        print("Built archive verification FAILED:", file=sys.stderr)
        for issue in verify_issues:
            print(f"  - {issue}", file=sys.stderr)
        sys.exit(1)

    digest = sha256_file(out)
    print(f"Built demo package: {out}")
    print(f"SHA-256: {digest}")
    print(f"Sidecar: {out.with_suffix('.zip.sha256')}")


if __name__ == "__main__":
    main()
