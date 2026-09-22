"""Regression tests for the ScholAR EACL 2027 Demonstration release package.

Verifies:
1. Allowlist enforcement and exclusion of private/runtime data.
2. Single archive root and absence of path traversal.
3. Strict determinism (two builds produce identical SHA-256 digests).
4. Manifest consistency and tamper detection.
5. Detection of secrets or unauthorized user home paths.
6. Execution of model-free smoke tests from a clean temporary extraction.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.package_demo_release import (
    ARCHIVE_ROOT,
    MANIFEST_NAME,
    build_demo_archive,
    collect_demo_files,
    scan_bytes,
    sha256_file,
    validate_demo_sources,
    verify_demo_archive,
)

ROOT = Path(__file__).resolve().parents[1]


class TestDemoReleasePackage(unittest.TestCase):
    def test_allowlist_contains_essential_runtime_and_excludes_private_state(self) -> None:
        files = collect_demo_files()
        names = {item.archive_name for item in files}

        # Required deliverables
        self.assertIn("DEMO_INSTALL.md", names)
        self.assertIn("LICENSE", names)
        self.assertIn("THIRD_PARTY_NOTICES.md", names)
        self.assertIn("requirements/locks/base-py312.txt", names)
        self.assertIn("requirements/locks/test-py312.txt", names)
        self.assertIn("examples/eacl_demo/sample_paper.pdf", names)
        self.assertIn("examples/eacl_demo/QUESTIONS.md", names)
        self.assertIn("examples/eacl_demo/LICENSE.md", names)
        self.assertIn("examples/eacl_demo/EXPECTED_EVIDENCE.json", names)
        self.assertIn("tests/test_demo_smoke.py", names)
        self.assertIn("frontend/package.json", names)
        self.assertIn("backend/main.py", names)

        # Disallowed files
        self.assertNotIn("LICENSE-ANONYMOUS", names)
        self.assertNotIn("ANONYMOUS_ARTIFACT.md", names)
        self.assertFalse(any(name.startswith("backend/data/") for name in names))
        self.assertFalse(any(name.startswith("evaluation/") for name in names))
        self.assertFalse(any(name.startswith("paper/") for name in names))
        self.assertFalse(any(Path(name).suffix in {".db", ".sqlite", ".npy", ".pkl", ".log"} for name in names))
        self.assertFalse(any(name.endswith(".env") or name.endswith(".env.local") for name in names))

    def test_source_scan_passes_and_mit_license_is_standard(self) -> None:
        files = collect_demo_files()
        self.assertEqual(validate_demo_sources(files), [])
        license_item = next(item for item in files if item.archive_name == "LICENSE")
        content = license_item.source.read_text(encoding="utf-8")
        self.assertIn("MIT License", content)
        self.assertNotIn("Anonymous", content)

    def test_build_is_deterministic_and_manifest_is_exact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_dir = Path(temporary)
            first_zip = build_demo_archive(temp_dir / "first.zip")
            second_zip = build_demo_archive(temp_dir / "second.zip")

            self.assertEqual(sha256_file(first_zip), sha256_file(second_zip))
            self.assertEqual(verify_demo_archive(first_zip, collect_demo_files()), [])

            with zipfile.ZipFile(first_zip) as archive:
                prefix = f"{ARCHIVE_ROOT}/"
                manifest_data = json.loads(archive.read(f"{prefix}{MANIFEST_NAME}"))
                self.assertEqual(manifest_data["release_name"], "ScholAR EACL 2027 Demo v1.0.0")
                self.assertEqual(manifest_data["git_tag"], "eacl-demo-2027-v1.0.0")

                declared_paths = {entry["path"] for entry in manifest_data["files"]}
                actual_paths = {
                    name[len(prefix):]
                    for name in archive.namelist()
                    if name != f"{prefix}{MANIFEST_NAME}"
                }
                self.assertEqual(declared_paths, actual_paths)

    def test_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_dir = Path(temporary)
            zip_path = build_demo_archive(temp_dir / "test.zip")
            tampered_path = temp_dir / "tampered.zip"

            prefix = f"{ARCHIVE_ROOT}/"
            with zipfile.ZipFile(zip_path, "r") as src, zipfile.ZipFile(tampered_path, "w") as dst:
                for item in src.infolist():
                    data = src.read(item.filename)
                    if item.filename.endswith("DEMO_INSTALL.md"):
                        data = data + b"\n# Tampered content\n"
                    dst.writestr(item, data)

            issues = verify_demo_archive(tampered_path)
            self.assertTrue(any("Hash mismatch" in issue for issue in issues))

    def test_secret_scanner_blocks_prohibited_patterns(self) -> None:
        fake_secret = b"s" + b"k-" + b"123456789012345678901234"
        issues = scan_bytes(fake_secret, "fake_file.py")
        self.assertTrue(any("OpenAI-style secret" in issue for issue in issues))

    def test_clean_extraction_passes_smoke_test(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            temp_dir = Path(temporary)
            zip_path = build_demo_archive(temp_dir / "clean_test.zip")

            extract_dir = temp_dir / "extracted"
            with zipfile.ZipFile(zip_path, "r") as archive:
                archive.extractall(extract_dir)

            package_root = extract_dir / ARCHIVE_ROOT
            self.assertTrue(package_root.is_dir())
            self.assertTrue((package_root / "DEMO_INSTALL.md").is_file())
            self.assertTrue((package_root / "examples/eacl_demo/sample_paper.pdf").is_file())
            self.assertTrue((package_root / "tests/test_demo_smoke.py").is_file())

            # Run smoke test inside extracted root using existing venv python
            python_bin = sys.executable
            result = subprocess.run(
                [python_bin, "-m", "pytest", "tests/test_demo_smoke.py"],
                cwd=package_root,
                env=dict(os.environ, PYTHONPATH=str(package_root)),
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(
                result.returncode,
                0,
                f"Smoke test failed in extracted root:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}",
            )


if __name__ == "__main__":
    unittest.main()
