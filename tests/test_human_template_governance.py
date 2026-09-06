"""Focused regressions for the checked-in human-study templates."""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from evaluation.validate_human_templates import BASE as LIVE_HUMAN_BASE
from evaluation.validate_human_templates import TEMPLATE_DIRS, validate_templates


def _write(path: Path, content: bytes = b"{}") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


class TestHumanTemplateGovernance(unittest.TestCase):
    def _copy_templates(self, destination: Path) -> None:
        for name in TEMPLATE_DIRS:
            shutil.copytree(LIVE_HUMAN_BASE / name, destination / name)

    def test_nested_extra_human_data_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "human_eval"
            self._copy_templates(base)
            extra = base / TEMPLATE_DIRS[0] / "completed" / "responses.json"
            _write(extra, b"[]")

            errors = validate_templates(base)

        self.assertTrue(any("completed" in error for error in errors), errors)
        self.assertTrue(any("responses.json" in error for error in errors), errors)

    def test_unrecognized_template_scaffold_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "human_eval"
            self._copy_templates(base)
            shutil.copytree(base / TEMPLATE_DIRS[0], base / "new_study_v1")

            errors = validate_templates(base)

        self.assertTrue(any("new_study_v1" in error for error in errors), errors)

    def test_symlinked_template_entry_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir) / "human_eval"
            self._copy_templates(base)
            outside = Path(tmpdir) / "outside.json"
            _write(outside)
            link = base / TEMPLATE_DIRS[0] / "linked.json"
            try:
                link.symlink_to(outside)
            except OSError as exc:  # pragma: no cover - platform permission boundary
                self.skipTest(f"symlinks unavailable: {exc}")

            errors = validate_templates(base)

        self.assertTrue(any("linked.json" in error for error in errors), errors)


if __name__ == "__main__":
    unittest.main()
