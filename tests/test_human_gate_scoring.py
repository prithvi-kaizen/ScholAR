"""Deterministic regressions for the predeclared human support/coverage gate."""

from __future__ import annotations

import hashlib
import unittest

from backend.schemas.answer_trace import AnswerTrace
from evaluation.release.human_scoring import score_primary_gate
from evaluation.release.io import canonical_json_bytes, read_jsonl, sha256_bytes
from evaluation.release.schemas import (
    CanonicalKey,
    FrozenKeyPoint,
    FrozenOutputRef,
    HumanEvaluationBundle,
    HumanSupportLabel,
    KeyPointCoverageLabel,
    PairedComparisonSpec,
    RawReleaseRow,
)
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "evaluation/fixtures/releases/release_v1_minimal/raw/rows.jsonl"


class TestHumanGateScoring(unittest.TestCase):
    def _rows(self) -> list[RawReleaseRow]:
        base = next(row for row in read_jsonl(FIXTURE, RawReleaseRow) if row.trace is not None and row.status.value == "SUCCESS")
        rows: list[RawReleaseRow] = []
        for case_id, seed in (("case_one", 11), ("case_two", 29)):
            for system in ("control", "treatment"):
                key = CanonicalKey(system=system, model="fixture-model:1", seed=seed, case_id=case_id)
                condition = hashlib.sha256(f"{system}:{case_id}:{seed}".encode()).hexdigest()
                identity = base.identity.model_copy(update={
                    "system_name": system,
                    "seed": seed,
                    "condition_sha256": condition,
                })
                trace = dict(base.trace or {})
                trace["trace_id"] = f"trace_{system}_{case_id}_{seed}"
                rows.append(base.model_copy(deep=True, update={
                    "key": key,
                    "identity": identity,
                    "trace": trace,
                }))
        return rows

    @staticmethod
    def _output(row: RawReleaseRow) -> FrozenOutputRef:
        trace = AnswerTrace.model_validate(row.trace)
        return FrozenOutputRef(
            release_id=row.release_id,
            run_id=row.run_id,
            key=row.key,
            condition_sha256=row.identity.condition_sha256,
            trace_id=trace.trace_id,
            raw_row_sha256=sha256_bytes(canonical_json_bytes(row)),
            final_answer_sha256=hashlib.sha256(trace.final_answer.encode()).hexdigest(),
        )

    def _bundle(self, rows: list[RawReleaseRow], *, treatment_supported: bool = True, omit_label: bool = False) -> HumanEvaluationBundle:
        instrument = "1" * 64
        support: list[HumanSupportLabel] = []
        coverage: list[KeyPointCoverageLabel] = []
        points: list[FrozenKeyPoint] = []
        for case_id in ("case_one", "case_two"):
            points.append(FrozenKeyPoint(
                case_id=case_id,
                key_point_id=f"kp_{case_id}",
                text_sha256=hashlib.sha256(case_id.encode()).hexdigest(),
            ))
        for row in rows:
            output = self._output(row)
            supported = treatment_supported if row.key.system == "treatment" else False
            for annotator in ("rater_a", "rater_b"):
                if not (omit_label and row.key.system == "treatment" and row.key.case_id == "case_two" and annotator == "rater_b"):
                    support.append(HumanSupportLabel(
                        output=output,
                        claim_id="C1",
                        claim_text_sha256="2" * 64,
                        label="SUPPORTED" if supported else "UNSUPPORTED",
                        annotator_id=annotator,
                        instrument_sha256=instrument,
                    ))
                coverage.append(KeyPointCoverageLabel(
                    output=output,
                    key_point_id=f"kp_{row.key.case_id}",
                    covered=True,
                    annotator_id=annotator,
                    instrument_sha256=instrument,
                ))
        return HumanEvaluationBundle(
            evidence_class="toy",
            claim_status="non_release",
            release_id=rows[0].release_id,
            run_id=rows[0].run_id,
            dataset_sha256="3" * 64,
            support_labels=support,
            key_points=points,
            coverage_labels=coverage,
        )

    def test_predeclared_gate_pass_fail_and_blocked_are_deterministic(self) -> None:
        rows = self._rows()
        spec = PairedComparisonSpec(
            control_system="control",
            treatment_system="treatment",
            bootstrap_samples=1000,
            bootstrap_seed=2027,
        )
        passed = score_primary_gate(self._bundle(rows), spec, rows)
        self.assertEqual(passed.decision, "PASS")
        self.assertEqual(passed.support_delta, 1.0)
        self.assertGreater(passed.support_interval_low or 0.0, 0.0)
        self.assertEqual(passed.coverage_delta, 0.0)

        failed = score_primary_gate(
            self._bundle(rows, treatment_supported=False), spec, rows
        )
        self.assertEqual(failed.decision, "FAIL")
        self.assertEqual(failed.support_delta, 0.0)

        blocked = score_primary_gate(
            self._bundle(rows, omit_label=True), spec, rows
        )
        self.assertEqual(blocked.decision, "BLOCKED")
        self.assertTrue(any("independent" in reason for reason in blocked.reasons))


if __name__ == "__main__":
    unittest.main()
