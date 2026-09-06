"""Evaluate explicit and implicit visual retrieval with paper-clustered intervals."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.answer_pipeline import _merge_source_figure_chunks  # noqa: E402
from backend.services.pdf_service import read_json  # noqa: E402
from backend.services.retrieval_service import retrieve_chunks  # noqa: E402


PAPERS_DIR = ROOT / "backend" / "data" / "papers"
DEFAULT_OUTPUT = ROOT / "evaluation" / "results" / "visual_retrieval_eval.json"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class GoldRegion(StrictModel):
    page: int = Field(ge=1)
    bbox_norm: list[float] = Field(min_length=4, max_length=4)

    @model_validator(mode="after")
    def validate_box(self) -> "GoldRegion":
        x0, y0, x1, y1 = self.bbox_norm
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError("gold bbox must be a positive normalized page region")
        return self


class VisualRetrievalCase(StrictModel):
    case_id: str = Field(min_length=1)
    pair_id: str = Field(min_length=1)
    paper_id: str = Field(min_length=1)
    query: str = Field(min_length=1, max_length=8000)
    formulation: Literal["explicit", "implicit"]
    visual_type: Literal["figure", "table", "plot", "diagram", "mixed"]
    visual_necessity: Literal["visual_only", "visual_dominant", "mixed"]
    gold_pages: list[int] = Field(min_length=1)
    gold_regions: list[GoldRegion] = Field(default_factory=list)
    answerable: bool = True
    split: Literal["development", "test"]


class VisualRetrievalBenchmark(StrictModel):
    schema_version: Literal["1.0"] = "1.0"
    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    paper_disjoint_from_development: bool
    cases: list[VisualRetrievalCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_ids_and_splits(self) -> "VisualRetrievalBenchmark":
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("case IDs must be unique")
        development_papers = {
            case.paper_id for case in self.cases if case.split == "development"
        }
        test_papers = {case.paper_id for case in self.cases if case.split == "test"}
        if self.paper_disjoint_from_development and development_papers & test_papers:
            raise ValueError("development and test paper IDs overlap")
        return self


CONDITIONS: dict[str, dict[str, Any]] = {
    "text_only": {
        "include_image_channel": False,
    },
    "clip_page_hybrid": {
        "include_image_channel": True,
        "include_crop_image_channel": False,
        "include_page_image_channel": True,
        "visual_page_backend": "clip",
    },
    "colqwen2_page_hybrid": {
        "include_image_channel": True,
        "include_crop_image_channel": False,
        "include_page_image_channel": True,
        "visual_page_backend": "colqwen2",
    },
    "colqwen2_full_hybrid": {
        "include_image_channel": True,
        "include_crop_image_channel": True,
        "include_page_image_channel": True,
        "visual_page_backend": "colqwen2",
    },
}


def load_chunks(paper_id: str) -> list[dict[str, Any]]:
    directory = PAPERS_DIR / paper_id
    chunks_path = directory / "chunks.json"
    if not chunks_path.is_file():
        raise FileNotFoundError(f"Missing prepared chunks: {chunks_path}")
    chunks = [
        {**item, "source_paper_id": item.get("source_paper_id") or paper_id}
        for item in read_json(chunks_path)
    ]
    figures_path = directory / "figures.json"
    figures = read_json(figures_path) if figures_path.is_file() else []
    return _merge_source_figure_chunks(
        paper_id,
        chunks,
        figures if isinstance(figures, list) else [],
    )


def _iou(left: list[float], right: list[float]) -> float:
    x0 = max(left[0], right[0])
    y0 = max(left[1], right[1])
    x1 = min(left[2], right[2])
    y1 = min(left[3], right[3])
    intersection = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0 else 0.0


def _candidate_boxes(hit: dict[str, Any]) -> list[list[float]]:
    boxes: list[list[float]] = []
    for region in hit.get("candidate_regions") or []:
        raw = region.get("bbox_norm") if isinstance(region, dict) else None
        if isinstance(raw, list) and len(raw) == 4:
            boxes.append([float(value) for value in raw])
    raw_hit = hit.get("bbox_norm")
    if isinstance(raw_hit, list) and len(raw_hit) == 4 and raw_hit != [0.0, 0.0, 1.0, 1.0]:
        boxes.append([float(value) for value in raw_hit])
    return boxes


def evaluate_case(
    case: VisualRetrievalCase,
    condition: str,
    top_k: int,
) -> dict[str, Any]:
    metadata: dict[str, Any] = {}
    hits = retrieve_chunks(
        case.query,
        load_chunks(case.paper_id),
        limit=top_k,
        paper_id=case.paper_id,
        retrieval_metadata=metadata,
        **CONDITIONS[condition],
    )
    pages = [
        int(hit["page"])
        for hit in hits
        if isinstance(hit.get("page"), int)
    ]
    first_relevant = next(
        (rank for rank, page in enumerate(pages, start=1) if page in case.gold_pages),
        None,
    )
    region_hit = False
    if case.gold_regions:
        for hit in hits:
            if hit.get("page") not in {region.page for region in case.gold_regions}:
                continue
            for candidate in _candidate_boxes(hit):
                if any(
                    region.page == hit.get("page")
                    and _iou(candidate, region.bbox_norm) >= 0.5
                    for region in case.gold_regions
                ):
                    region_hit = True
                    break
            if region_hit:
                break
    status = dict(metadata.get("visual_page_retrieval") or {})
    if condition.startswith("colqwen2") and not status.get("model_loaded"):
        raise RuntimeError(
            f"{condition} did not execute ColQwen2 for {case.case_id}: "
            f"{status.get('failure_reason')}"
        )
    return {
        "case_id": case.case_id,
        "pair_id": case.pair_id,
        "paper_id": case.paper_id,
        "condition": condition,
        "formulation": case.formulation,
        "visual_type": case.visual_type,
        "visual_necessity": case.visual_necessity,
        "gold_pages": case.gold_pages,
        "retrieved_pages": pages,
        "first_relevant_rank": first_relevant,
        "recall_at_1": float(first_relevant is not None and first_relevant <= 1),
        "recall_at_3": float(first_relevant is not None and first_relevant <= 3),
        "recall_at_5": float(first_relevant is not None and first_relevant <= 5),
        "reciprocal_rank": 0.0 if first_relevant is None else 1.0 / first_relevant,
        "region_recall_at_05_iou": float(region_hit) if case.gold_regions else None,
        "visual_status": status,
    }


def clustered_interval(
    rows: list[dict[str, Any]],
    metric: str,
    *,
    samples: int = 10_000,
    seed: int = 2027,
) -> dict[str, float | int | None]:
    by_paper: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(metric)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            by_paper[row["paper_id"]].append(float(value))
    paper_means = {
        paper: sum(values) / len(values) for paper, values in by_paper.items()
    }
    values = list(paper_means.values())
    if not values:
        return {"mean": None, "ci_low": None, "ci_high": None, "papers": 0}
    mean = sum(values) / len(values)
    if len(values) == 1:
        return {"mean": mean, "ci_low": mean, "ci_high": mean, "papers": 1}
    generator = random.Random(seed)
    bootstrap = sorted(
        sum(generator.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return {
        "mean": mean,
        "ci_low": bootstrap[int(0.025 * samples)],
        "ci_high": bootstrap[min(int(0.975 * samples), samples - 1)],
        "papers": len(values),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    metrics = (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "reciprocal_rank",
        "region_recall_at_05_iou",
    )
    for condition in CONDITIONS:
        summary[condition] = {}
        for formulation in ("all", "explicit", "implicit"):
            subset = [
                row for row in rows
                if row["condition"] == condition
                and (formulation == "all" or row["formulation"] == formulation)
            ]
            summary[condition][formulation] = {
                "cases": len(subset),
                **{
                    metric: clustered_interval(subset, metric)
                    for metric in metrics
                },
            }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--split", choices=("development", "test"), default="test")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--conditions",
        nargs="+",
        choices=tuple(CONDITIONS),
        default=list(CONDITIONS),
    )
    parser.add_argument("--allow-non-disjoint-test", action="store_true")
    args = parser.parse_args()
    benchmark = VisualRetrievalBenchmark.model_validate(
        json.loads(args.benchmark.read_text(encoding="utf-8"))
    )
    if (
        args.split == "test"
        and not benchmark.paper_disjoint_from_development
        and not args.allow_non_disjoint_test
    ):
        raise SystemExit("Test evaluation requires a paper-disjoint benchmark.")
    cases = [case for case in benchmark.cases if case.split == args.split]
    if not cases:
        raise SystemExit(f"Benchmark has no {args.split} cases.")

    rows = [
        evaluate_case(case, condition, args.top_k)
        for condition in args.conditions
        for case in cases
    ]
    payload = {
        "schema_version": "1.0",
        "benchmark": {
            "name": benchmark.name,
            "version": benchmark.version,
            "paper_disjoint_from_development": benchmark.paper_disjoint_from_development,
            "split": args.split,
        },
        "conditions": args.conditions,
        "top_k": args.top_k,
        "summary": summarize(rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
