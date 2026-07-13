"""
build_capability_breakdown.py
Break the scaled retrieval run down by question type (text / mathematical / visual), so the
paper can carry the per-category table that M3SciQA (Table 4) and SciDQA (Table 2) report and
we previously lacked. Aggregates hide where retrieval actually fails; this shows it.

Pure derivation: reads the row-level results already produced by run_retrieval_eval.py and the
capability tags on the diverse cases. No re-run, no model.

Run from repo root:  python3 evaluation/build_capability_breakdown.py [--selfcheck]
Writes: evaluation/results/retrieval_by_capability.json
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
ROWS = EVAL / "results" / "retrieval_eval_results_scaled.json"
CASES = EVAL / "human_eval" / "cases.json"
OUT = EVAL / "results" / "retrieval_by_capability.json"

ORDER = ["single_doc_text", "math", "visual"]


def recall_at(ranks: list[int | None], k: int) -> float:
    return round(sum(1 for r in ranks if r and r <= k) / len(ranks), 3)


def mrr(ranks: list[int | None]) -> float:
    return round(sum((1 / r if r else 0.0) for r in ranks) / len(ranks), 3)


def build() -> dict:
    cap = {c["case_id"]: c["capability"] for c in json.loads(CASES.read_text())}
    rows = json.loads(ROWS.read_text())["rows"]

    buckets: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for r in rows:
        c = cap.get(r["case_id"])
        if c:
            buckets[r["retriever"]][c].append(r.get("first_relevant_rank"))

    out: dict[str, dict] = {}
    for retriever, per_cap in buckets.items():
        out[retriever] = {
            c: {"n": len(per_cap[c]),
                "recall@1": recall_at(per_cap[c], 1),
                "recall@5": recall_at(per_cap[c], 5),
                "mrr": mrr(per_cap[c])}
            for c in ORDER if per_cap.get(c)
        }
    return out


def _selfcheck(res: dict) -> None:
    for retriever, per_cap in res.items():
        total = sum(v["n"] for v in per_cap.values())
        assert total == 100, f"{retriever}: capability counts must sum to 100, got {total}"
        assert per_cap["single_doc_text"]["n"] == 50 and per_cap["math"]["n"] == 25
        for v in per_cap.values():
            assert 0.0 <= v["recall@5"] <= 1.0 and 0.0 <= v["mrr"] <= 1.0
            assert v["recall@1"] <= v["recall@5"], "recall must be monotone in k"
    print("selfcheck OK")


if __name__ == "__main__":
    res = build()
    _selfcheck(res)
    if "--selfcheck" in sys.argv:
        sys.exit()
    OUT.write_text(json.dumps(res, indent=2))
    for retriever in ("bm25_primary_no_page_hints", "dense_only"):
        if retriever in res:
            print(f"\n{retriever}")
            for c, v in res[retriever].items():
                print(f"  {c:16} n={v['n']:3}  R@5={v['recall@5']:.2f}  MRR={v['mrr']:.3f}")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
