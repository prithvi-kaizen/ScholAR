"""
add_confidence_intervals.py
Every comparative claim in the paper rests on a single-run point estimate with no variance
(reviewers flagged this across Tables 2, 6, 7). This computes bootstrap 95% CIs from the
row-level result files already on disk, plus the paired ScholAR-vs-PaperQA2 deltas that the
abstract's word "leads" depends on. No re-generation, no model calls.

Run:  python3 evaluation/add_confidence_intervals.py
Writes: evaluation/results/confidence_intervals.json
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "evaluation" / "results"
OUT = RES / "confidence_intervals.json"
RNG = random.Random(20260715)
B = 10000


def ci(xs: list[float]) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    n = len(xs)
    means = sorted(sum(xs[RNG.randrange(n)] for _ in range(n)) / n for _ in range(B))
    return {"mean": round(sum(xs) / n, 3), "lo": round(means[int(0.025 * B)], 3),
            "hi": round(means[int(0.975 * B)], 3), "n": n}


def paired_delta_ci(a: list[float], b: list[float]) -> dict:
    """Bootstrap CI of mean(a) - mean(b) over paired cases (a,b aligned by case)."""
    pairs = [(x, y) for x, y in zip(a, b) if x is not None and y is not None]
    n = len(pairs)
    diffs = sorted(
        sum((lambda p: p[0] - p[1])(pairs[RNG.randrange(n)]) for _ in range(n)) / n
        for _ in range(B))
    point = sum(x - y for x, y in pairs) / n
    lo, hi = diffs[int(0.025 * B)], diffs[int(0.975 * B)]
    return {"delta": round(point, 3), "lo": round(lo, 3), "hi": round(hi, 3), "n": n,
            "significant": bool(lo > 0 or hi < 0)}


def comparison() -> dict:
    d = json.loads((RES / "comparison_results.json").read_text())["rows"]
    by = defaultdict(lambda: defaultdict(dict))  # system -> metric -> {case: val}
    for r in d.values():
        for m in ("gen_faithfulness", "must_include_recall", "citation_f1"):
            by[r["system"]][m][r["case_id"]] = r.get(m)
    out = {"per_system": {}, "paired_vs_paperqa2": {}}
    for sysname, mets in by.items():
        out["per_system"][sysname] = {m: ci(list(v.values())) for m, v in mets.items()}
    # the claim: ScholAR "leads" PaperQA2-style on correctness and faithfulness
    for m in ("must_include_recall", "gen_faithfulness", "citation_f1"):
        s = by["scholar"][m]
        p = by["paperqa2"][m]
        cases = sorted(set(s) & set(p))
        out["paired_vs_paperqa2"][m] = paired_delta_ci([s[c] for c in cases], [p[c] for c in cases])
    return out


def retrieval() -> dict:
    d = json.loads((RES / "retrieval_eval_results_scaled.json").read_text())["rows"]
    ranks = defaultdict(list)  # retriever -> [first_relevant_rank or None]
    for r in d:
        ranks[r["retriever"]].append(r.get("first_relevant_rank"))
    out = {}
    for ret, rk in ranks.items():
        r5 = [1.0 if (x and x <= 5) else 0.0 for x in rk]
        mrr = [(1.0 / x if x else 0.0) for x in rk]
        out[ret] = {"recall@5": ci(r5), "mrr": ci(mrr)}
    return out


def abstention() -> dict:
    d = json.loads((RES / "abstention_results.json").read_text())["raw"]
    out = {}
    for model, rows in d.items():
        out[model] = ci([1.0 if r["abstained"] else 0.0 for r in rows])
    return out


if __name__ == "__main__":
    res = {"bootstrap_resamples": B, "comparison": comparison(),
           "retrieval": retrieval(), "abstention": abstention()}
    OUT.write_text(json.dumps(res, indent=2))

    print("== Comparison, paired ScholAR - PaperQA2-style (the 'leads' claim) ==")
    for m, v in res["comparison"]["paired_vs_paperqa2"].items():
        verdict = "SIGNIFICANT" if v["significant"] else "not sig. (CI spans 0)"
        print(f"  {m:20} delta={v['delta']:+.3f}  95% CI [{v['lo']:+.3f}, {v['hi']:+.3f}]  {verdict}")
    print("\n== Comparison per-system 95% CIs ==")
    for s, mets in res["comparison"]["per_system"].items():
        f = mets["gen_faithfulness"]; c = mets["must_include_recall"]
        print(f"  {s:12} faith {f['mean']} [{f['lo']},{f['hi']}]   correct {c['mean']} [{c['lo']},{c['hi']}]")
    print("\n== Retrieval R@5 95% CIs ==")
    for ret, v in res["retrieval"].items():
        r = v["recall@5"]; print(f"  {ret:32} R@5 {r['mean']} [{r['lo']},{r['hi']}]")
    print("\n== Abstention 95% CIs ==")
    for m, v in res["abstention"].items():
        print(f"  {m:12} {v['mean']} [{v['lo']},{v['hi']}]  (n={v['n']})")
    print(f"\nwrote {OUT.relative_to(ROOT)}")
