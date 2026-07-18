"""
faithfulness_negative_control.py
A negative control for the faithfulness scorer, directly answering the reviewers' challenge:
"score a deliberately corrupted answer; if the metric does not drop, it does not measure what
it claims." The old cosine scorer FAILS this (a negated or number-swapped claim keeps a high
cosine to its source); the LLM-judge scorer should PASS it.

For each labeled faithfulness case we take its gold claim (entailed by the supporting chunk) and
build a corrupted twin by (a) perturbing every number and (b) negating the main relation. We score
both the true and the corrupted claim against the same supporting chunk and report how far the
metric separates them, and how often it fires CONTRADICTION on the corruption.

Run (judge on):  FAITH_JUDGE=llm python3 evaluation/faithfulness_negative_control.py [--n 20]
Writes: evaluation/results/faithfulness_negative_control.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "evaluation"))

import nli_faithfulness as nli  # noqa: E402
from embedder import LocalEmbedder  # noqa: E402

CASES = ROOT / "evaluation" / "faithfulness_cases.json"
DATA = ROOT / "backend" / "data" / "papers"
OUT = ROOT / "evaluation" / "results" / "faithfulness_negative_control.json"

_NUM = re.compile(r"\d+(?:\.\d+)?")


def corrupt(claim: str) -> str:
    """Perturb every number and negate the relation, so the claim conflicts with its source."""
    def bump(m: re.Match) -> str:
        v = float(m.group(0))
        v = v * 3 + 7 if v else 42          # deterministic, guaranteed different
        return str(int(v)) if v.is_integer() else f"{v:.2f}"
    out = _NUM.sub(bump, claim)
    if out == claim:  # no numbers -> negate
        out = re.sub(r"\bis\b", "is not", out, count=1)
        if out == claim:
            out = "It is false that " + claim[0].lower() + claim[1:]
    return out


def chunk_text(paper_id: str, chunk_id: str) -> str:
    for c in json.loads((DATA / paper_id / "chunks.json").read_text()):
        if c.get("chunk_id") == chunk_id:
            return c.get("text", "")
    return ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    args = ap.parse_args()

    nli.set_embedder(LocalEmbedder())          # so cosine fallback has an embedder if judge is off
    scorer = nli.NLIFaithfulnessScorer()
    print(f"scorer mode: {scorer._mode}")

    cases = json.loads(CASES.read_text())[:args.n]
    rows = []
    for c in cases:
        ct = chunk_text(c["paper_id"], c["supporting_chunk_id"])
        if not ct:
            continue
        chunks = [{"text": ct}]
        true_claim = c["expected_claim"]
        bad_claim = corrupt(true_claim)
        t = scorer.score_full(true_claim, chunks, top_k=1)
        b = scorer.score_full(bad_claim, chunks, top_k=1)
        rows.append({"id": c["id"], "true_cfs": t["cfs"], "true_contradicted": t["n_contradicted"],
                     "corrupt_cfs": b["cfs"], "corrupt_contradicted": b["n_contradicted"],
                     "corrupt_atoms": b["n_atoms"], "corrupt_claim": bad_claim})
        print(f"  {c['id']:26} true={t['cfs']:.2f}(contra {t['n_contradicted']})  "
              f"corrupt={b['cfs']:.2f}(contra {b['n_contradicted']})")

    n = len(rows)
    mean_true = round(sum(r["true_cfs"] for r in rows) / n, 3)
    mean_bad = round(sum(r["corrupt_cfs"] for r in rows) / n, 3)
    # SENSITIVITY: does the metric drop on corrupted claims?
    caught = round(sum(1 for r in rows if r["corrupt_cfs"] < 0.5) / n, 3)
    any_contra = round(sum(1 for r in rows if r["corrupt_contradicted"] > 0) / n, 3)
    # SPECIFICITY: does the metric spuriously flag TRUE (uncorrupted, entailed) claims?
    true_flagged = round(sum(1 for r in rows if r["true_cfs"] < 0.5) / n, 3)
    true_false_contra = round(sum(1 for r in rows if r["true_contradicted"] > 0) / n, 3)
    summary = {"mode": scorer._mode, "n": n, "mean_true_cfs": mean_true,
               "mean_corrupt_cfs": mean_bad, "separation": round(mean_true - mean_bad, 3),
               "corrupt_caught_rate": caught, "corrupt_contradiction_rate": any_contra,
               "true_flagged_unfaithful_rate": true_flagged,
               "true_false_contradiction_rate": true_false_contra}
    OUT.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2))
    print(f"\n== validation ({scorer._mode}, n={n}) ==")
    print(f"  SENSITIVITY  corrupted flagged unfaithful {caught*100:.0f}%, fired contradiction {any_contra*100:.0f}%")
    print(f"  SPECIFICITY  TRUE claims wrongly flagged unfaithful {true_flagged*100:.0f}%, "
          f"spurious contradiction {true_false_contra*100:.0f}%")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
