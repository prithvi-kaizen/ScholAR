"""
rescore_comparison_judge.py
Re-score the local-baseline comparison's GENERATION FAITHFULNESS with the LLM entailment judge,
fixing the reviewer-flagged inconsistency: the comparison table reported cosine-based faithfulness
(scored before the judge existed) while the per-model table reports judge-based faithfulness, giving
two different numbers for the same system. This puts both on the judge.

Reuses the stored comparison answers (comparison_results.json raw rows carry `answer` and
`ctx_texts`), so no regeneration. Resumable with per-answer checkpoints (long jobs get killed here).

Run (Ollama up, no backend; one judge model loaded):
    FAITH_JUDGE=llm python3 evaluation/rescore_comparison_judge.py
Writes: evaluation/results/comparison_faithfulness_judged.json
"""
from __future__ import annotations

import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ["FAITH_JUDGE"] = "llm"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "evaluation"))
import nli_faithfulness as nli  # noqa: E402  (honours FAITH_JUDGE=llm)

COMP = ROOT / "evaluation" / "results" / "comparison_results.json"
OUT = ROOT / "evaluation" / "results" / "comparison_faithfulness_judged.json"
_SCORER = nli.NLIFaithfulnessScorer()
CHECKPOINT_EVERY = 10


def prose_of(answer: str) -> str:
    m = re.search(r"\*\*\s*Answer\s*\*\*(.*?)(?:\*\*\s*Evidence\s*\*\*|\Z)", answer,
                  re.IGNORECASE | re.DOTALL)
    p = m.group(1) if m else answer
    p = re.sub(r"\[\s*p\.?\s*\d+\s*\]", "", p, flags=re.I)   # drop [p. N]
    p = re.sub(r"\[\s*E?\d+(?:\s*,\s*E?\d+)*\s*\]", "", p)     # drop [n] / [E1, E4]
    return p.strip()


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    return {"n": n,
            "judge_gen_faithfulness": round(sum(r["faith"] for r in rows) / n, 3),
            "judge_hallucination_rate": round(sum(r["halluc"] for r in rows) / n, 3)}


def _save(blob: dict) -> None:
    blob["generated_at"] = datetime.now().isoformat(timespec="seconds")
    for s, rows in blob["raw"].items():
        if rows:
            blob["summary"][s] = summarize(rows)
    OUT.write_text(json.dumps(blob, indent=2))


def main() -> None:
    rows = json.loads(COMP.read_text())["rows"]
    blob = json.loads(OUT.read_text()) if OUT.exists() else {"judge": os.getenv("FAITH_JUDGE_MODEL", "qwen3.5:9b"), "summary": {}, "raw": {}}
    blob.setdefault("raw", {}); blob.setdefault("summary", {})
    done = {(s, r["case_id"]) for s, rs in blob["raw"].items() for r in rs}
    todo = [r for r in rows.values() if (r["system"], r["case_id"]) not in done]
    print(f"resume: {len(done)} done; {len(todo)} to score")

    fresh = 0
    for i, r in enumerate(todo, 1):
        prose = prose_of(r.get("answer", ""))
        chunks = [{"text": t} for t in (r.get("ctx_texts") or []) if t]
        g = _SCORER.score_full(prose, chunks, top_k=len(chunks) or 1) if (prose and chunks) else {"cfs": 0.0, "n_atoms": 0, "n_contradicted": 0}
        na = max(g["n_atoms"], 1)
        blob["raw"].setdefault(r["system"], []).append(
            {"case_id": r["case_id"], "faith": g["cfs"], "halluc": round(g["n_contradicted"] / na, 3)})
        fresh += 1
        if fresh % CHECKPOINT_EVERY == 0:
            _save(blob)
            print(f"  checkpoint {i}/{len(todo)}")
    _save(blob)
    print("\n== judge-based generation faithfulness (was cosine in comparison_results) ==")
    for s, sm in blob["summary"].items():
        print(f"  {s:12} faith {sm['judge_gen_faithfulness']}  halluc {sm['judge_hallucination_rate']}  (n={sm['n']})")
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
