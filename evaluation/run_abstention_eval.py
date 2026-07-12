"""
run_abstention_eval.py
Measure whether ScholAR abstains (declines) rather than fabricates when the answer is NOT in the
document, using the provably-unanswerable cases from build_abstention_benchmark.py. Each case
retrieves from the WRONG (target) paper, so a grounded system should refuse; a hallucinating one
invents an answer, worse still if it cites evidence for a fact that is not there.

Detection reuses the exact classifier from the comparison harness (_REFUSAL + substantive_sentences),
so "abstained" here means the same thing as everywhere else in the evaluation. Raw answers are
stored so every label is auditable offline.

Run one model at a time (resumable; merges per model):
  python3 evaluation/run_abstention_eval.py --model qwen3.5:9b
  ... gemma4:12b / llama3.1:8b / mistral:7b
Output: evaluation/results/abstention_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from backend.services.retrieval_service import retrieve_chunks  # noqa: E402
from backend.services.ollama_service import OLLAMA_MODEL, generate  # noqa: E402
from run_comparison_eval import (  # noqa: E402  (reuse the prompt + citation-aware sentence split)
    substantive_sentences, _evidence_block, INDIRECT_RULES, TOP_K,
)

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES = EVAL_DIR / "abstention_cases.json"
OUT = EVAL_DIR / "results" / "abstention_results.json"

# Refusal detector tuned for the UNANSWERABLE setting. The comparison harness's _REFUSAL is
# deliberately narrow (on answerable questions "there is no significant difference" is a real
# answer, not a refusal); here every "there is no information/mention of X" is a genuine
# decline, so we detect those phrasings too. Validated by manual audit of all 4 models' outputs:
# it catches every hand-labeled refusal and leaves only genuine assertions (see the paper's
# abstention subsection). We keep this local rather than widen the shared regex.
ABSTAIN_RE = re.compile(
    r"does ?n.?t (contain|mention|provide|state|include|list|discuss|address|specify|indicate|reference|appear)"
    r"|do(es)? not (contain|mention|provide|have|state|include|list|discuss|address|specify)"
    r"|do ?n.?t (contain|mention|provide|have)"
    r"|is not (explicitly )?(state|mention|includ|list|available|present|provid|discuss|address|specif|found)"
    r"|are not (mention|includ|list|present|provid|discuss)"
    r"|no (information|mention|evidence|reference|data|detail|indication|record|specific|corresponding|explicit|"
    r"numerical|such|clear)"
    r"|cannot (be )?(determin|answer|found)|can.?not be answered|can.?t (answer|provide|determin|be)"
    r"|impossible to (determin|answer|say)"
    r"|not (present|found|available|provided|possible to)"
    r"|there (is|are) no |not answer",
    re.I)


def classify(answer: str) -> tuple[bool, bool]:
    """(abstained, answered_with_citation). Abstained = explicit refusal or no factual claim.
    A non-abstaining answer here is a fabrication about content absent from the paper."""
    subs = substantive_sentences(answer)
    abstained = bool(ABSTAIN_RE.search(answer)) or len(subs) == 0
    cited = any(has_cite for _, has_cite in subs)
    return abstained, (not abstained and cited)


async def answer_one(case: dict, model: str) -> str:
    chunks = json.loads((DATA_DIR / case["target_paper"] / "chunks.json").read_text())
    ctx = retrieve_chunks(case["question"], chunks, limit=TOP_K)
    prompt = f"{INDIRECT_RULES}\n\nEvidence:\n{_evidence_block(ctx)}\n\nQuestion: {case['question']}"
    try:
        return await generate(prompt, temperature=0.1, model=model)
    except Exception as exc:
        return f"[GENERATION ERROR: {type(exc).__name__}: {exc}]"


async def run(model: str, cases: list[dict]) -> list[dict]:
    rows = []
    for i, c in enumerate(cases, 1):
        ans = await answer_one(c, model)
        abst, fab_cite = classify(ans)
        rows.append({"id": c["id"], "target_paper": c["target_paper"],
                     "abstained": abst, "answered_with_citation": fab_cite, "answer": ans})
        print(f"[{model}] [{i}/{len(cases)}] {'ABSTAIN' if abst else 'ANSWERED'}  {c['id']}")
    return rows


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    abst = sum(r["abstained"] for r in rows)
    return {"n": n,
            "abstention_rate": round(abst / n, 3) if n else None,
            "fabrication_rate": round((n - abst) / n, 3) if n else None,
            "fabricated_with_citation": sum(r["answered_with_citation"] for r in rows)}


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--rescore", action="store_true",
                    help="re-classify stored answers with the current detector (no generation)")
    args = ap.parse_args()

    blob = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": {}, "raw": {}}
    if OUT.exists():
        blob = json.loads(OUT.read_text())
        blob["generated_at"] = datetime.now().isoformat(timespec="seconds")

    if args.rescore:
        for model, rows in blob["raw"].items():
            for r in rows:
                r["abstained"], r["answered_with_citation"] = classify(r["answer"])
            blob["summary"][model] = summarize(rows)
        OUT.write_text(json.dumps(blob, indent=2, ensure_ascii=False))
        for model, s in blob["summary"].items():
            print(f"{model:14} abstain={s['abstention_rate']} fabricate={s['fabrication_rate']} "
                  f"(fab+cite {s['fabricated_with_citation']}) n={s['n']}")
        return

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    rows = await run(args.model, cases)
    blob["raw"][args.model] = rows
    blob["summary"][args.model] = summarize(rows)
    OUT.write_text(json.dumps(blob, indent=2, ensure_ascii=False))

    s = blob["summary"][args.model]
    print(f"\n== {args.model} (n={s['n']}) ==")
    print(f"  abstention rate : {s['abstention_rate']}")
    print(f"  fabrication rate: {s['fabrication_rate']} ({s['fabricated_with_citation']} with a citation)")
    print(f"wrote {OUT.name} ({len(blob['summary'])} model(s))")


def _selfcheck() -> None:
    assert classify("The evidence does not contain this information.") == (True, False)
    assert classify("") == (True, False)
    a, fc = classify("The model was trained on 80 sentence pairs per minibatch [E1], which is standard.")
    assert a is False and fc is True, (a, fc)
    a2, fc2 = classify("It uses a learning rate schedule that warms up then decays over training steps.")
    assert a2 is False and fc2 is False, (a2, fc2)
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        asyncio.run(main())
