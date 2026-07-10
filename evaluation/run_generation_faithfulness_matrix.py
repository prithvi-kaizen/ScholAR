"""
run_generation_faithfulness_matrix.py
Automated 4-model generation-faithfulness matrix. Runs the SAME per-case
generation-faithfulness evaluation as run_generation_faithfulness_eval.py, but
across several local models, so we get an automated model-comparison table (the
automated sibling of the human-evaluation Table 4) with no human scoring.

# ponytail: reuses evaluate_case()/aggregate() from run_generation_faithfulness_eval;
# no new scoring logic here, only the per-model loop and matrix assembly.

Prerequisites: backend running (make backend) with the anchor papers prepared, and
the models installed. Contrast with run_generation_faithfulness_eval.py (one model)
and run_faithfulness_eval.py (gold claims vs retrieval, no generation).

Run from repo root:
    python3 evaluation/run_generation_faithfulness_matrix.py [--models a,b,c] [--limit N] [--backend URL]
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

EVAL_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EVAL_DIR))

# Importing the single-model evaluator also wires up its embedder + NLI scorer.
import run_generation_faithfulness_eval as gfe  # noqa: E402

DEFAULT_MODELS = ["qwen3.5:9b", "gemma4:12b", "llama3.1:8b", "mistral:7b"]
MATRIX_JSON = gfe.RESULTS_DIR / "generation_faithfulness_matrix.json"
MATRIX_REPORT = gfe.RESULTS_DIR / "generation_faithfulness_matrix_report.md"

# Two benchmarks: the 100-case diverse set (25 papers, same set the human eval scores)
# and the 51-case labeled set (3 papers, comparable to the single-model Table 3).
CASE_SETS = {"diverse": EVAL_DIR / "human_eval" / "cases.json", "labeled": gfe.CASES_PATH}


def _normalize(case: dict) -> dict:
    """Map the diverse human-eval schema (question/case_id/capability) onto the fields
    evaluate_case() reads (query/id/claim_type). Labeled cases already match and pass through.
    Generation-faithfulness needs no gold chunk, so the unlabeled diverse set is fine."""
    if "query" in case:
        return case
    return {**case,
            "id": case.get("case_id") or case.get("id"),
            "query": case.get("question", ""),
            "claim_type": case.get("capability", "general")}


def run_model(model: str, cases: list[dict], backend: str) -> dict:
    rows = []
    for i, case in enumerate(cases, start=1):
        try:
            row = gfe.evaluate_case(case, backend, model)
        except Exception as exc:  # keep going; one bad case should not sink the run
            print(f"  [{model}] [{i}/{len(cases)}] {case['id']}: ERROR {type(exc).__name__}: {exc}")
            continue
        rows.append(row)
        print(f"  [{model}] [{i}/{len(cases)}] {case['id']}: "
              f"gen_faith={row['generation_faithfulness']} cit={row['citation_support_rate']}")
    return {"model": model, "n_evaluated": len(rows), "summary": gfe.aggregate(rows), "rows": rows}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument("--models", default=",".join(DEFAULT_MODELS),
                    help="comma-separated model tags (default: the 4 human-eval models)")
    ap.add_argument("--cases", default="diverse",
                    help="'diverse' (100-case/25-paper, default), 'labeled' (51-case/3-paper), or a JSON path")
    ap.add_argument("--limit", type=int, default=0, help="first N cases (smoke test)")
    args = ap.parse_args()

    try:
        with urllib.request.urlopen(f"{args.backend}/health", timeout=10) as r:
            if not json.loads(r.read().decode()).get("ollama_available"):
                sys.exit("Backend up but Ollama unavailable.")
    except Exception as exc:
        sys.exit(f"Cannot reach backend at {args.backend} (run `make backend`): {exc}")

    cases_path = CASE_SETS.get(args.cases, Path(args.cases))
    cases = [_normalize(c) for c in json.loads(cases_path.read_text())]
    if args.limit:
        cases = cases[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]

    per_model = []
    for model in models:
        print(f"\n=== {model} ({len(cases)} cases) ===")
        per_model.append(run_model(model, cases, args.backend))

    # Merge into any existing matrix so per-model runs accumulate (and are resumable).
    # Only merge when the prior file was built on the SAME case set, so diverse and
    # labeled results never mix in one matrix.
    by_model: dict[str, dict] = {}
    if MATRIX_JSON.exists():
        try:
            prior = json.loads(MATRIX_JSON.read_text())
        except (json.JSONDecodeError, OSError):
            prior = {}
        if prior.get("case_set") == args.cases:
            by_model = {m["model"]: m for m in prior.get("detail", [])}
        elif prior.get("detail"):
            print(f"[warn] existing matrix is case_set={prior.get('case_set')!r}, this run is "
                  f"{args.cases!r}; starting a fresh matrix so sets don't mix.")
    for m in per_model:
        by_model[m["model"]] = m  # upsert: refresh this run's model(s), keep the rest

    order = {name: i for i, name in enumerate(DEFAULT_MODELS)}
    detail = sorted(by_model.values(), key=lambda m: (order.get(m["model"], len(order)), m["model"]))

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "nli_mode": gfe._NLI._mode,
        "case_set": args.cases,
        "n_cases": len(cases),
        "measures": "per-model faithfulness of GENERATED answers; "
                    "automated sibling of the human-evaluation model matrix",
        "models": [{"model": m["model"], "n_evaluated": m["n_evaluated"], **m["summary"]} for m in detail],
        "detail": detail,
    }
    gfe.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MATRIX_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _write_report(payload)
    print(f"\nWrote {MATRIX_JSON}\nWrote {MATRIX_REPORT}")
    for m in payload["models"]:
        d = m["citation_support_distribution"]
        print(f"  {m['model']:<14} gen_faith={m['mean_generation_faithfulness']} "
              f"contra={m['mean_hallucination_rate']} cite={m['citation_support_rate_micro']} "
              f"(S/P/U {d['supported']}/{d['partial']}/{d['unsupported']})")


def _write_report(p: dict) -> None:
    L = [
        "# ScholAR Generation-Faithfulness Matrix (automated, 4 local models)",
        "",
        f"NLI mode: {p['nli_mode']}. Case set: {p['case_set']} ({p['n_cases']} cases).",
        "",
        "Automated per-model faithfulness of ScholAR's generated answers. Same metric and "
        "same set as the single-model generation-faithfulness table; only the generation "
        "model changes. No human scoring is involved.",
        "",
        "| Model | Gen-faithfulness | Contradiction rate | Citation support | S / P / U |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in p["models"]:
        d = m["citation_support_distribution"]
        L.append(f"| `{m['model']}` | {m['mean_generation_faithfulness']} | "
                 f"{m['mean_hallucination_rate']} | {m['citation_support_rate_micro']} | "
                 f"{d['supported']} / {d['partial']} / {d['unsupported']} |")
    MATRIX_REPORT.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
