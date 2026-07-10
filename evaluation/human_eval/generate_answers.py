"""
generate_answers.py
Runs every human-evaluation case through the live ScholAR chat endpoint once per
model, so the same retrieval and citation-grounding pipeline is exercised across
several local generation models (the model-agnostic comparison). Writes answers.json.

Text cases (single_doc_text, multi_doc, hard_retrieval) are answered by all TEXT_MODELS.
Visual cases are answered only by VISION_MODELS (the multimodal ones).

Prerequisites:
  - Backend running from repo root:  make backend   (defaults to http://localhost:8000)
  - The 4 models pulled in Ollama:   ollama pull qwen3.5:9b gemma4:12b llama3.1:8b mistral:7b
  - For multi_doc cases, the secondary papers must already be prepared under
    backend/data/papers/ (they are, for the curated set).

Run from repo root:  python3 evaluation/human_eval/generate_answers.py
Options:  --backend URL   --limit N (first N cases, for a smoke test)   --only-installed
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CASES = HERE / "cases.json"
OUT = HERE / "answers.json"

TEXT_MODELS = ["qwen3.5:9b", "gemma4:12b", "llama3.1:8b", "mistral:7b"]
VISION_MODELS = ["qwen3.5:9b", "gemma4:12b"]


def installed_models() -> set[str]:
    try:
        out = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=15)
        names = set()
        for line in out.stdout.splitlines()[1:]:
            parts = line.split()
            if parts:
                names.add(parts[0])
        return names
    except Exception:
        return set()


def post_chat(backend: str, paper_id: str, body: dict, timeout: float = 300.0) -> dict:
    url = f"{backend}/api/papers/{paper_id}/chat"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def models_for(case: dict) -> list[str]:
    return VISION_MODELS if case["capability"] == "visual" else TEXT_MODELS


def unload_model(model: str) -> None:
    """Evict a model from memory so only one model is ever resident (18GB-safe)."""
    try:
        subprocess.run(["ollama", "stop", model], capture_output=True, timeout=30)
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument("--limit", type=int, default=0, help="only the first N cases (smoke test)")
    ap.add_argument("--only-installed", action="store_true",
                    help="skip models not currently pulled in Ollama instead of erroring")
    ap.add_argument("--models", default="",
                    help="comma-separated subset of models to run (default: all four); "
                         "pass one model to run it alone in its own terminal (resumable)")
    args = ap.parse_args()
    requested = {m.strip() for m in args.models.split(",") if m.strip()} or None

    cases = json.loads(CASES.read_text(encoding="utf-8"))
    if args.limit:
        cases = cases[: args.limit]

    # preflight: backend reachable
    try:
        with urllib.request.urlopen(f"{args.backend}/health", timeout=10) as r:
            health = json.loads(r.read().decode("utf-8"))
        if not health.get("ollama_available"):
            sys.exit("Backend is up but Ollama is not available. Start Ollama first.")
    except Exception as exc:
        sys.exit(f"Cannot reach backend at {args.backend} (run `make backend`): {exc}")

    have = installed_models()
    wanted = requested or (set(TEXT_MODELS) | set(VISION_MODELS))
    absent = sorted(wanted - have) if have else []
    if absent:
        msg = f"Models not pulled in Ollama: {absent}. Run: ollama pull {' '.join(absent)}"
        if args.only_installed:
            print(f"[warn] {msg}\n[warn] --only-installed set: those models will be skipped.")
        else:
            sys.exit(msg + "\n(or pass --only-installed to run a partial set now)")

    # Resume: keep any answers already generated, skip those (case_id, model) pairs.
    # Keyed by (case_id, model) so a retried pair (previously errored) REPLACES the
    # old row instead of appending a duplicate (the C7 fix).
    loaded = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else []
    by_key = {(r["case_id"], r["model"]): r for r in loaded}
    done_pairs = {k for k, r in by_key.items() if not r.get("error")}

    # Which models to run, in a fixed order. We iterate MODEL-OUTER, cases-inner,
    # and unload each model before loading the next, so only ONE model is ever
    # resident in memory at a time (safe for an 18GB machine).
    run_models = [m for m in TEXT_MODELS if (not args.only_installed or m in have)]
    for m in VISION_MODELS:
        if m not in run_models and (not args.only_installed or m in have):
            run_models.append(m)
    if requested:
        run_models = [m for m in run_models if m in requested]

    # count remaining work
    def cases_for_model(model):
        return [c for c in cases if model in models_for(c) and (c["case_id"], model) not in done_pairs]
    total = sum(len(cases_for_model(m)) for m in run_models)
    if done_pairs:
        print(f"[resume] {len(done_pairs)} answers already present; {total} remaining.")
    done = 0

    for model in run_models:
        todo = cases_for_model(model)
        if not todo:
            continue
        print(f"\n=== model {model} ({len(todo)} cases) --- loading; other models stay unloaded ===")
        for case in todo:
            body = {
                "message": case["question"],
                "history": [],
                "web_search": False,
                "secondary_paper_ids": case.get("secondary_paper_ids", []),
                "model": model,
            }
            t0 = time.monotonic()
            try:
                resp = post_chat(args.backend, case["paper_id"], body)
                answer = resp.get("answer", "")
                citations = resp.get("citations", [])
                err = None
            except Exception as exc:
                resp, answer, citations, err = {}, "", [], f"{type(exc).__name__}: {exc}"
            done += 1
            print(f"[{done}/{total}] {case['case_id']} <- {model} "
                  f"({time.monotonic()-t0:.1f}s){' ERROR' if err else ''}")
            by_key[(case["case_id"], model)] = {
                "case_id": case["case_id"],
                "capability": case["capability"],
                "paper_id": case["paper_id"],
                "model": model,
                "question": case["question"],
                "answer": answer,
                "citations": citations,
                "vision": bool(resp.get("vision")) if not err else False,
                "error": err,
            }
            OUT.write_text(json.dumps(list(by_key.values()), indent=2, ensure_ascii=False), encoding="utf-8")
        # done with this model: free its memory before the next one loads
        print(f"=== unloading {model} ===")
        unload_model(model)

    final = list(by_key.values())
    n_err = sum(1 for r in final if r["error"])
    print(f"\nWrote {OUT} ({len(final)} answers, {n_err} errors)")


if __name__ == "__main__":
    main()
