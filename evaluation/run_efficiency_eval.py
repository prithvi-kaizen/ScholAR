"""
run_efficiency_eval.py
Measure the local-deployability cost of the ScholAR answering path, per model, so the
paper can back its "runs on a laptop" claim with numbers instead of adjectives.

For a sample of real benchmark questions it times the two stages that actually cost
wall-clock -- retrieval (BM25 over the paper's chunks) and generation (the local LLM) --
and reads Ollama's own timing metadata (prompt/eval token counts and durations) plus the
loaded model's memory footprint. No scoring, no NLI, no embedder: this file only measures
speed and size, so nothing here perturbs the quality numbers reported elsewhere.

Run one model at a time (a laptop holds one at a time):
  python3 evaluation/run_efficiency_eval.py --model qwen3.5:9b
  python3 evaluation/run_efficiency_eval.py --model gemma4:12b
  python3 evaluation/run_efficiency_eval.py --model llama3.1:8b
  python3 evaluation/run_efficiency_eval.py --model mistral:7b
Results merge per model (resumable) into evaluation/results/efficiency_results.json.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.retrieval_service import retrieve_chunks  # noqa: E402
from backend.services.ollama_service import (  # noqa: E402
    OLLAMA_BASE_URL, OLLAMA_MODEL, _ollama_options,
)

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES = PROJECT_ROOT / "evaluation" / "human_eval" / "cases.json"
OUT = PROJECT_ROOT / "evaluation" / "results" / "efficiency_results.json"
TOP_K = 4  # matches the answering path in run_comparison_eval.py

INDIRECT_RULES = (
    "Answer the question using ONLY the evidence below. Keep it under 180 words. "
    "After each factual claim, cite the supporting evidence by its identifier like [E1]. "
    "Cite only evidence identifiers that appear in the list; never write a page number yourself."
)


def evidence_block(chunks: list[dict]) -> str:
    return "\n".join(
        f"[E{i+1}] (p. {c.get('page')}): {c.get('text','')[:600]}" for i, c in enumerate(chunks)
    )


async def ollama_generate(prompt: str, model: str) -> dict:
    """Call /api/generate directly and keep Ollama's timing metadata (durations in ns)."""
    payload = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "options": _ollama_options(0.1),
    }
    async with httpx.AsyncClient(timeout=300.0, trust_env=False) as client:
        r = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        r.raise_for_status()
        return r.json()


async def loaded_footprint(model: str) -> dict:
    """Model size on disk (/api/tags) and loaded size incl. KV cache (/api/ps)."""
    out = {"disk_gb": None, "loaded_gb": None}
    async with httpx.AsyncClient(timeout=30.0, trust_env=False) as client:
        try:
            tags = (await client.get(f"{OLLAMA_BASE_URL}/api/tags")).json().get("models", [])
            for m in tags:
                if m.get("name") == model or m.get("model") == model:
                    out["disk_gb"] = round(m.get("size", 0) / 1e9, 2)
            ps = (await client.get(f"{OLLAMA_BASE_URL}/api/ps")).json().get("models", [])
            for m in ps:
                if m.get("name") == model or m.get("model") == model:
                    out["loaded_gb"] = round(m.get("size", 0) / 1e9, 2)
        except Exception as exc:  # pragma: no cover
            print(f"[footprint] unavailable ({exc})")
    return out


async def measure(model: str, n: int) -> dict:
    cases = json.loads(CASES.read_text(encoding="utf-8"))[:n]
    per_case = []
    # warm-up (load weights into memory; excluded from stats)
    c0 = cases[0]
    chunks0 = json.loads((DATA_DIR / c0["paper_id"] / "chunks.json").read_text())
    await ollama_generate(
        f"{INDIRECT_RULES}\n\nEvidence:\n{evidence_block(retrieve_chunks(c0['question'], chunks0, limit=TOP_K))}"
        f"\n\nQuestion: {c0['question']}", model)

    for i, c in enumerate(cases, 1):
        chunks = json.loads((DATA_DIR / c["paper_id"] / "chunks.json").read_text())
        t0 = time.perf_counter()
        ctx = retrieve_chunks(c["question"], chunks, limit=TOP_K)
        retr_ms = (time.perf_counter() - t0) * 1000
        prompt = f"{INDIRECT_RULES}\n\nEvidence:\n{evidence_block(ctx)}\n\nQuestion: {c['question']}"
        t1 = time.perf_counter()
        data = await ollama_generate(prompt, model)
        gen_wall_s = time.perf_counter() - t1
        eval_count = data.get("eval_count", 0)
        eval_dur_s = data.get("eval_duration", 0) / 1e9
        per_case.append({
            "case_id": c["case_id"],
            "retr_ms": round(retr_ms, 2),
            "gen_wall_s": round(gen_wall_s, 3),
            "e2e_s": round(retr_ms / 1000 + gen_wall_s, 3),
            "prompt_tokens": data.get("prompt_eval_count", 0),
            "gen_tokens": eval_count,
            "tok_per_s": round(eval_count / eval_dur_s, 1) if eval_dur_s else None,
        })
        print(f"[{model}] [{i}/{len(cases)}] e2e {per_case[-1]['e2e_s']}s  {per_case[-1]['tok_per_s']} tok/s")

    fp = await loaded_footprint(model)
    return {"per_case": per_case, "footprint": fp, "n": len(per_case)}


def _mean(xs): return round(statistics.mean(xs), 3) if xs else None
def _pct(xs, q): return round(sorted(xs)[min(len(xs) - 1, int(q * len(xs)))], 3) if xs else None


def summarize(m: dict) -> dict:
    pc = m["per_case"]
    e2e = [x["e2e_s"] for x in pc]
    return {
        "n": m["n"],
        "disk_gb": m["footprint"]["disk_gb"],
        "loaded_gb": m["footprint"]["loaded_gb"],
        "retr_ms_mean": _mean([x["retr_ms"] for x in pc]),
        "gen_s_mean": _mean([x["gen_wall_s"] for x in pc]),
        "e2e_s_mean": _mean(e2e),
        "e2e_s_p95": _pct(e2e, 0.95),
        "tok_per_s_mean": _mean([x["tok_per_s"] for x in pc if x["tok_per_s"]]),
        "gen_tokens_mean": _mean([x["gen_tokens"] for x in pc]),
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--n", type=int, default=20, help="number of questions to time")
    args = ap.parse_args()

    raw = await measure(args.model, args.n)

    blob = {"generated_at": datetime.now().isoformat(timespec="seconds"), "summary": {}, "raw": {}}
    if OUT.exists():
        blob = json.loads(OUT.read_text())
        blob["generated_at"] = datetime.now().isoformat(timespec="seconds")
    blob["raw"][args.model] = raw
    blob["summary"][args.model] = summarize(raw)
    OUT.write_text(json.dumps(blob, indent=2, ensure_ascii=False))

    s = blob["summary"][args.model]
    print(f"\n== {args.model} (n={s['n']}) ==")
    print(f"  footprint : {s['disk_gb']} GB disk, {s['loaded_gb']} GB loaded")
    print(f"  retrieval : {s['retr_ms_mean']} ms/query")
    print(f"  generation: {s['gen_s_mean']} s/query, {s['tok_per_s_mean']} tok/s, {s['gen_tokens_mean']} tokens")
    print(f"  end-to-end: {s['e2e_s_mean']} s mean, {s['e2e_s_p95']} s p95")
    print(f"wrote {OUT.name} ({len(blob['summary'])} model(s))")


def _selfcheck() -> None:
    m = {"n": 2, "footprint": {"disk_gb": 5.4, "loaded_gb": 6.7}, "per_case": [
        {"retr_ms": 10, "gen_wall_s": 4.0, "e2e_s": 4.01, "gen_tokens": 100, "tok_per_s": 25.0},
        {"retr_ms": 20, "gen_wall_s": 6.0, "e2e_s": 6.02, "gen_tokens": 200, "tok_per_s": 33.0}]}
    s = summarize(m)
    assert s["retr_ms_mean"] == 15 and s["e2e_s_mean"] == round((4.01 + 6.02) / 2, 3), s
    assert s["e2e_s_p95"] == 6.02 and s["tok_per_s_mean"] == 29.0, s
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        asyncio.run(main())
