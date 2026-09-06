"""
run_generation_faithfulness_eval.py
Measures the faithfulness of ScholAR's actual generated answers rather than a
pre-written gold claim. It generates a real answer for each query and scores whether that answer
is grounded in the retrieved context, plus whether each of its inline citations
is actually supported. This is the automated counterpart to the human-eval
citation grading and directly addresses generation hallucination.

Reuses: nli_faithfulness (scoring), retrieve_chunks (context), the live chat
endpoint (generation). Does not modify or re-run the retrieval-support metric.

Prerequisites: backend running (make backend) with the anchor papers prepared,
and one model installed (default OLLAMA_MODEL, e.g. gemma4:12b or qwen3.5:9b).

Run from repo root:
    python3 evaluation/run_generation_faithfulness_eval.py [--model M] [--limit N] [--backend URL]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import urllib.request
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from backend.services.ollama_service import OLLAMA_MODEL  # noqa: E402
from scholar_runner import run_scholar_http  # noqa: E402
import nli_faithfulness as _nli  # noqa: E402

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES_PATH = EVAL_DIR / "faithfulness_cases.json"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "generation_faithfulness_results.json"
REPORT_MD = RESULTS_DIR / "generation_faithfulness_report.md"

# ── load the shared embedder so the NLI tier runs in semantic mode ────────
try:
    from embedder import LocalEmbedder
    _EMBEDDER = LocalEmbedder()
    print(f"[embedder] loaded MiniLM ({_EMBEDDER._loaded} tensors)")
except Exception as exc:  # pragma: no cover
    _EMBEDDER = None
    print(f"[embedder] unavailable ({exc}); NLI runs in lexical fallback mode")
_nli.set_embedder(_EMBEDDER)
_NLI = _nli.NLIFaithfulnessScorer()


def load_chunks(paper_id: str) -> list[dict]:
    return json.loads((DATA_DIR / paper_id / "chunks.json").read_text())


def post_chat(backend: str, paper_id: str, body: dict, timeout: float = 300.0) -> dict:
    url = f"{backend}/api/papers/{paper_id}/chat"
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _answer_prose(answer: str) -> str:
    """Extract the model's own prose for scoring. ScholAR answers are typically
    '**Answer**\\n<prose>\\n\\n**Evidence**\\n* quotes...'. Score the Answer section
    (the model's words), not the verbatim Evidence quotes which would trivially
    entail. Falls back to the whole text if there is no such structure."""
    m = re.search(r"\*\*\s*Answer\s*\*\*(.*?)(?:\*\*\s*Evidence\s*\*\*|\Z)", answer, re.IGNORECASE | re.DOTALL)
    prose = m.group(1) if m else answer
    return _clean(prose)


def _clean(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)   # unwrap bold
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)  # drop headers
    text = re.sub(r"\[\d+\]", "", text)               # drop inline [n] citations
    return re.sub(r"[ \t]+", " ", text).strip()


def _sentences_with_citations(answer: str) -> list[tuple[str, list[int]]]:
    """Return (sentence, [citation numbers in it]) for each answer sentence that
    carries at least one [n] marker. Used to check citation support."""
    out = []
    for sent in re.split(r"(?<=[.!?])\s+", answer.replace("\n", " ")):
        refs = [int(n) for n in re.findall(r"\[(\d+)\]", sent)]
        if refs and len(_clean(sent)) > 15:
            out.append((_clean(sent), refs))
    return out


def evaluate_case(case: dict, backend: str, model: str) -> dict:
    paper_id = case["paper_id"]
    query = case["query"]
    run = run_scholar_http(
        backend,
        paper_id,
        query,
        model,
        require_local_model=True,
        experiment_id="generation-faithfulness-v1",
    )
    trace = run.trace
    answer = trace.final_answer
    citations = run.citations
    generator_evidence = [
        {"text": item.quote, "global_id": item.identity.global_id}
        for item in trace.prompt_evidence
    ]

    prose = _answer_prose(answer)
    # Score against the exact evidence strings shown to this production generation,
    # not an independently re-run retriever with potentially different budgets.
    gen = _NLI.score_full(prose, generator_evidence, top_k=len(generator_evidence)) if prose and generator_evidence else {
        "cfs": 0.0, "n_atoms": 0, "n_entailed": 0, "n_neutral": 0, "n_contradicted": 0}
    n_atoms = max(gen["n_atoms"], 1)
    hallucination_rate = round(gen["n_contradicted"] / n_atoms, 3)

    # Citation support: for each [n] in the answer, does the cited chunk support
    # the sentence it is attached to? (automated Supported/Partial/Unsupported)
    evidence_text_by_identity = {
        (item.identity.source_id, item.identity.local_id): item.quote
        for item in trace.prompt_evidence
    }
    cit_by_ref = {citation.get("ref_id"): citation for citation in citations}
    cit_labels: list[str] = []
    for sent, refs in _sentences_with_citations(answer):
        for ref in refs:
            cit = cit_by_ref.get(ref)
            if not cit:
                continue
            source_id = cit.get("source_paper_id") or cit.get("document_id") or paper_id
            chunk_text = evidence_text_by_identity.get(
                (str(source_id), str(cit.get("chunk_id") or cit.get("source_evidence_id") or ""))
            ) or cit.get("quote", "")
            if not chunk_text:
                continue
            best = _NLI.score_claim_vs_chunks(sent, [chunk_text], top_k=1)
            label = {"ENTAILMENT": "Supported", "NEUTRAL": "Partial",
                     "CONTRADICTION": "Unsupported"}[best["label"]]
            cit_labels.append(label)

    n_cit = len(cit_labels)
    supported = cit_labels.count("Supported")
    partial = cit_labels.count("Partial")
    unsupported = cit_labels.count("Unsupported")

    return {
        "case_id": case["id"],
        "paper_id": paper_id,
        "query": query,
        "claim_type": case.get("claim_type", "general"),
        "model": model,
        "trace_id": trace.trace_id,
        "trace_schema_version": trace.schema_version,
        "pipeline_version": trace.run_identity.pipeline_version,
        "generation_mode": trace.generation.mode.value,
        "pipeline_status": trace.status.value,
        "abstained": trace.abstention.abstained,
        "answer": answer,
        "citations": citations,
        "prompt_evidence": [item.model_dump(mode="json") for item in trace.prompt_evidence],
        "answer_chars": len(answer),
        "generation_faithfulness": gen["cfs"],   # fraction of answer atoms entailed
        "hallucination_rate": hallucination_rate,
        "n_atoms": gen["n_atoms"],
        "n_entailed": gen["n_entailed"],
        "n_contradicted": gen["n_contradicted"],
        "n_citations_checked": n_cit,
        "citations_supported": supported,
        "citations_partial": partial,
        "citations_unsupported": unsupported,
        "citation_support_rate": round(supported / n_cit, 3) if n_cit else None,
    }


def aggregate(rows: list[dict]) -> dict:
    n = len(rows)
    def avg(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return round(sum(vals) / len(vals), 3) if vals else None
    sup = sum(r["citations_supported"] for r in rows)
    par = sum(r["citations_partial"] for r in rows)
    uns = sum(r["citations_unsupported"] for r in rows)
    tot = sup + par + uns
    return {
        "n_cases": n,
        "mean_generation_faithfulness": avg("generation_faithfulness"),
        "mean_hallucination_rate": avg("hallucination_rate"),
        "total_citations_checked": tot,
        "citation_support_distribution": {
            "supported": sup, "partial": par, "unsupported": uns,
            "supported_pct": round(sup / tot, 3) if tot else None,
        },
        "citation_support_rate_micro": round(sup / tot, 3) if tot else None,
    }


def by_claim_type(rows: list[dict]) -> dict:
    out = {}
    for ct in sorted({r["claim_type"] for r in rows}):
        sub = [r for r in rows if r["claim_type"] == ct]
        vals = [r["generation_faithfulness"] for r in sub if r["generation_faithfulness"] is not None]
        out[ct] = {"n": len(sub), "mean_generation_faithfulness": round(sum(vals) / len(vals), 3) if vals else None}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="http://localhost:8000")
    ap.add_argument("--model", default=OLLAMA_MODEL)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument(
        "--require-encoder",
        action="store_true",
        help="fail instead of silently using lexical fallback when MiniLM is unavailable",
    )
    args = ap.parse_args()

    if args.require_encoder and _EMBEDDER is None:
        sys.exit("MiniLM encoder is required for this run; lexical fallback is not permitted.")

    # preflight
    try:
        with urllib.request.urlopen(f"{args.backend}/health", timeout=10) as r:
            if not json.loads(r.read().decode()).get("ollama_available"):
                sys.exit("Backend up but Ollama unavailable.")
    except Exception as exc:
        sys.exit(f"Cannot reach backend at {args.backend} (run `make backend`): {exc}")

    cases = json.loads(CASES_PATH.read_text())
    if args.limit:
        cases = cases[: args.limit]

    rows = []
    for i, case in enumerate(cases, start=1):
        try:
            row = evaluate_case(case, args.backend, args.model)
            print(f"[{i}/{len(cases)}] {case['id']}: gen_faith={row['generation_faithfulness']} "
                  f"halluc={row['hallucination_rate']} cit_support={row['citation_support_rate']}")
        except Exception as exc:
            print(f"[{i}/{len(cases)}] {case['id']}: ERROR {type(exc).__name__}: {exc}")
            row = {
                "case_id": case["id"],
                "paper_id": case["paper_id"],
                "query": case["query"],
                "claim_type": case.get("claim_type", "general"),
                "model": args.model,
                "pipeline_status": "ERROR",
                "error": f"{type(exc).__name__}: {exc}",
                "answer_chars": 0,
                "generation_faithfulness": 0.0,
                "hallucination_rate": None,
                "n_atoms": 0,
                "n_entailed": 0,
                "n_contradicted": 0,
                "n_citations_checked": 0,
                "citations_supported": 0,
                "citations_partial": 0,
                "citations_unsupported": 0,
                "citation_support_rate": None,
            }
        rows.append(row)

    if not rows:
        sys.exit("No cases evaluated.")

    summary = aggregate(rows)
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "model": args.model,
        "nli_mode": _NLI._mode,
        "measures": "faithfulness of GENERATED answers (not gold claims); "
                    "this scorer evaluates generated answers against their recorded context",
        "summary": summary,
        "by_claim_type": by_claim_type(rows),
        "rows": rows,
    }
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    _write_report(payload)
    print(f"\nWrote {RESULTS_JSON}\nWrote {REPORT_MD}")
    print(json.dumps(summary, indent=2))


def _write_report(p: dict) -> None:
    s = p["summary"]
    d = s["citation_support_distribution"]
    L = [
        "# ScholAR Generation-Faithfulness Report",
        "",
        f"Model: `{p['model']}`. NLI mode: {p['nli_mode']}. Cases: {s['n_cases']}.",
        "",
        "This measures the faithfulness of ScholAR's **generated answers**, not of a "
        "pre-written gold claim. It answers: does the generated answer stay grounded in "
        "the retrieved context, and are its inline citations actually supporting? "
        "Contrast with the retrieval-support CFS in `faithfulness_eval_report_v3.md`, "
        "which scores gold claims against retrieval.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Mean generation-faithfulness (answer atoms entailed) | {s['mean_generation_faithfulness']} |",
        f"| Mean hallucination rate (answer atoms contradicted) | {s['mean_hallucination_rate']} |",
        f"| Citation-support rate (Supported / all checked) | {s['citation_support_rate_micro']} |",
        f"| Citations checked (Supported / Partial / Unsupported) | {d['supported']} / {d['partial']} / {d['unsupported']} |",
        "",
        "## Generation faithfulness by claim type",
        "",
        "| Claim type | N | Mean generation-faithfulness |",
        "|---|---:|---:|",
    ]
    for ct, v in p["by_claim_type"].items():
        L.append(f"| {ct} | {v['n']} | {v['mean_generation_faithfulness']} |")
    REPORT_MD.write_text("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
