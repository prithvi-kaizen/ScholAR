"""
run_comparison_eval.py
Fair, local, apples-to-apples comparison that isolates ScholAR's contribution.

Five systems, all on the SAME local model and SAME cases, differing only in
controlled dimensions (chunking, RCS on/off, citation mechanism):

  1. pdfchat     : whole paper in context, free-form [p. N] citations       (Q1 baseline)
  2. vanilla_rag : fixed cross-page chunks + dense retrieval, free-form cites (Q2 baseline)
  3. paperqa2    : retrieval + RCS (summarize+score rerank), free-form cites  (SOTA method, local)
  4. scholar     : page-preserving retrieval + indirect (evidence-ID) cites  (ScholAR)
  5. scholar_rcs_intervention : approximate ScholAR+RCS intervention (not production)

3 and 5 share the SAME RCS step; they differ ONLY in the citation mechanism, so the
pair isolates "the citation guarantee" from "the retrieval technique".

Metrics (beyond RAG+citation):
  - citation-hallucination rate: cited page out of range, or the citing sentence is
    not entailed by that page's text (NLI). ScholAR cites a retrieved chunk's own
    page, so its page is always in range (invalid-page rate 0 by construction).
  - answer correctness: must_include recall (diverse cases).
  - generation faithfulness: the shared NLI-CFS scorer.
  - citation F1: cited-page vs the gold supporting chunk's page (labeled cases).

Resumable: results are saved per (system, case); rerun to continue. Needs Ollama
running and the anchor papers prepared. Model calls go straight through generate()
(no backend needed).

Run from repo root:
    python3 evaluation/run_comparison_eval.py [--systems a,b] [--cases labeled|diverse|both] [--limit N] [--model M]
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from backend.services.retrieval_service import retrieve_chunks  # noqa: E402
from backend.services.ollama_service import OLLAMA_MODEL, generate  # noqa: E402
from scholar_runner import run_scholar_http  # noqa: E402
import nli_faithfulness as _nli  # noqa: E402

DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_JSON = RESULTS_DIR / "comparison_results.json"
REPORT_MD = RESULTS_DIR / "comparison_report.md"

try:
    from embedder import LocalEmbedder
    _EMBEDDER = LocalEmbedder()
except Exception as exc:  # pragma: no cover
    _EMBEDDER = None
    print(f"[embedder] unavailable ({exc}); vanilla_rag + NLI use lexical fallback")
_nli.set_embedder(_EMBEDDER)
_NLI = _nli.NLIFaithfulnessScorer()

ALL_SYSTEMS = ["pdfchat", "vanilla_rag", "paperqa2", "scholar", "scholar_rcs_intervention"]
TOP_K = 4            # chunks fed to the answer prompt (kept small for local compute)
CTX_CHARS = 8000     # context budget for the PDF-chat baseline


# ── data helpers ──────────────────────────────────────────────────────────
def load_json(paper_id: str, name: str) -> Any:
    return json.loads((DATA_DIR / paper_id / name).read_text())


def page_text(pages: list[dict], n: int) -> str:
    for p in pages:
        if p.get("page") == n:
            return p.get("text", "")
    return pages[n - 1].get("text", "") if 1 <= n <= len(pages) else ""


def num_pages(pages: list[dict]) -> int:
    return max((p.get("page", 0) for p in pages), default=len(pages))


def fixed_chunks(pages: list[dict], size: int = 1200, overlap: int = 150) -> list[dict]:
    """Naive cross-page chunker for the vanilla-RAG baseline: concatenate the whole
    document and slide a fixed character window, ignoring page boundaries. Each chunk's
    page hint is the page its midpoint falls in (deliberately imprecise, which is the
    weakness page-preserving chunking fixes)."""
    spans, offset = [], 0
    for p in pages:
        t = p.get("text", "")
        spans.append((offset, offset + len(t), p.get("page")))
        offset += len(t) + 1
    full = "\n".join(p.get("text", "") for p in pages)
    def page_at(pos: int) -> int:
        for a, b, pg in spans:
            if a <= pos < b:
                return pg
        return spans[-1][2] if spans else 1
    out, start = [], 0
    while start < len(full):
        end = min(start + size, len(full))
        out.append({"text": full[start:end], "page": page_at((start + end) // 2)})
        if end >= len(full):
            break
        start = end - overlap
    return out


# ── citation parsing ──────────────────────────────────────────────────────
_PCITE = re.compile(r"\[\s*p\.?\s*(\d+)\s*\]|\(\s*pages?\s*(\d+)(?:\s*[-–]\s*\d+)?\s*\)", re.I)
# Models group identifiers ("[E1, E4]") as often as they emit them singly; a single-id pattern
# would count those sentences as uncited and understate citation recall.
_ECITE = re.compile(r"\[\s*E\s*(\d+)(?:\s*(?:,|;|/|&|and)\s*E?\s*\d+)*\s*\]", re.I)
_NCITE = re.compile(r"\[\s*(\d+)\s*\]")
_EIDS = re.compile(r"\d+")
_SENT = re.compile(r"(?<=[.!?])\s+")


def _clean(text: str) -> str:
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    text = re.sub(r"\[\s*(?:p\.?\s*\d+|E\s*\d+)\s*\]", "", text, flags=re.I)
    text = re.sub(r"\(\s*pages?\s*\d+(?:\s*[-–]\s*\d+)?\s*\)", "", text, flags=re.I)
    return re.sub(r"[ \t]+", " ", text).strip()


def sentences_with(answer: str, pattern: re.Pattern, all_ids: bool = False) -> list[tuple[str, list[int]]]:
    # Collapse the space inside "[p. 3]" -> "[p.3]" so the period inside the citation
    # is not treated as a sentence boundary (which would split the citation in two).
    answer = re.sub(r"\[\s*p\.?\s*(\d+)\s*\]", r"[p.\1]", answer, flags=re.I)
    out = []
    for sent in _SENT.split(answer.replace("\n", " ")):
        if all_ids:
            # "[E1, E4]" carries two citations; take every identifier in the bracket, not just
            # the first capture group, or the second one is silently dropped.
            ids = [int(n) for m in pattern.finditer(sent) for n in _EIDS.findall(m.group(0))]
        else:
            ids = [int(next(g for g in m.groups() if g)) for m in pattern.finditer(sent)]
        if ids and len(_clean(sent)) > 12:
            out.append((_clean(sent), ids))
    return out


# ── model call ────────────────────────────────────────────────────────────
async def gen(prompt: str, model: str) -> str:
    try:
        return await generate(prompt, temperature=0.1, model=model)
    except Exception as exc:
        return f"[GENERATION ERROR: {type(exc).__name__}: {exc}]"


FREEFORM_RULES = (
    "Answer the question using ONLY the context below. Keep it under 180 words. "
    "After each factual claim, cite the source page in square brackets like [p. 3]. "
    "Do not cite a page that is not in the context."
)
INDIRECT_RULES = (
    "Answer the question using ONLY the evidence below. Keep it under 180 words. "
    "After each factual claim, cite the supporting evidence by its identifier like [E1]. "
    "Cite only evidence identifiers that appear in the list; never write a page number yourself."
)


def _evidence_block(chunks: list[dict]) -> str:
    return "\n".join(
        f"[E{i+1}] (p. {c.get('page')}): {c.get('text','')[:600]}" for i, c in enumerate(chunks)
    )


def _page_block(chunks: list[dict]) -> str:
    return "\n".join(f"[p. {c.get('page')}] {c.get('text','')[:600]}" for i, c in enumerate(chunks))


# ── RCS: rerank + contextual summarization (PaperQA2's core), local ────────
async def rcs(chunks: list[dict], question: str, model: str) -> list[dict]:
    scored = []
    for c in chunks:
        prompt = (
            f"Excerpt:\n{c.get('text','')[:900]}\n\nQuestion: {question}\n\n"
            "Summarize only the information in the excerpt that helps answer the question, "
            "then rate its relevance 0-10. Reply as JSON: "
            '{\"summary\": \"...\", \"relevance\": <int>}.'
        )
        raw = await gen(prompt, model)
        m = re.search(r"\{.*\}", raw, re.S)
        summary, score = c.get("text", ""), 0
        if m:
            try:
                obj = json.loads(m.group(0))
                summary = obj.get("summary") or summary
                score = int(obj.get("relevance", 0))
            except (json.JSONDecodeError, ValueError, TypeError):
                pass
        scored.append({**c, "text": summary, "relevance": score})
    scored.sort(key=lambda x: x["relevance"], reverse=True)
    return scored[:TOP_K]


# ── the five systems: each returns (answer, cited_sentences, context_chunks, mode) ──
async def run_system(system: str, case: dict, model: str, backend: str = "http://127.0.0.1:8000") -> dict:
    paper_id = case["paper_id"]
    question = case["question"]
    pages = load_json(paper_id, "pages.json")
    chunks = load_json(paper_id, "chunks.json")
    trace_metadata: dict[str, Any] = {}

    if system == "pdfchat":
        body = "\n".join(f"[p. {p.get('page')}] {p.get('text','')}" for p in pages)[:CTX_CHARS]
        prompt = f"{FREEFORM_RULES}\n\nContext (the paper):\n{body}\n\nQuestion: {question}"
        answer, mode, ctx = await gen(prompt, model), "freeform", chunks[:TOP_K]
    elif system == "vanilla_rag":
        fc = fixed_chunks(pages)
        from hybrid_retrieval import _dense_ranked  # reuse dense retrieval
        ranked = _dense_ranked(question, fc, _EMBEDDER) if _EMBEDDER else [(0.0, i) for i in range(len(fc))]
        ctx = [fc[i] for _, i in ranked[:TOP_K]]
        prompt = f"{FREEFORM_RULES}\n\nContext:\n{_page_block(ctx)}\n\nQuestion: {question}"
        answer, mode = await gen(prompt, model), "freeform"
    elif system == "paperqa2":
        ctx = await rcs(retrieve_chunks(question, chunks, limit=6), question, model)
        prompt = f"{FREEFORM_RULES}\n\nContext:\n{_page_block(ctx)}\n\nQuestion: {question}"
        answer, mode = await gen(prompt, model), "freeform"
    elif system == "scholar":
        result = await asyncio.to_thread(
            run_scholar_http,
            backend,
            paper_id,
            question,
            model,
            require_local_model=True,
            experiment_id="comparison-scholar-v1",
        )
        answer = result.answer
        ctx = [
            {"text": item.quote, "page": item.page, "global_id": item.identity.global_id}
            for item in result.trace.prompt_evidence
        ]
        mode = "numeric_indirect"
        trace_metadata = {
            "trace_id": result.trace.trace_id,
            "trace_schema_version": result.trace.schema_version,
            "pipeline_version": result.trace.run_identity.pipeline_version,
            "generation_mode": result.trace.generation.mode.value,
        }
    elif system in {"scholar_rcs", "scholar_rcs_intervention"}:
        ctx = await rcs(retrieve_chunks(question, chunks, limit=6), question, model)
        prompt = f"{INDIRECT_RULES}\n\nEvidence:\n{_evidence_block(ctx)}\n\nQuestion: {question}"
        answer, mode = await gen(prompt, model), "indirect"
    else:
        raise ValueError(system)

    # resolve cited page numbers
    if mode == "indirect":
        cited = [(sentence, [ctx[evidence_id - 1].get("page") for evidence_id in evidence_ids if 1 <= evidence_id <= len(ctx)])
                 for sentence, evidence_ids in sentences_with(answer, _ECITE, all_ids=True)]
    elif mode == "numeric_indirect":
        cited = [(sentence, [ctx[ref_id - 1].get("page") for ref_id in ref_ids if 1 <= ref_id <= len(ctx)])
                 for sentence, ref_ids in sentences_with(answer, _NCITE, all_ids=True)]
    else:
        cited = sentences_with(answer, _PCITE)
    return {"answer": answer, "cited": cited, "ctx": ctx, "mode": mode,
            "n_pages": num_pages(pages), "pages": pages, **trace_metadata}


# ── scoring (pure; recomputable offline from stored raw rows via --rescore) ─
_REFUSAL = re.compile(
    r"does not contain|not provided|cannot (be )?determin|is not (explicitly )?"
    r"(state|mention|includ|list|available|present)|no (specific|corresponding|explicit|"
    r"numerical|such|clear)", re.I)


def must_include_recall(answer: str, must: list[str] | None) -> float | None:
    if not must:
        return None
    a = re.sub(r"\s+", " ", answer.lower())
    hit = sum(1 for m in must if re.sub(r"\s+", " ", str(m).lower()) in a)
    return round(hit / len(must), 3)


def substantive_sentences(answer: str) -> list[tuple[str, bool]]:
    """Factual sentences and whether each carries a citation. Refusal sentences make
    no claim, so they are not citation-worthy and are excluded from the recall base."""
    ans = re.sub(r"\[\s*p\.?\s*(\d+)\s*\]", r"[p.\1]", answer, flags=re.I)
    out = []
    for sent in _SENT.split(ans.replace("\n", " ")):
        c = _clean(sent)
        if len(c) <= 40 or _REFUSAL.search(sent):
            continue
        out.append((c, bool(_PCITE.search(sent) or _ECITE.search(sent) or _NCITE.search(sent))))
    return out


def score_row(raw: dict) -> dict:
    """All metrics from a stored raw row. Only NLI runs; no generation. Citation quality
    follows ALCE/OpenScholar: precision = supported/cited citations, recall = fraction of
    factual sentences that carry a citation, so a system that simply under-cites pays in
    recall (this de-confounds the earlier raw hallucination rate)."""
    pages = load_json(raw["paper_id"], "pages.json")
    n_pages = raw["n_pages"]
    answer = raw["answer"]

    prose = _clean(answer)
    ctx_chunks = [{"text": t} for t in raw.get("ctx_texts", [])]
    faith = _NLI.score_full(prose, ctx_chunks, top_k=TOP_K).get("cfs", 0.0) if prose else 0.0

    total = supported = invalid = 0
    cited_pages = set()
    for sent, cpages in raw["cited"]:
        for pg in cpages:
            if pg is None:
                continue
            total += 1
            cited_pages.add(pg)
            if not (isinstance(pg, int) and 1 <= pg <= n_pages):
                invalid += 1
                continue
            if _NLI.score_claim_vs_chunks(sent, [page_text(pages, pg)], top_k=1)["label"] == "ENTAILMENT":
                supported += 1
    precision = round(supported / total, 3) if total else None

    subs = substantive_sentences(answer)
    recall = round(sum(1 for _, c in subs if c) / len(subs), 3) if subs else None
    f1 = (round(2 * precision * recall / (precision + recall), 3)
          if precision and recall else (0.0 if precision is not None and recall is not None else None))

    gold = raw.get("gold_page")
    return {
        "gen_faithfulness": faith,
        "must_include_recall": must_include_recall(answer, raw.get("gold_must_include")),
        "n_citations": total,
        "n_substantive": len(subs),
        "citation_precision": precision,
        "citation_recall": recall,
        "citation_f1": f1,
        "invalid_page_rate": round(invalid / total, 3) if total else None,
        "hallucination_rate": round(1 - supported / total, 3) if total else None,
        "gold_cite_recall": (1.0 if gold in cited_pages else 0.0) if gold is not None else None,
    }


# ── driver (resumable) ────────────────────────────────────────────────────
def load_cases(which: str, limit: int) -> list[dict]:
    out = []
    if which in ("labeled", "both"):
        for c in json.loads((EVAL_DIR / "faithfulness_cases.json").read_text()):
            out.append({"id": c["id"], "paper_id": c["paper_id"], "question": c["query"],
                        "source": "labeled", "supporting_chunk_id": c.get("supporting_chunk_id")})
    if which in ("diverse", "both"):
        div = json.loads((EVAL_DIR / "human_eval" / "cases.json").read_text())
        if limit:
            div = div[:limit]
        for c in div:
            out.append({"id": c["case_id"], "paper_id": c["paper_id"], "question": c["question"],
                        "source": "diverse", "must_include": c.get("must_include")})
    return out


def gold_page_for(case: dict) -> int | None:
    sid = case.get("supporting_chunk_id")
    if not sid:
        return None
    for c in load_json(case["paper_id"], "chunks.json"):
        if c.get("chunk_id") == sid:
            return c.get("page")
    return None


def raw_row(system: str, case: dict, res: dict) -> dict:
    """Everything needed to re-score offline without re-generating."""
    return {
        "system": system, "case_id": case["id"], "source": case["source"],
        "paper_id": case["paper_id"], "answer": res["answer"],
        "cited": [[s, list(ps)] for s, ps in res["cited"]],
        "ctx_texts": [c.get("text", "") for c in res["ctx"]],
        "n_pages": res["n_pages"],
        "gold_must_include": case.get("must_include"),
        "gold_page": gold_page_for(case) if case["source"] == "labeled" else None,
        **{
            key: res[key]
            for key in ("trace_id", "trace_schema_version", "pipeline_version", "generation_mode")
            if key in res
        },
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(ALL_SYSTEMS))
    ap.add_argument("--cases", default="both", choices=["labeled", "diverse", "both"])
    ap.add_argument("--limit", type=int, default=40, help="max diverse cases")
    ap.add_argument("--model", default="qwen3.5:9b")
    ap.add_argument("--backend", default="http://127.0.0.1:8000")
    ap.add_argument("--rescore", action="store_true",
                    help="recompute all metrics from stored raw rows; no model calls")
    args = ap.parse_args()

    prior = json.loads(RESULTS_JSON.read_text()) if RESULTS_JSON.exists() else {}
    rows: dict = prior.get("rows", {})

    if not args.rescore:
        systems = [s.strip() for s in args.systems.split(",") if s.strip()]
        cases = load_cases(args.cases, args.limit)
        for si, system in enumerate(systems, 1):
            for ci, case in enumerate(cases, 1):
                key = f"{system}::{case['id']}"
                if key in rows and "answer" in rows[key]:
                    continue
                try:
                    res = await run_system(system, case, args.model, args.backend)
                    rows[key] = raw_row(system, case, res)
                except Exception as exc:
                    print(f"  [{system}] {case['id']}: ERROR {type(exc).__name__}: {exc}")
                    continue
                print(f"[{si}/{len(systems)} {system}] [{ci}/{len(cases)} {case['id']}] generated")
                _save(rows, args.model)

    # scoring pass (always): recompute metrics from raw, merge back in
    for key, raw in list(rows.items()):
        if "answer" in raw:
            rows[key] = {**raw, **score_row(raw)}

    _save(rows, args.model)
    _write_report(rows)
    print(f"\nWrote {RESULTS_JSON}\nWrote {REPORT_MD}")
    _print_summary(rows)


_METRIC_KEYS = ["gen_faithfulness", "citation_precision", "citation_recall", "citation_f1",
                "hallucination_rate", "invalid_page_rate", "must_include_recall",
                "gold_cite_recall", "n_citations", "n_substantive"]


def _agg(rows: dict) -> dict:
    systems: dict = {}
    for r in rows.values():
        if "gen_faithfulness" not in r:
            continue
        s = systems.setdefault(r["system"], {k: [] for k in _METRIC_KEYS})
        for k in _METRIC_KEYS:
            if r.get(k) is not None:
                s[k].append(r[k])
    def mean(xs): return round(sum(xs) / len(xs), 3) if xs else None
    return {sys: {"n": len(v["gen_faithfulness"]), **{k: mean(v[k]) for k in _METRIC_KEYS}}
            for sys, v in systems.items()}


def _save(rows: dict, model: str) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_JSON.write_text(json.dumps(
        {"generated_at": datetime.now().isoformat(timespec="seconds"), "model": model,
         "nli_mode": _NLI._mode, "summary": _agg(rows), "rows": rows}, indent=2, ensure_ascii=False))


def _write_report(rows: dict) -> None:
    agg = _agg(rows)
    L = ["# ScholAR vs Local Baselines (same model, same cases)", "",
         "Citation quality is ALCE/OpenScholar-style: precision = supported / cited citations, "
         "recall = fraction of factual sentences that carry a citation.", "",
         "| System | N | Gen-faith | Cite-P | Cite-R | Cite-F1 | Must-incl | Gold-cite-R | Invalid-page |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for s in ALL_SYSTEMS:
        if s in agg:
            a = agg[s]
            L.append(f"| {s} | {a['n']} | {a['gen_faithfulness']} | {a['citation_precision']} | "
                     f"{a['citation_recall']} | {a['citation_f1']} | {a['must_include_recall']} | "
                     f"{a['gold_cite_recall']} | {a['invalid_page_rate']} |")
    REPORT_MD.write_text("\n".join(L) + "\n")


def _print_summary(rows: dict) -> None:
    for s in ALL_SYSTEMS:
        a = _agg(rows).get(s)
        if a:
            print(f"  {s:<12} n={a['n']} faith={a['gen_faithfulness']} "
                  f"cite_P={a['citation_precision']} cite_R={a['citation_recall']} "
                  f"cite_F1={a['citation_f1']} mi={a['must_include_recall']} goldcite={a['gold_cite_recall']}")


def _selfcheck() -> None:
    # citation detection: factual sentences flagged for citation presence; refusals excluded
    subs = substantive_sentences(
        "The model uses BM25 retrieval with a k1 value of 1.4 [p. 2]. "
        "It was trained on a large web corpus without any human labels. "
        "The context does not contain the learning rate.")
    assert len(subs) == 2, subs                      # refusal sentence dropped
    assert subs[0][1] is True and subs[1][1] is False, subs
    assert must_include_recall("uses bm25 retrieval", ["BM25"]) == 1.0
    assert must_include_recall("no match here", ["BM25"]) == 0.0
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        asyncio.run(main())
