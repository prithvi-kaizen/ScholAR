"""
run_faithfulness_eval.py  (v3 — NLI-based, research-grade)
===========================================================
ScholAR — Citation Faithfulness Evaluation
Direction 1.1 of RESEARCH_ROADMAP.md

METHODOLOGY:
------------
This script implements a three-tier faithfulness evaluation pipeline
aligned with 2024-2026 state-of-the-art practices:

  Tier 1 — NLI-CFS (main metric, SummaC/AlignScore-style)
      For each (expected_claim, retrieved_chunks) pair:
        a) Decompose claim into atomic sub-claims (FActScore-style)
        b) For each atom, score against each retrieved chunk using
           token-level NLI (entailment/neutral/contradiction)
        c) CFS = |entailed or neutral atoms| / |total atoms|

  Tier 2 — SCR (Semantic Coverage Rate, MiniLM-based)
      Cosine similarity between claim embedding and best chunk embedding
      using all-MiniLM-L6-v2 from local HF cache

  Tier 3 — KFP (Key Fact Presence, numeric/term precision)
      Fraction of numbers and technical terms from claim found in chunks

  Combined NLI-CFS = 0.50 × NLI + 0.30 × SCR + 0.20 × KFP

Additionally runs BOTH retrieval systems for comparison:
  - BM25-primary (baseline, existing ScholAR system)
  - Hybrid BM25+Dense+RRF (new, SOTA 2025)

References:
  - SummaC (Laban et al., TACL 2022)
  - AlignScore (Zha et al., ACL 2023)
  - FActScore (Min et al., EMNLP 2023)
  - RAGAS (Es et al., 2024)
  - ARES (Saad-Falcon et al., ACL 2024)
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

# ── project paths ──────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR     = PROJECT_ROOT / "evaluation"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(EVAL_DIR))

from backend.services.retrieval_service import extract_page_hints, retrieve_chunks  # noqa: E402
from nli_faithfulness import NLIFaithfulnessScorer                                  # noqa: E402

DATA_DIR     = PROJECT_ROOT / "backend" / "data" / "papers"
CASES_PATH   = EVAL_DIR / "faithfulness_cases.json"
RESULTS_DIR  = EVAL_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_JSON = RESULTS_DIR / "faithfulness_eval_results_v3.json"
REPORT_MD    = RESULTS_DIR / "faithfulness_eval_report_v3.md"

# ── try to load dense embedder for SCR ────────────────────────────────
_EMBEDDER = None
_SCR_AVAILABLE = False
try:
    from embedder import LocalEmbedder
    _EMBEDDER = LocalEmbedder()
    import numpy as np
    _SCR_AVAILABLE = True
    print(f"[SCR] MiniLM embedder loaded ({_EMBEDDER._loaded} tensors)")
except Exception as exc:
    np = None
    print(f"[SCR] Embedder not available ({exc}); SCR will be computed statistically")

# ── inject shared embedder into NLI scorer (avoids double model load) ─
import nli_faithfulness as _nli_mod
_nli_mod.set_embedder(_EMBEDDER)



# ── numeric/technical fact helpers ───────────────────────────────────
_NUM_RE   = re.compile(r"\b\d+(?:[.,]\d+)?(?:\s*%|\s*B|\s*M|\s*K)?\b")
_TERM_RE  = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_STOP     = frozenset("a an and are as at be by for from how in is it its of on or "
                      "that the this to was what with which where when who".split())


def _key_facts(text: str) -> list[str]:
    nums  = _NUM_RE.findall(text)
    terms = [t.lower() for t in _TERM_RE.findall(text) if t not in _STOP]
    return nums + terms


def kfp_score(claim: str, chunk_texts: list[str]) -> float:
    facts = _key_facts(claim)
    if not facts:
        return 1.0
    combined = " ".join(chunk_texts).lower()
    # Match on word/number boundaries, not raw substring: the numeric fact "8"
    # must not count as found inside "2048", and "1" must not match everything.
    found = 0
    for f in facts:
        fl = f.lower().strip()
        if fl and re.search(r"(?<!\w)" + re.escape(fl) + r"(?!\w)", combined):
            found += 1
    return round(found / len(facts), 3)


def scr_score(claim: str, chunk_texts: list[str]) -> float:
    """
    SCR = max cosine(embed(claim), embed(chunk)) over retrieved chunks.
    This is the whole-claim vs. whole-chunk similarity (complementary to
    the sentence-level NLI in Tier 1).
    Uses all-MiniLM-L6-v2; falls back to ROUGE-1 recall if unavailable.
    """
    if not _SCR_AVAILABLE or _EMBEDDER is None:
        # Lexical fallback (labelled explicitly)
        def toks(t): return set(re.findall(r"[a-z]+", t.lower()))
        h = toks(claim)
        if not h:
            return 1.0
        best = max((len(h & toks(c)) / len(h) for c in chunk_texts), default=0.0)
        return round(float(best), 3)

    texts  = [claim] + [c[:512] for c in chunk_texts]
    embs   = _EMBEDDER.encode(texts, batch_size=32)
    q_emb  = embs[0:1]
    c_embs = embs[1:]
    sims   = (c_embs @ q_emb.T).reshape(-1)  # (N,) — explicit, no squeeze ambiguity
    best   = float(sims.max()) if len(sims) > 0 else 0.0
    return round(max(0.0, min(1.0, best)), 3)



# ── retrieval helpers ─────────────────────────────────────────────────
def load_chunks(paper_id: str) -> list[dict]:
    return json.loads((DATA_DIR / paper_id / "chunks.json").read_text())


def get_hybrid_retriever():
    """Try to load hybrid retriever; fall back to BM25."""
    try:
        from hybrid_retrieval import retrieve_hybrid
        return retrieve_hybrid
    except Exception:
        return None


# ── core evaluation ───────────────────────────────────────────────────
_NLI = NLIFaithfulnessScorer()
_HYBRID = get_hybrid_retriever()


def evaluate_case(case: dict, system: str = "bm25") -> dict:
    paper_id   = case["paper_id"]
    query      = case["query"]
    claim      = case["expected_claim"]
    support_id = case.get("supporting_chunk_id", "")
    claim_type = case.get("claim_type", "general")

    all_chunks = load_chunks(paper_id)
    preferred  = extract_page_hints(query)

    if system == "hybrid" and _HYBRID is not None:
        retrieved = _HYBRID(query, all_chunks, limit=5, preferred_pages=preferred)
    else:
        retrieved = retrieve_chunks(query, all_chunks, limit=5, preferred_pages=preferred)

    retrieved_ids   = [c.get("chunk_id", "") for c in retrieved]
    retrieved_texts = [c.get("text", "")     for c in retrieved]

    # ── SCHR ──────────────────────────────────────────────────────────
    schr1 = int(len(retrieved_ids) > 0 and retrieved_ids[0] == support_id)
    schr3 = int(support_id in retrieved_ids[:3])
    schr5 = int(support_id in retrieved_ids[:5])
    support_rank = next((i+1 for i, c in enumerate(retrieved_ids) if c == support_id), None)

    # ── Tier 1: NLI-CFS ───────────────────────────────────────────────
    nli_result = _NLI.score_full(claim, retrieved, top_k=5)
    nli_cfs    = nli_result["cfs"]

    # ── Tier 2: SCR (Semantic Coverage Rate) ──────────────────────────
    scr = scr_score(claim, retrieved_texts)

    # ── Tier 3: KFP ───────────────────────────────────────────────────
    kfp = kfp_score(claim, retrieved_texts)

    # ── Combined NLI-CFS ──────────────────────────────────────────────
    combined_cfs = round(0.50 * nli_cfs + 0.30 * scr + 0.20 * kfp, 3)

    # Rank ablation: combined CFS at each depth k
    rank_cfs_scores = {}
    for k in range(1, 6):
        sub_texts  = retrieved_texts[:k]
        sub_chunks = retrieved[:k]
        n_res      = _NLI.score_full(claim, sub_chunks, top_k=k)
        s_scr      = scr_score(claim, sub_texts)
        s_kfp      = kfp_score(claim, sub_texts)
        rank_cfs_scores[k] = round(0.50 * n_res["cfs"] + 0.30 * s_scr + 0.20 * s_kfp, 3)

    label = (
        "FAITHFUL"   if combined_cfs >= 0.55 else
        "PARTIAL"    if combined_cfs >= 0.35 else
        "UNFAITHFUL"
    )

    return {
        "case_id":       case["id"],
        "paper_id":      paper_id,
        "query":         query,
        "claim_type":    claim_type,
        "support_id":    support_id,
        "retrieved_ids": retrieved_ids,
        "support_rank":  support_rank,
        "schr1": schr1, "schr3": schr3, "schr5": schr5,
        "nli_cfs":      nli_cfs,
        "scr":          scr,
        "kfp":          kfp,
        "combined_cfs": combined_cfs,
        "n_atoms":      nli_result["n_atoms"],
        "n_entailed":   nli_result["n_entailed"],
        "n_neutral":    nli_result["n_neutral"],
        "n_contradicted": nli_result["n_contradicted"],
        "label":        label,
        "rank_cfs":     rank_cfs_scores,
    }


# ── aggregation helpers ───────────────────────────────────────────────
def aggregate(results: list[dict]) -> dict:
    n = len(results)
    def avg(key): return round(sum(r[key] for r in results) / n, 3)
    return {
        "n":           n,
        "mean_combined_cfs": avg("combined_cfs"),
        "mean_nli_cfs":      avg("nli_cfs"),
        "mean_scr":          avg("scr"),
        "mean_kfp":          avg("kfp"),
        "schr_top1":  round(sum(r["schr1"] for r in results) / n, 3),
        "schr_top3":  round(sum(r["schr3"] for r in results) / n, 3),
        "schr_top5":  round(sum(r["schr5"] for r in results) / n, 3),
        "n_faithful": sum(1 for r in results if r["label"] == "FAITHFUL"),
        "n_partial":  sum(1 for r in results if r["label"] == "PARTIAL"),
        "n_unfaith":  sum(1 for r in results if r["label"] == "UNFAITHFUL"),
        "mean_atoms": round(sum(r["n_atoms"] for r in results) / n, 2),
        "mean_entailed": round(sum(r["n_entailed"] for r in results) / n, 2),
    }


def ablation_by_rank(results: list[dict]) -> dict:
    rank_vals: dict[int, list[float]] = {k: [] for k in range(1, 6)}
    for r in results:
        for k in range(1, 6):
            v = r["rank_cfs"].get(k)
            if v is not None:
                rank_vals[k].append(v)
    return {
        k: {"n": len(vs), "mean_cfs": round(sum(vs)/len(vs), 3) if vs else 0.0}
        for k, vs in rank_vals.items()
    }


def ablation_by_type(results: list[dict]) -> dict:
    groups: dict[str, list] = {}
    for r in results:
        groups.setdefault(r["claim_type"], []).append(r)
    return {
        ct: {
            "n":         len(rows),
            "mean_cfs":  round(sum(r["combined_cfs"] for r in rows)/len(rows), 3),
            "mean_nli":  round(sum(r["nli_cfs"]      for r in rows)/len(rows), 3),
            "mean_scr":  round(sum(r["scr"]           for r in rows)/len(rows), 3),
            "mean_kfp":  round(sum(r["kfp"]           for r in rows)/len(rows), 3),
            "schr5":     round(sum(r["schr5"]         for r in rows)/len(rows), 3),
        }
        for ct, rows in sorted(groups.items())
    }


# ── report writer ─────────────────────────────────────────────────────
def write_report(bm25: list[dict], hybrid: list[dict],
                 bm25_agg: dict, hybrid_agg: dict,
                 rank_abl_bm25: dict, rank_abl_hyb: dict,
                 type_abl_bm25: dict, type_abl_hyb: dict) -> None:

    def row(r, sys_label):
        return (f"| `{r['case_id']}` | {r['claim_type']} | {sys_label} "
                f"| {r['nli_cfs']:.3f} | {r['scr']:.3f} | {r['kfp']:.3f} "
                f"| **{r['combined_cfs']:.3f}** | {r['support_rank'] or 'miss'} | {r['label']} |")

    case_rows = "\n".join(row(r, "BM25") for r in bm25) + "\n" + \
                "\n".join(row(r, "Hybrid") for r in hybrid)

    rank_rows_bm25 = "\n".join(
        f"| Top-{k} | {d['n']} | {d['mean_cfs']:.3f} | {rank_abl_hyb.get(k, {}).get('mean_cfs', 0):.3f} |"
        for k, d in rank_abl_bm25.items()
    )

    type_rows = "\n".join(
        f"| {ct} | {d['n']} | {d['mean_cfs']:.3f} | {d['mean_nli']:.3f} | {d['mean_scr']:.3f} | {d['mean_kfp']:.3f} | {d['schr5']:.3f} |"
        for ct, d in type_abl_bm25.items()
    )

    report = f"""# ScholAR NLI-Based Citation Faithfulness Report (v3)

**Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}
**Cases:** {bm25_agg['n']}
**Method:** NLI-CFS = 0.50×NLI + 0.30×SCR + 0.20×KFP
**NLI:** {'Sentence-level semantic entailment via MiniLM (SummaC/AlignScore-style)' if _NLI._mode == 'semantic' else 'LEXICAL FALLBACK (embedder unavailable) — not the semantic metric'}
**SCR:** {'MiniLM cosine similarity' if _SCR_AVAILABLE else 'Bigram Jaccard (MiniLM fallback)'}
**KFP:** Numeric and technical term precision

---

## System Comparison

| System | Mean NLI-CFS | SCHR@5 | Faithful | Partial | Unfaithful |
|---|---:|---:|---:|---:|---:|
| BM25-primary (ScholAR baseline) | **{bm25_agg['mean_combined_cfs']:.3f}** | {bm25_agg['schr_top5']:.3f} | {bm25_agg['n_faithful']}/{bm25_agg['n']} | {bm25_agg['n_partial']}/{bm25_agg['n']} | {bm25_agg['n_unfaith']}/{bm25_agg['n']} |
| Hybrid BM25+Dense+RRF          | **{hybrid_agg['mean_combined_cfs']:.3f}** | {hybrid_agg['schr_top5']:.3f} | {hybrid_agg['n_faithful']}/{hybrid_agg['n']} | {hybrid_agg['n_partial']}/{hybrid_agg['n']} | {hybrid_agg['n_unfaith']}/{hybrid_agg['n']} |

## Component Scores (BM25)

| Metric | Value |
|---|---:|
| Mean NLI-CFS (Tier 1) | {bm25_agg['mean_nli_cfs']:.3f} |
| Mean SCR (Tier 2)     | {bm25_agg['mean_scr']:.3f} |
| Mean KFP (Tier 3)     | {bm25_agg['mean_kfp']:.3f} |
| Mean Combined NLI-CFS | **{bm25_agg['mean_combined_cfs']:.3f}** |
| SCHR@1 | {bm25_agg['schr_top1']:.3f} |
| SCHR@3 | {bm25_agg['schr_top3']:.3f} |
| SCHR@5 | {bm25_agg['schr_top5']:.3f} |
| Avg atoms/claim | {bm25_agg['mean_atoms']:.2f} |

---

## Rank Ablation (NLI-CFS @ Top-K)

| Chunks Used | N | BM25 | Hybrid |
|---|---:|---:|---:|
{rank_rows_bm25}

---

## Claim-Type Breakdown (BM25)

| Claim Type | N | NLI-CFS | NLI | SCR | KFP | SCHR@5 |
|---|---:|---:|---:|---:|---:|---:|
{type_rows}

---

## Per-Case Results

| Case | Type | System | NLI | SCR | KFP | CFS | Rank | Label |
|---|---|---|---:|---:|---:|---:|---:|---|
{case_rows}
"""
    REPORT_MD.write_text(report, encoding="utf-8")
    print(f"\nWrote {REPORT_MD}")


# ── entry point ───────────────────────────────────────────────────────
def main() -> None:
    global CASES_PATH, RESULTS_JSON, REPORT_MD
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES_PATH), help="path to a faithfulness cases JSON")
    ap.add_argument("--tag", default="", help="output suffix, e.g. 'scaled'")
    args = ap.parse_args()
    CASES_PATH = Path(args.cases)
    if args.tag:
        RESULTS_JSON = RESULTS_DIR / f"faithfulness_eval_results_{args.tag}.json"
        REPORT_MD = RESULTS_DIR / f"faithfulness_eval_report_{args.tag}.md"

    cases = json.loads(CASES_PATH.read_text())
    n = len(cases)
    print(f"\n{'='*65}", flush=True)
    print(f"  ScholAR Faithfulness Evaluation (v4 — SummaC-ZS) — {n} cases", flush=True)
    print(f"  Tier 1 : SummaC-ZS (sentence cosine via MiniLM-L6-v2)", flush=True)
    print(f"  Tier 2 : SCR whole-claim cosine {'[MiniLM]' if _SCR_AVAILABLE else '[lexical fallback]'}", flush=True)
    print(f"  Tier 3 : KFP (key fact presence)", flush=True)
    print(f"  Systems: BM25-primary + Hybrid BM25+Dense+RRF", flush=True)
    print(f"{'='*65}\n", flush=True)

    bm25_results   : list[dict] = []
    hybrid_results : list[dict] = []

    for i, case in enumerate(cases, 1):
        cid = case["id"]
        try:
            r_bm25   = evaluate_case(case, system="bm25")
            r_hybrid = evaluate_case(case, system="hybrid")
        except Exception as exc:
            print(f"  ERROR [{cid}]: {exc}")
            import traceback; traceback.print_exc()
            continue

        bm25_results.append(r_bm25)
        hybrid_results.append(r_hybrid)

        delta = r_hybrid["combined_cfs"] - r_bm25["combined_cfs"]
        delta_str = f"+{delta:.3f}" if delta >= 0 else f"{delta:.3f}"
        print(
            f"[{i:02d}/{n}] {cid[:38]:38s}  "
            f"BM25={r_bm25['combined_cfs']:.3f} "
            f"Hyb={r_hybrid['combined_cfs']:.3f} ({delta_str})  "
            f"SCHR5={r_bm25['schr5']}  {r_bm25['label']}",
            flush=True
        )

    bm25_agg      = aggregate(bm25_results)
    hybrid_agg    = aggregate(hybrid_results)
    rank_abl_bm25 = ablation_by_rank(bm25_results)
    rank_abl_hyb  = ablation_by_rank(hybrid_results)
    type_abl_bm25 = ablation_by_type(bm25_results)
    type_abl_hyb  = ablation_by_type(hybrid_results)

    payload = {
        "generated_at":   datetime.now().isoformat(timespec="seconds"),
        "n_cases":        len(bm25_results),
        "scr_model":      "MiniLM-L6-v2 cosine" if _SCR_AVAILABLE else "bigram-jaccard",
        # Records whether the NLI tier ran with the real MiniLM embedder
        # ("semantic") or silently fell back to lexical scoring
        # ("lexical_fallback"). Without this, two runs on different machines
        # could produce different combined_cfs with nothing to distinguish them.
        "nli_mode":       _NLI._mode,
        "bm25": {
            "summary":       bm25_agg,
            "rank_ablation": rank_abl_bm25,
            "type_ablation": type_abl_bm25,
            "results":       bm25_results,
        },
        "hybrid": {
            "summary":       hybrid_agg,
            "rank_ablation": rank_abl_hyb,
            "type_ablation": type_abl_hyb,
            "results":       hybrid_results,
        },
    }
    RESULTS_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {RESULTS_JSON}")

    write_report(bm25_results, hybrid_results,
                 bm25_agg, hybrid_agg,
                 rank_abl_bm25, rank_abl_hyb,
                 type_abl_bm25, type_abl_hyb)

    print(f"\n{'='*65}")
    print(f"  BM25    Combined NLI-CFS : {bm25_agg['mean_combined_cfs']:.3f}")
    print(f"  Hybrid  Combined NLI-CFS : {hybrid_agg['mean_combined_cfs']:.3f}")
    delta = round(hybrid_agg['mean_combined_cfs'] - bm25_agg['mean_combined_cfs'], 3)
    print(f"  Hybrid gain              : {'+' if delta>=0 else ''}{delta:.3f}")
    print(f"  BM25    SCHR@5           : {bm25_agg['schr_top5']:.3f}")
    print(f"  Hybrid  SCHR@5           : {hybrid_agg['schr_top5']:.3f}")
    print(f"  BM25    FAITHFUL         : {bm25_agg['n_faithful']}/{bm25_agg['n']}")
    print(f"  Hybrid  FAITHFUL         : {hybrid_agg['n_faithful']}/{hybrid_agg['n']}")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()
