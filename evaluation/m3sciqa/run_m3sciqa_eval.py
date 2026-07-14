"""
run_m3sciqa_eval.py
Evaluate ScholAR's multi-document localization on M3SciQA's locality task, the external baseline
our multi-doc result previously lacked. The task: given an anchor paper and a question about one
of its figures, rank the anchor's bibliography (mean 47.9 candidates) and find the reference paper
that answers the question. Metric is MRR, matching M3SciQA's Table 2.

Two tiers:
  Tier A (--tier text, no LLM): rank candidates from the question alone, with BM25 (ScholAR's
    production parameters k1=1.4, b=0.72), dense MiniLM, and hybrid BM25+Dense+RRF. Comparable to
    M3SciQA's BM25 / Contriever / text-embedding rows.
  Tier B (--tier vision --model M): the question refers to a FIGURE, so first send that figure to
    the local multimodal model to resolve what the question points at, then rank with
    question + resolved entity. Comparable to M3SciQA's LMM rows.

The random floor is computed, not assumed: M3SciQA reports 0.126, which does not match our pool
size, so we report the exact expected-random MRR for the pools we actually rank.

Run from repo root:
    python3 evaluation/m3sciqa/run_m3sciqa_eval.py --tier text
    python3 evaluation/m3sciqa/run_m3sciqa_eval.py --tier vision --model qwen3.5:9b
Writes: evaluation/results/m3sciqa_results.json (merges per tier/model; resumable)
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "evaluation"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL))

from hybrid_retrieval import _bm25_ranked, _dense_ranked, _rrf, _get_embedder  # noqa: E402
from backend.services.ollama_service import generate  # noqa: E402

CASES = Path(__file__).resolve().parent / "m3sciqa_cases.json"
OUT = EVAL / "results" / "m3sciqa_results.json"

VISION_PROMPT = (
    "Look at the figure from a scientific paper and answer the question about it in at most "
    "12 words. Name the specific model, method, dataset or system the question asks about. "
    "Reply with the answer only.\n\nQuestion: {q}"
)


def pseudo_chunks(candidates: list[dict]) -> list[dict]:
    """Each candidate reference paper becomes one retrievable unit (title + abstract)."""
    return [{"chunk_id": i, "text": f"{c['title']}. {c['abstract']}"}
            for i, c in enumerate(candidates)]


def rank_of(ranked_idx: list[int], gold: int) -> int | None:
    return ranked_idx.index(gold) + 1 if gold in ranked_idx else None


def metrics(ranks: list[int | None], pools: list[int]) -> dict:
    n = len(ranks)
    mrr = sum((1 / r if r else 0.0) for r in ranks) / n
    out = {"n": n, "mrr": round(mrr, 3)}
    for k in (1, 5, 10):
        out[f"recall@{k}"] = round(sum(1 for r in ranks if r and r <= k) / n, 3)
    # expected MRR of a uniformly random ranking over the SAME pools (not assumed from the paper)
    rnd = sum(sum(1 / r for r in range(1, p + 1)) / p for p in pools) / n
    out["random_mrr"] = round(rnd, 3)
    out["mean_pool"] = round(sum(pools) / n, 1)
    return out


def rank_all(query: str, cands: list[dict], embedder) -> dict[str, list[int]]:
    ch = pseudo_chunks(cands)
    bm = [i for _, i in _bm25_ranked(query, ch)]
    out = {"bm25": bm}
    if embedder:
        dn = [i for _, i in _dense_ranked(query, ch, embedder)]
        out["dense"] = dn
        fused = _rrf([bm, dn])
        out["hybrid_rrf"] = [i for i, _ in fused]
    return out


async def vision_entity(case: dict, model: str) -> str:
    fig = ROOT / case["figure_path"] if case.get("figure_path") else None
    if not fig or not fig.exists():
        return ""
    b64 = base64.b64encode(fig.read_bytes()).decode()
    try:
        return (await generate(VISION_PROMPT.format(q=case["question"]),
                               temperature=0.1, images=[b64], model=model)).strip()
    except Exception as exc:
        return f"[ERROR {type(exc).__name__}]"


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=["text", "vision"], default="text")
    ap.add_argument("--model", default="qwen3.5:9b", help="multimodal model (vision tier)")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    cases = json.loads(CASES.read_text())
    if args.limit:
        cases = cases[:args.limit]
    embedder = _get_embedder()

    blob = json.loads(OUT.read_text()) if OUT.exists() else {"summary": {}, "raw": {}}
    blob["generated_at"] = datetime.now().isoformat(timespec="seconds")

    if args.tier == "text":
        ranks: dict[str, list] = {"bm25": [], "dense": [], "hybrid_rrf": []}
        pools = []
        for i, c in enumerate(cases, 1):
            r = rank_all(c["question"], c["candidates"], embedder)
            pools.append(len(c["candidates"]))
            for k in ranks:
                if k in r:
                    ranks[k].append(rank_of(r[k], c["gold_index"]))
            if i % 50 == 0:
                print(f"  [text] {i}/{len(cases)}")
        for k, rk in ranks.items():
            if rk:
                blob["summary"][k] = metrics(rk, pools)
    else:
        rows, ranks, pools = [], [], []
        for i, c in enumerate(cases, 1):
            ent = await vision_entity(c, args.model)
            q = f"{c['question']} {ent}".strip()
            r = rank_all(q, c["candidates"], embedder)["bm25"]
            rk = rank_of(r, c["gold_index"])
            ranks.append(rk)
            pools.append(len(c["candidates"]))
            rows.append({"case_id": c["case_id"], "entity": ent, "rank": rk})
            print(f"  [{args.model}] {i}/{len(cases)} rank={rk} :: {ent[:60]}")
        key = f"vision_bm25::{args.model}"
        blob["summary"][key] = metrics(ranks, pools)
        blob["raw"][key] = rows

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(blob, indent=2))
    print(f"\n{'system':22}{'MRR':>7}{'R@1':>7}{'R@5':>7}{'R@10':>7}{'rand':>7}")
    for k, s in blob["summary"].items():
        print(f"{k:22}{s['mrr']:>7.3f}{s['recall@1']:>7.3f}{s['recall@5']:>7.3f}"
              f"{s['recall@10']:>7.3f}{s['random_mrr']:>7.3f}")
    print(f"\nM3SciQA published (their Table 2): random 0.126 | BM25 0.127 | Contriever 0.184 | "
          f"text-emb-3-large 0.297 | open LMMs 0.056-0.144 | GPT-4o 0.500 | human 0.796")
    print(f"wrote {OUT.relative_to(ROOT)}")


def _selfcheck() -> None:
    cands = [{"title": "A", "abstract": "alpha beta"}, {"title": "B", "abstract": "gamma delta"}]
    ch = pseudo_chunks(cands)
    assert ch[0]["text"].startswith("A."), ch
    assert rank_of([1, 0], 0) == 2 and rank_of([1, 0], 1) == 1
    m = metrics([1, 2, None], [10, 10, 10])
    assert m["mrr"] == round((1 + 0.5 + 0) / 3, 3), m
    assert m["recall@1"] == round(1 / 3, 3) and m["recall@5"] == round(2 / 3, 3)
    # random floor for a pool of 10 is H_10/10 ~= 0.293
    assert 0.29 <= m["random_mrr"] <= 0.30, m["random_mrr"]
    print("selfcheck OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
    else:
        asyncio.run(main())
