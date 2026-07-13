"""
build_benchmark_stats.py
Compute the key statistics of ScholAR's corpus and benchmarks, so the paper can carry the
dataset-statistics table that every comparable paper has (M3SciQA Table 1, OpenScholar Table 1,
SciRAG Table 1, ALCE Table 1, SciDQA Tables 1-2) and we currently lack.

Everything here is derived from files already in the repo. Nothing is estimated.

Run from repo root:  python3 evaluation/build_benchmark_stats.py [--selfcheck]
Writes: evaluation/results/benchmark_stats.json
"""
from __future__ import annotations

import json
import statistics
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVAL = ROOT / "evaluation"
PAPERS = ROOT / "backend" / "data" / "papers"
OUT = EVAL / "results" / "benchmark_stats.json"


def _load(p: Path):
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def _mean(xs, nd=1):
    return round(statistics.mean(xs), nd) if xs else None


def corpus_stats(paper_ids: list[str]) -> dict:
    chunks, pages, words = [], [], []
    for pid in paper_ids:
        ch = _load(PAPERS / pid / "chunks.json") or []
        chunks.append(len(ch))
        pages.append(len({c.get("page") for c in ch if c.get("page")}))
        words.append(sum(len(str(c.get("text", "")).split()) for c in ch))
    return {
        "papers": len(paper_ids),
        "chunks_per_paper_mean": _mean(chunks),
        "chunks_per_paper_range": [min(chunks), max(chunks)] if chunks else None,
        "pages_per_paper_mean": _mean(pages),
        "pages_per_paper_range": [min(pages), max(pages)] if pages else None,
        "words_per_paper_mean": int(statistics.mean(words)) if words else None,
    }


def build() -> dict:
    cases = _load(EVAL / "human_eval" / "cases.json") or []
    answers = _load(EVAL / "human_eval" / "answers.json") or []
    paper_ids = sorted({c["paper_id"] for c in cases})

    caps = Counter(c["capability"] for c in cases)
    n_cit = [len(a.get("citations") or []) for a in answers]

    stats = {
        "corpus": corpus_stats(paper_ids),
        "diverse_benchmark": {
            "questions": len(cases),
            "by_capability": dict(caps),
            "question_words_mean": _mean([len(c["question"].split()) for c in cases]),
            "gold_facts_per_question_mean": _mean([len(c.get("must_include") or []) for c in cases]),
        },
        "model_answers": {
            "answers": len(answers),
            "models": sorted({a["model"] for a in answers}),
            "citations_total": sum(n_cit),
            "citations_per_answer_mean": _mean(n_cit),
        },
        "labeled_benchmarks": {
            "retrieval_anchor": len(_load(EVAL / "benchmark_cases.json") or []),
            "faithfulness_anchor": len(_load(EVAL / "faithfulness_cases.json") or []),
            "retrieval_scaled": len(_load(EVAL / "benchmark_cases_scaled.json") or []),
            "faithfulness_scaled": len(_load(EVAL / "faithfulness_cases_scaled.json") or []),
            "visual": len(_load(EVAL / "visual_benchmark.json") or []),
            "multidoc": len(_load(EVAL / "multidoc_benchmark.json") or []),
            "abstention": len(_load(EVAL / "abstention_cases.json") or []),
        },
    }
    return stats


def _selfcheck(s: dict) -> None:
    d = s["diverse_benchmark"]
    assert sum(d["by_capability"].values()) == d["questions"], "capability split must sum to total"
    m = s["model_answers"]
    # 75 text+math cases x 4 models + 25 visual x 2 multimodal = 350
    text_math = d["by_capability"]["single_doc_text"] + d["by_capability"]["math"]
    visual = d["by_capability"]["visual"]
    assert m["answers"] == text_math * 4 + visual * 2, (m["answers"], text_math, visual)
    assert s["corpus"]["papers"] == 25
    print("selfcheck OK")


if __name__ == "__main__":
    s = build()
    if "--selfcheck" in sys.argv:
        _selfcheck(s)
        sys.exit()
    _selfcheck(s)
    OUT.write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    print(f"\nwrote {OUT.relative_to(ROOT)}")
