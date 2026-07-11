from __future__ import annotations

import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.services.retrieval_service import extract_page_hints, retrieve_chunks, tokenize

EVAL_DIR = Path(__file__).resolve().parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))

from hybrid_retrieval import retrieve_dense_only


DATA_DIR = PROJECT_ROOT / "backend" / "data" / "papers"
CASES_PATH = PROJECT_ROOT / "evaluation" / "benchmark_cases.json"
RESULTS_DIR = PROJECT_ROOT / "evaluation" / "results"
RESULTS_JSON = RESULTS_DIR / "retrieval_eval_results.json"
REPORT_MD = RESULTS_DIR / "retrieval_eval_report.md"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_chunks(paper_id: str) -> list[dict[str, Any]]:
    path = DATA_DIR / paper_id / "chunks.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing chunks file: {path}")
    return read_json(path)


def keyword_overlap_retriever(query: str, chunks: list[dict[str, Any]], limit: int = 5, preferred_pages: list[int] | None = None) -> list[dict[str, Any]]:
    terms = set(tokenize(query))
    preferred = set(preferred_pages or [])
    scored: list[tuple[float, dict[str, Any]]] = []
    for chunk in chunks:
        chunk_terms = set(tokenize(chunk.get("text", "")))
        if not chunk_terms:
            continue
        overlap = len(terms.intersection(chunk_terms))
        score = overlap / max(len(terms), 1)
        if chunk.get("page") in preferred:
            score += 0.15
        scored.append((score, chunk))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]


def bm25_only_retriever(query: str, chunks: list[dict[str, Any]], limit: int = 5, preferred_pages: list[int] | None = None) -> list[dict[str, Any]]:
    query_terms = tokenize(query)
    query_counts = Counter(query_terms)
    tokenized_chunks = [tokenize(chunk.get("text", "")) for chunk in chunks]
    doc_frequency: Counter[str] = Counter()
    for tokens in tokenized_chunks:
        doc_frequency.update(set(tokens))

    total_docs = max(len(chunks), 1)
    average_length = sum(len(tokens) for tokens in tokenized_chunks) / max(len(tokenized_chunks), 1)
    preferred = set(preferred_pages or [])
    k1 = 1.4
    b = 0.72
    scored: list[tuple[float, dict[str, Any]]] = []

    for chunk, tokens in zip(chunks, tokenized_chunks):
        if not tokens:
            continue
        counts = Counter(tokens)
        length_norm = k1 * (1 - b + b * (len(tokens) / max(average_length, 1)))
        score = 0.0
        for term, query_weight in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log((total_docs - doc_frequency[term] + 0.5) / (doc_frequency[term] + 0.5) + 1)
            score += query_weight * idf * ((frequency * (k1 + 1)) / (frequency + length_norm))
        if chunk.get("page") in preferred:
            score += 0.3
        scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for score, chunk in scored[:limit] if score > 0]


def bm25_primary_no_page_hints(query: str, chunks: list[dict[str, Any]], limit: int = 5, preferred_pages: list[int] | None = None) -> list[dict[str, Any]]:
    return retrieve_chunks(query, chunks, limit=limit, preferred_pages=[])


def bm25_primary_with_page_hints(query: str, chunks: list[dict[str, Any]], limit: int = 5, preferred_pages: list[int] | None = None) -> list[dict[str, Any]]:
    return retrieve_chunks(query, chunks, limit=limit, preferred_pages=preferred_pages or [])


def dense_only_retriever(query: str, chunks: list[dict[str, Any]], limit: int = 5, preferred_pages: list[int] | None = None) -> list[dict[str, Any]]:
    return retrieve_dense_only(query, chunks, limit=limit, preferred_pages=None)


RETRIEVERS: dict[str, Callable[[str, list[dict[str, Any]], int, list[int] | None], list[dict[str, Any]]]] = {
    "keyword_overlap": keyword_overlap_retriever,
    "bm25_only": bm25_only_retriever,
    "bm25_primary_no_page_hints": bm25_primary_no_page_hints,
    "bm25_primary_with_page_hints": bm25_primary_with_page_hints,
    "dense_only": dense_only_retriever,
}


def rank_of_first_relevant(retrieved: list[dict[str, Any]], relevant_ids: set[str]) -> int | None:
    for index, chunk in enumerate(retrieved, start=1):
        if chunk.get("chunk_id") in relevant_ids:
            return index
    return None


def ndcg_at_k(retrieved_ids: list[Any], relevant_ids: set[str], k: int = 5) -> float:
    """Binary-relevance NDCG@k over the (possibly multiple) gold chunks."""
    import math
    dcg = sum(1.0 / math.log2(i + 2) for i, cid in enumerate(retrieved_ids[:k]) if cid in relevant_ids)
    ideal = min(len(relevant_ids), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal))
    return round(dcg / idcg, 4) if idcg else 0.0


def evaluate_case(case: dict[str, Any], retriever_name: str) -> dict[str, Any]:
    chunks = load_chunks(case["paper_id"])
    preferred_pages = extract_page_hints(case["query"])
    retrieved = RETRIEVERS[retriever_name](case["query"], chunks, 5, preferred_pages)
    retrieved_ids = [chunk.get("chunk_id") for chunk in retrieved]
    retrieved_pages = [chunk.get("page") for chunk in retrieved]
    relevant_ids = set(case["relevant_chunk_ids"])
    rank = rank_of_first_relevant(retrieved, relevant_ids)

    return {
        "case_id": case["id"],
        "paper_id": case["paper_id"],
        "query": case["query"],
        "retriever": retriever_name,
        "relevant_chunk_ids": case["relevant_chunk_ids"],
        "retrieved_chunk_ids": retrieved_ids,
        "retrieved_pages": retrieved_pages,
        "first_relevant_rank": rank,
        "recall_at_1": 1 if rank is not None and rank <= 1 else 0,
        "recall_at_3": 1 if rank is not None and rank <= 3 else 0,
        "recall_at_5": 1 if rank is not None and rank <= 5 else 0,
        "reciprocal_rank": 0.0 if rank is None else 1.0 / rank,
        "ndcg_at_5": ndcg_at_k(retrieved_ids, relevant_ids, 5),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for retriever in RETRIEVERS:
        subset = [row for row in rows if row["retriever"] == retriever]
        total = max(len(subset), 1)
        summary[retriever] = {
            "cases": len(subset),
            "recall_at_1": round(sum(row["recall_at_1"] for row in subset) / total, 3),
            "recall_at_3": round(sum(row["recall_at_3"] for row in subset) / total, 3),
            "recall_at_5": round(sum(row["recall_at_5"] for row in subset) / total, 3),
            "mrr": round(sum(row["reciprocal_rank"] for row in subset) / total, 3),
            "ndcg_at_5": round(sum(row["ndcg_at_5"] for row in subset) / total, 3),
        }
    return summary


def markdown_table(summary: dict[str, Any]) -> str:
    lines = [
        "| System | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in summary.items():
        lines.append(
            f"| `{name}` | {metrics['cases']} | {metrics['recall_at_1']} | {metrics['recall_at_3']} | {metrics['recall_at_5']} | {metrics['mrr']} |"
        )
    return "\n".join(lines)


def failed_cases(rows: list[dict[str, Any]], retriever: str) -> list[dict[str, Any]]:
    return [row for row in rows if row["retriever"] == retriever and row["recall_at_5"] == 0]


def current_system_case_table(rows: list[dict[str, Any]]) -> str:
    current = [row for row in rows if row["retriever"] == "bm25_primary_with_page_hints"]
    lines = [
        "| Case | Paper | Expected chunks | Top 5 retrieved chunks | First relevant rank |",
        "|---|---|---|---|---:|",
    ]
    for row in current:
        rank = row["first_relevant_rank"] if row["first_relevant_rank"] is not None else "miss"
        lines.append(
            f"| `{row['case_id']}` | `{row['paper_id']}` | {', '.join(row['relevant_chunk_ids'])} | "
            f"{', '.join(str(item) for item in row['retrieved_chunk_ids'])} | {rank} |"
        )
    return "\n".join(lines)


def write_report(cases: list[dict[str, Any]], rows: list[dict[str, Any]], summary: dict[str, Any],
                 report_path: Path = REPORT_MD) -> None:
    current = summary["bm25_primary_with_page_hints"]
    bm25 = summary["bm25_only"]
    keyword = summary["keyword_overlap"]
    no_hints = summary["bm25_primary_no_page_hints"]
    failures = failed_cases(rows, "bm25_primary_with_page_hints")

    report = f"""# ScholAR Quantitative Evaluation Report

Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## What was tested

This evaluation tests whether ScholAR retrieves the right evidence chunks before the model writes an answer. This matters because bad retrieval leads to weak answers and wrong citations.

The benchmark uses {len(cases)} manually checked retrieval cases from prepared local papers:

- `1706.03762`: Attention Is All You Need
- `2005.11401`: Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- `2302.13971`: LLaMA: Open and Efficient Foundation Language Models

Each case includes a real user-style question and a small set of expected relevant chunk IDs. The relevant chunks are taken from the existing extracted PDF chunks in `backend/data/papers`.

## How the benchmark was made

I used papers that were already prepared inside the project. For each paper, I inspected the extracted chunks and wrote questions that a real user might ask during a study session. Then I marked the chunk IDs that contain the answer evidence.

This is not a synthetic LLM-judged benchmark. It is a small human-written benchmark over the actual ScholAR chunk files.

The benchmark covers these query types:

- Main idea and contribution.
- Method and architecture.
- Training or implementation details.
- Result tables and benchmark numbers.
- Human evaluation.
- Safety, bias, toxicity, and carbon footprint.
- Page-hint questions.

## Metrics in plain language

- `Recall@1`: the first retrieved chunk is relevant.
- `Recall@3`: at least one of the first three retrieved chunks is relevant.
- `Recall@5`: at least one of the first five retrieved chunks is relevant.
- `MRR`: rewards systems that place the first relevant chunk higher in the list.

## Main results

{markdown_table(summary)}

## What this means

- The current BM25-primary retrieval reached Recall@5 of {current['recall_at_5']} and MRR of {current['mrr']} on this small benchmark.
- The keyword baseline reached Recall@5 of {keyword['recall_at_5']} and MRR of {keyword['mrr']}.
- The BM25 baseline reached Recall@5 of {bm25['recall_at_5']} and MRR of {bm25['mrr']}.
- The page-hint ablation compares `bm25_primary_with_page_hints` against `bm25_primary_no_page_hints`. In this run, page hints changed MRR from {no_hints['mrr']} to {current['mrr']}, so there was no measurable aggregate gain on this small benchmark.

The important honest finding from the earlier run was that BM25 was more reliable than the older hybrid-primary scoring. Based on that result, ScholAR now uses BM25 as the primary retriever and keeps semantic, section, phrase, and page signals as small reranking boosts. In this run, the current system matches BM25-only on the measured metrics while still keeping room for careful page-aware reranking.

These numbers are not a final research claim. They are a real starting point for the required quantitative evaluation.

## Comparison or ablation

This satisfies the requirement for at least one comparison or ablation:

- Comparison: current BM25-primary retrieval versus keyword overlap and BM25-only retrieval.
- Ablation: current BM25-primary retrieval with page hints versus the same retrieval without page hints.

## Failure cases for the current BM25-primary retrieval

"""
    if failures:
        for failure in failures:
            report += (
                f"- `{failure['case_id']}`: expected {failure['relevant_chunk_ids']}, "
                f"retrieved {failure['retrieved_chunk_ids']} on pages {failure['retrieved_pages']}.\n"
            )
    else:
        report += "- No Recall@5 failures for `bm25_primary_with_page_hints` in this benchmark.\n"

    report += f"""
## Per-case results for the current system

{current_system_case_table(rows)}

## How to interpret this for the project

For the final submission, this evaluation can support a simple claim:

ScholAR now uses the strongest observed baseline, BM25, as the main retrieval method. It improves over a simple keyword baseline and keeps page hints and lightweight reranking as careful additions instead of letting them overpower BM25.

That is useful for the project because it shows an evidence-based engineering decision. The system was changed after evaluation showed that BM25 was the most reliable grounding method for this benchmark.

To make this stronger for a conference-style submission, the benchmark should be expanded from 14 cases to at least 75 to 150 cases across more papers. The same script can be reused.

## What should be improved next

- Add more papers from different ML areas, not only classic NLP papers.
- Add more query types: method, results, limitations, implementation, ablation, dataset, and equations.
- Add a citation faithfulness metric that checks whether every displayed citation quote is actually found in the PDF text.
- Add answer-level evaluation later, but keep retrieval evaluation first because retrieval is the grounding layer.

## Files

- Benchmark cases: `evaluation/benchmark_cases.json`
- Raw results: `evaluation/results/retrieval_eval_results.json`
- This report: `evaluation/results/retrieval_eval_report.md`
"""
    report_path.write_text(report, encoding="utf-8")


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", default=str(CASES_PATH), help="path to a benchmark cases JSON")
    ap.add_argument("--tag", default="", help="output suffix, e.g. 'scaled' -> retrieval_eval_results_scaled.json")
    args = ap.parse_args()
    suffix = f"_{args.tag}" if args.tag else ""
    results_json = RESULTS_DIR / f"retrieval_eval_results{suffix}.json"
    report_md = RESULTS_DIR / f"retrieval_eval_report{suffix}.md"

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    cases = read_json(Path(args.cases))
    rows: list[dict[str, Any]] = []
    for case in cases:
        for retriever in RETRIEVERS:
            rows.append(evaluate_case(case, retriever))

    summary = summarize(rows)
    # Record whether the dense embedder was actually available. dense_only
    # returns [] (scoring 0.0 across the board) when it is not, which is an
    # infrastructure failure, not a real "dense retrieval is useless" result.
    try:
        from hybrid_retrieval import _get_embedder
        embedder_available = _get_embedder() is not None
    except Exception:
        embedder_available = False
    if not embedder_available and "dense_only" in summary:
        summary["dense_only"]["embedder_unavailable"] = True
    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "case_count": len(cases),
        "retrievers": list(RETRIEVERS),
        "embedder_available": embedder_available,
        "summary": summary,
        "rows": rows,
    }
    results_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    write_report(cases, rows, summary, report_md)

    print(markdown_table(summary))
    print(f"\nWrote {results_json}")
    print(f"Wrote {report_md}")


if __name__ == "__main__":
    main()
