# ScholAR Quantitative Evaluation Report

Generated on: 2026-06-16 09:58:43

## What was tested

This evaluation tests whether ScholAR retrieves the right evidence chunks before the model writes an answer. This matters because bad retrieval leads to weak answers and wrong citations.

The benchmark uses 14 manually checked retrieval cases from prepared local papers:

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

| System | Cases | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|---:|
| `keyword_overlap` | 14 | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 14 | 0.714 | 0.929 | 1.0 | 0.812 |
| `bm25_primary_no_page_hints` | 14 | 0.714 | 0.929 | 1.0 | 0.812 |
| `bm25_primary_with_page_hints` | 14 | 0.714 | 0.929 | 1.0 | 0.812 |

## What this means

- The current BM25-primary retrieval reached Recall@5 of 1.0 and MRR of 0.812 on this small benchmark.
- The keyword baseline reached Recall@5 of 0.929 and MRR of 0.687.
- The BM25 baseline reached Recall@5 of 1.0 and MRR of 0.812.
- The page-hint ablation compares `bm25_primary_with_page_hints` against `bm25_primary_no_page_hints`. In this run, page hints changed MRR from 0.812 to 0.812, so there was no measurable aggregate gain on this small benchmark.

The important honest finding from the earlier run was that BM25 was more reliable than the older hybrid-primary scoring. Based on that result, ScholAR now uses BM25 as the primary retriever and keeps semantic, section, phrase, and page signals as small reranking boosts. In this run, the current system matches BM25-only on the measured metrics while still keeping room for careful page-aware reranking.

These numbers are not a final research claim. They are a real starting point for the required quantitative evaluation.

## Comparison or ablation

This satisfies the requirement for at least one comparison or ablation:

- Comparison: current BM25-primary retrieval versus keyword overlap and BM25-only retrieval.
- Ablation: current BM25-primary retrieval with page hints versus the same retrieval without page hints.

## Failure cases for the current BM25-primary retrieval

- No Recall@5 failures for `bm25_primary_with_page_hints` in this benchmark.

## Per-case results for the current system

| Case | Paper | Expected chunks | Top 5 retrieved chunks | First relevant rank |
|---|---|---|---|---:|
| `attn_architecture` | `1706.03762` | chunk_003, chunk_004, chunk_005 | chunk_003, chunk_001, chunk_002, chunk_005, chunk_006 | 1 |
| `attn_results_bleu` | `1706.03762` | chunk_008, chunk_009 | chunk_009, chunk_001, chunk_008, chunk_010, chunk_012 | 1 |
| `attn_complexity` | `1706.03762` | chunk_006, chunk_007 | chunk_006, chunk_007, chunk_010, chunk_002, chunk_008 | 1 |
| `attn_training_details` | `1706.03762` | chunk_007, chunk_008 | chunk_007, chunk_008, chunk_010, chunk_009, chunk_002 | 1 |
| `rag_core_idea` | `2005.11401` | chunk_001, chunk_002, chunk_003 | chunk_002, chunk_019, chunk_009, chunk_001, chunk_006 | 1 |
| `rag_training_objective` | `2005.11401` | chunk_003, chunk_004 | chunk_003, chunk_004, chunk_002, chunk_019, chunk_005 | 1 |
| `rag_qa_results` | `2005.11401` | chunk_006 | chunk_004, chunk_019, chunk_018, chunk_002, chunk_006 | 5 |
| `rag_human_eval` | `2005.11401` | chunk_008 | chunk_017, chunk_005, chunk_008, chunk_006, chunk_018 | 3 |
| `llama_training_data` | `2302.13971` | chunk_001, chunk_002 | chunk_006, chunk_003, chunk_002, chunk_011, chunk_010 | 3 |
| `llama_model_sizes` | `2302.13971` | chunk_003 | chunk_003, chunk_008, chunk_001, chunk_011, chunk_007 | 1 |
| `llama_commonsense_results` | `2302.13971` | chunk_004, chunk_005 | chunk_004, chunk_005, chunk_008, chunk_006, chunk_007 | 1 |
| `llama_bias_toxicity_carbon` | `2302.13971` | chunk_010, chunk_011 | chunk_011, chunk_010, chunk_007, chunk_009, chunk_008 | 1 |
| `rag_page_hint_ablation` | `2005.11401` | chunk_006, chunk_007 | chunk_019, chunk_006, chunk_009, chunk_015, chunk_010 | 2 |
| `attn_page_hint_ablation` | `1706.03762` | chunk_008, chunk_009 | chunk_008, chunk_009, chunk_001, chunk_010, chunk_012 | 1 |

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
