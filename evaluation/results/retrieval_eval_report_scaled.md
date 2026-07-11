# ScholAR Quantitative Evaluation Report

Generated on: 2026-07-11 13:18:59

## What was tested

This evaluation tests whether ScholAR retrieves the right evidence chunks before the model writes an answer. This matters because bad retrieval leads to weak answers and wrong citations.

The benchmark uses 100 manually checked retrieval cases from prepared local papers:

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
| `keyword_overlap` | 100 | 0.67 | 0.86 | 0.95 | 0.779 |
| `bm25_only` | 100 | 0.81 | 0.91 | 0.94 | 0.863 |
| `bm25_primary_no_page_hints` | 100 | 0.81 | 0.91 | 0.93 | 0.861 |
| `bm25_primary_with_page_hints` | 100 | 0.81 | 0.91 | 0.93 | 0.861 |
| `dense_only` | 100 | 0.47 | 0.65 | 0.74 | 0.572 |

## What this means

- The current BM25-primary retrieval reached Recall@5 of 0.93 and MRR of 0.861 on this small benchmark.
- The keyword baseline reached Recall@5 of 0.95 and MRR of 0.779.
- The BM25 baseline reached Recall@5 of 0.94 and MRR of 0.863.
- The page-hint ablation compares `bm25_primary_with_page_hints` against `bm25_primary_no_page_hints`. In this run, page hints changed MRR from 0.861 to 0.861, so there was no measurable aggregate gain on this small benchmark.

The important honest finding from the earlier run was that BM25 was more reliable than the older hybrid-primary scoring. Based on that result, ScholAR now uses BM25 as the primary retriever and keeps semantic, section, phrase, and page signals as small reranking boosts. In this run, the current system matches BM25-only on the measured metrics while still keeping room for careful page-aware reranking.

These numbers are not a final research claim. They are a real starting point for the required quantitative evaluation.

## Comparison or ablation

This satisfies the requirement for at least one comparison or ablation:

- Comparison: current BM25-primary retrieval versus keyword overlap and BM25-only retrieval.
- Ablation: current BM25-primary retrieval with page hints versus the same retrieval without page hints.

## Failure cases for the current BM25-primary retrieval

- `mine_text_044`: expected ['chunk_023', 'chunk_020'], retrieved ['chunk_003', 'chunk_030', 'chunk_014', 'chunk_007', 'chunk_025'] on pages [3, 30, 14, 7, 25].
- `mine_math_002`: expected ['chunk_041', 'chunk_008'], retrieved ['chunk_046', 'chunk_043', 'chunk_049', 'chunk_038', 'chunk_048'] on pages [46, 43, 49, 38, 48].
- `mine_math_019`: expected ['chunk_037'], retrieved ['chunk_011', 'chunk_019', 'chunk_005', 'chunk_014', 'chunk_020'] on pages [11, 19, 5, 14, 20].
- `mine_figure_003`: expected ['chunk_004', 'chunk_008', 'chunk_018'], retrieved ['chunk_006', 'chunk_014', 'chunk_019', 'chunk_009', 'chunk_015'] on pages [6, 14, 19, 9, 15].
- `mine_figure_012`: expected ['chunk_003'], retrieved ['chunk_004', 'chunk_006', 'chunk_007', 'chunk_009', 'chunk_005'] on pages [4, 6, 7, 9, 5].
- `mine_figure_014`: expected ['chunk_019'], retrieved ['chunk_004', 'chunk_014', 'chunk_013', 'chunk_005', 'chunk_020'] on pages [4, 14, 13, 5, 20].
- `mine_figure_016`: expected ['chunk_008'], retrieved ['chunk_006', 'chunk_007', 'chunk_009', 'chunk_010', 'chunk_001'] on pages [6, 7, 9, 10, 1].

## Per-case results for the current system

| Case | Paper | Expected chunks | Top 5 retrieved chunks | First relevant rank |
|---|---|---|---|---:|
| `mine_text_001` | `2006.11239` | chunk_005 | chunk_005, chunk_013, chunk_014, chunk_019, chunk_020 | 1 |
| `mine_text_002` | `1810.04805` | chunk_013, chunk_001, chunk_002, chunk_003, chunk_012, chunk_014 | chunk_003, chunk_014, chunk_001, chunk_013, chunk_002 | 1 |
| `mine_text_003` | `2005.11401` | chunk_003, chunk_004, chunk_006, chunk_007, chunk_008, chunk_017 | chunk_003, chunk_004, chunk_008, chunk_007, chunk_017 | 1 |
| `mine_text_004` | `1409.0473` | chunk_015, chunk_005 | chunk_014, chunk_005, chunk_015, chunk_012, chunk_008 | 2 |
| `mine_text_005` | `2005.14165` | chunk_008, chunk_009, chunk_010, chunk_012, chunk_022, chunk_025, chunk_033, chunk_043 | chunk_046, chunk_008, chunk_009, chunk_043, chunk_020 | 2 |
| `mine_text_006` | `2001.08361` | chunk_013, chunk_012, chunk_023, chunk_024 | chunk_013, chunk_012, chunk_024, chunk_020, chunk_015 | 1 |
| `mine_text_007` | `2111.00364` | chunk_010 | chunk_010, chunk_003, chunk_002, chunk_008, chunk_001 | 1 |
| `mine_text_008` | `2201.11903` | chunk_023, chunk_005, chunk_006, chunk_019 | chunk_005, chunk_019, chunk_020, chunk_023, chunk_029 | 1 |
| `mine_text_009` | `2010.11929` | chunk_022 | chunk_022, chunk_005, chunk_006, chunk_009, chunk_015 | 1 |
| `mine_text_010` | `2203.02155` | chunk_067 | chunk_018, chunk_065, chunk_014, chunk_013, chunk_067 | 5 |
| `mine_text_011` | `2010.00133` | chunk_013 | chunk_013, chunk_007, chunk_014, chunk_015, chunk_002 | 1 |
| `mine_text_012` | `2107.03374` | chunk_021 | chunk_021, chunk_024, chunk_029, chunk_030, chunk_026 | 1 |
| `mine_text_013` | `2201.08239` | chunk_026, chunk_015, chunk_027, chunk_028, chunk_034 | chunk_026, chunk_034 | 1 |
| `mine_text_014` | `2005.11401` | chunk_008 | chunk_008, chunk_007, chunk_006, chunk_005, chunk_004 | 1 |
| `mine_text_015` | `1409.0473` | chunk_003 | chunk_003, chunk_009, chunk_004, chunk_012, chunk_002 | 1 |
| `mine_text_016` | `1810.04805` | chunk_013, chunk_001, chunk_002, chunk_003, chunk_012, chunk_014 | chunk_002, chunk_003, chunk_014, chunk_013, chunk_007 | 1 |
| `mine_text_017` | `2010.11929` | chunk_008 | chunk_008, chunk_005, chunk_016, chunk_007, chunk_015 | 1 |
| `mine_text_018` | `1910.10683` | chunk_046 | chunk_046, chunk_041, chunk_036, chunk_012, chunk_029 | 1 |
| `mine_text_019` | `1901.02860` | chunk_014 | chunk_014, chunk_008, chunk_007, chunk_013, chunk_005 | 1 |
| `mine_text_020` | `2009.11462` | chunk_021 | chunk_021, chunk_007, chunk_006, chunk_020, chunk_004 | 1 |
| `mine_text_021` | `2005.11401` | chunk_009 | chunk_009, chunk_017, chunk_002, chunk_001, chunk_008 | 1 |
| `mine_text_022` | `1706.03762` | chunk_007 | chunk_007, chunk_006, chunk_011, chunk_004, chunk_005 | 1 |
| `mine_text_023` | `1409.0473` | chunk_008 | chunk_008, chunk_007, chunk_006, chunk_015, chunk_014 | 1 |
| `mine_text_024` | `2112.11446` | chunk_101 | chunk_101, chunk_100, chunk_099, chunk_047, chunk_098 | 1 |
| `mine_text_025` | `2006.11239` | chunk_006, chunk_022, chunk_023 | chunk_013, chunk_022, chunk_023, chunk_024, chunk_014 | 2 |
| `mine_text_026` | `1810.04805` | chunk_013, chunk_001, chunk_002, chunk_003, chunk_012, chunk_014 | chunk_003, chunk_006, chunk_014, chunk_001, chunk_013 | 1 |
| `mine_text_027` | `2107.03374` | chunk_021 | chunk_021, chunk_024, chunk_003, chunk_020, chunk_023 | 1 |
| `mine_text_028` | `2009.11462` | chunk_012 | chunk_012, chunk_013, chunk_009, chunk_008, chunk_002 | 1 |
| `mine_text_029` | `2005.11401` | chunk_003, chunk_002, chunk_004, chunk_006, chunk_007, chunk_008, chunk_017, chunk_019 | chunk_008, chunk_007, chunk_005, chunk_003, chunk_017 | 1 |
| `mine_text_030` | `2005.14165` | chunk_060 | chunk_012, chunk_060, chunk_007, chunk_006, chunk_005 | 2 |
| `mine_text_031` | `1712.00409` | chunk_012 | chunk_012, chunk_004, chunk_014, chunk_011, chunk_010 | 1 |
| `mine_text_032` | `1909.08053` | chunk_013 | chunk_013 | 1 |
| `mine_text_033` | `1910.10683` | chunk_052 | chunk_052, chunk_040, chunk_008, chunk_045, chunk_038 | 1 |
| `mine_text_034` | `1901.02860` | chunk_020 | chunk_020, chunk_019, chunk_016, chunk_018, chunk_002 | 1 |
| `mine_text_035` | `2010.00133` | chunk_007, chunk_006 | chunk_006, chunk_005, chunk_007, chunk_013, chunk_010 | 1 |
| `mine_text_036` | `2107.03374` | chunk_011 | chunk_011, chunk_029, chunk_013, chunk_010, chunk_026 | 1 |
| `mine_text_037` | `1409.0473` | chunk_015, chunk_005 | chunk_014, chunk_005, chunk_015, chunk_012, chunk_004 | 2 |
| `mine_text_038` | `2005.14165` | chunk_002, chunk_005, chunk_010, chunk_013, chunk_015, chunk_017, chunk_018, chunk_019, chunk_020, chunk_021, chunk_024, chunk_042, chunk_063, chunk_064, chunk_065, chunk_075 | chunk_020, chunk_010, chunk_005, chunk_019, chunk_018 | 1 |
| `mine_text_039` | `2111.00364` | chunk_006, chunk_005, chunk_010 | chunk_006, chunk_010, chunk_005, chunk_007, chunk_002 | 1 |
| `mine_text_040` | `1810.04805` | chunk_013, chunk_001, chunk_002, chunk_003, chunk_006, chunk_007, chunk_008, chunk_014, chunk_015 | chunk_003, chunk_014, chunk_001, chunk_008, chunk_013 | 1 |
| `mine_text_041` | `2201.11903` | chunk_008, chunk_004, chunk_015, chunk_022, chunk_029, chunk_036, chunk_037 | chunk_008, chunk_019, chunk_007, chunk_003, chunk_023 | 1 |
| `mine_text_042` | `2203.02155` | chunk_053 | chunk_053, chunk_055, chunk_041, chunk_058, chunk_057 | 1 |
| `mine_text_043` | `2010.00133` | chunk_013 | chunk_013, chunk_015, chunk_014, chunk_007, chunk_002 | 1 |
| `mine_text_044` | `2107.03374` | chunk_023, chunk_020 | chunk_003, chunk_030, chunk_014, chunk_007, chunk_025 | miss |
| `mine_text_045` | `1909.12673` | chunk_006 | chunk_006, chunk_005, chunk_021, chunk_008 | 1 |
| `mine_text_046` | `1712.00409` | chunk_017 | chunk_007, chunk_017, chunk_016, chunk_005, chunk_006 | 2 |
| `mine_text_047` | `2006.11239` | chunk_005 | chunk_005, chunk_013, chunk_014, chunk_019, chunk_020 | 1 |
| `mine_text_048` | `2111.00364` | chunk_003, chunk_001 | chunk_003, chunk_006, chunk_004, chunk_001, chunk_002 | 1 |
| `mine_text_049` | `2201.11990` | chunk_007 | chunk_007, chunk_006, chunk_003, chunk_004, chunk_005 | 1 |
| `mine_text_050` | `2010.11929` | chunk_008 | chunk_008, chunk_002, chunk_007, chunk_009, chunk_006 | 1 |
| `mine_math_001` | `2010.00133` | chunk_004 | chunk_004, chunk_005, chunk_008, chunk_014, chunk_013 | 1 |
| `mine_math_002` | `2203.02155` | chunk_041, chunk_008 | chunk_046, chunk_043, chunk_049, chunk_038, chunk_048 | miss |
| `mine_math_003` | `2005.11401` | chunk_003 | chunk_003, chunk_017, chunk_008, chunk_004, chunk_006 | 1 |
| `mine_math_004` | `1909.08053` | chunk_004 | chunk_004, chunk_005, chunk_011, chunk_003, chunk_001 | 1 |
| `mine_math_005` | `1909.12673` | chunk_005, chunk_001, chunk_002, chunk_003, chunk_004, chunk_006, chunk_008, chunk_017 | chunk_005, chunk_001, chunk_010, chunk_002, chunk_015 | 1 |
| `mine_math_006` | `1706.03762` | chunk_004 | chunk_004, chunk_005, chunk_001, chunk_007, chunk_013 | 1 |
| `mine_math_007` | `2009.11462` | chunk_017 | chunk_017, chunk_008, chunk_016, chunk_001, chunk_005 | 1 |
| `mine_math_008` | `2010.00133` | chunk_004 | chunk_004, chunk_005, chunk_008, chunk_012, chunk_013 | 1 |
| `mine_math_009` | `2201.08239` | chunk_017 | chunk_017, chunk_031, chunk_016, chunk_025, chunk_002 | 1 |
| `mine_math_010` | `2107.03374` | chunk_022 | chunk_021, chunk_022, chunk_003, chunk_005, chunk_028 | 2 |
| `mine_math_011` | `2010.00133` | chunk_004 | chunk_004, chunk_005, chunk_008, chunk_012, chunk_013 | 1 |
| `mine_math_012` | `2107.03374` | chunk_022 | chunk_022, chunk_008, chunk_009, chunk_025, chunk_012 | 1 |
| `mine_math_013` | `1910.10683` | chunk_042 | chunk_042, chunk_036, chunk_030, chunk_037, chunk_034 | 1 |
| `mine_math_014` | `2201.11990` | chunk_009 | chunk_009, chunk_018, chunk_022, chunk_010, chunk_041 | 1 |
| `mine_math_015` | `2005.14165` | chunk_019, chunk_018 | chunk_020, chunk_010, chunk_019, chunk_005, chunk_018 | 3 |
| `mine_math_016` | `2107.03374` | chunk_022 | chunk_022 | 1 |
| `mine_math_017` | `2201.11903` | chunk_023, chunk_005, chunk_006, chunk_019 | chunk_023, chunk_005, chunk_006, chunk_018, chunk_029 | 1 |
| `mine_math_018` | `1706.03762` | chunk_005 | chunk_005, chunk_004, chunk_003, chunk_002, chunk_014 | 1 |
| `mine_math_019` | `1910.10683` | chunk_037 | chunk_011, chunk_019, chunk_005, chunk_014, chunk_020 | miss |
| `mine_math_020` | `2006.11239` | chunk_006 | chunk_006, chunk_005, chunk_004, chunk_003, chunk_001 | 1 |
| `mine_math_021` | `1901.02860` | chunk_014 | chunk_014, chunk_013, chunk_008, chunk_007, chunk_009 | 1 |
| `mine_math_022` | `2005.14165` | chunk_011, chunk_039 | chunk_011, chunk_039, chunk_044, chunk_009, chunk_045 | 1 |
| `mine_math_023` | `2005.11401` | chunk_003 | chunk_003, chunk_004, chunk_008, chunk_007, chunk_017 | 1 |
| `mine_math_024` | `1910.10683` | chunk_024 | chunk_024, chunk_023, chunk_057, chunk_021, chunk_013 | 1 |
| `mine_math_025` | `1706.03762` | chunk_005 | chunk_004, chunk_005, chunk_003, chunk_007, chunk_008 | 2 |
| `mine_figure_001` | `2302.13971` | chunk_014, chunk_009, chunk_010 | chunk_010, chunk_009, chunk_006, chunk_005, chunk_002 | 1 |
| `mine_figure_002` | `2201.11903` | chunk_003, chunk_004, chunk_025, chunk_026, chunk_027, chunk_029, chunk_032, chunk_033, chunk_034 | chunk_003, chunk_002, chunk_029, chunk_004, chunk_011 | 1 |
| `mine_figure_003` | `2010.11929` | chunk_004, chunk_008, chunk_018 | chunk_006, chunk_014, chunk_019, chunk_009, chunk_015 | miss |
| `mine_figure_004` | `2512.24601` | chunk_002, chunk_003 | chunk_002, chunk_001, chunk_018, chunk_004, chunk_003 | 1 |
| `mine_figure_005` | `2006.11239` | chunk_016 | chunk_016, chunk_008, chunk_015 | 1 |
| `mine_figure_006` | `2302.13971` | chunk_003 | chunk_006, chunk_008, chunk_003, chunk_017, chunk_001 | 3 |
| `mine_figure_007` | `1512.03385` | chunk_011, chunk_010 | chunk_011, chunk_008, chunk_010, chunk_012, chunk_001 | 1 |
| `mine_figure_008` | `2201.11903` | chunk_004, chunk_005, chunk_006, chunk_007, chunk_008, chunk_015, chunk_020, chunk_023, chunk_025, chunk_027, chunk_028, chunk_032, chunk_033 | chunk_033, chunk_032, chunk_027, chunk_028, chunk_034 | 1 |
| `mine_figure_009` | `2010.11929` | chunk_006 | chunk_006, chunk_005, chunk_008, chunk_014, chunk_002 | 1 |
| `mine_figure_010` | `2512.24601` | chunk_008, chunk_016 | chunk_016, chunk_008, chunk_041, chunk_005, chunk_036 | 1 |
| `mine_figure_011` | `2006.11239` | chunk_002 | chunk_002, chunk_014, chunk_003, chunk_004, chunk_005 | 1 |
| `mine_figure_012` | `2302.13971` | chunk_003 | chunk_004, chunk_006, chunk_007, chunk_009, chunk_005 | miss |
| `mine_figure_013` | `2201.11903` | chunk_034, chunk_035, chunk_036, chunk_037, chunk_038, chunk_039, chunk_040, chunk_041, chunk_042, chunk_043 | chunk_041, chunk_034, chunk_007, chunk_023, chunk_022 | 1 |
| `mine_figure_014` | `2010.11929` | chunk_019 | chunk_004, chunk_014, chunk_013, chunk_005, chunk_020 | miss |
| `mine_figure_015` | `1512.03385` | chunk_005 | chunk_005, chunk_004, chunk_003, chunk_002, chunk_007 | 1 |
| `mine_figure_016` | `1706.03762` | chunk_008 | chunk_006, chunk_007, chunk_009, chunk_010, chunk_001 | miss |
| `mine_figure_017` | `2201.11903` | chunk_023, chunk_025, chunk_032, chunk_033 | chunk_033, chunk_032, chunk_027, chunk_028, chunk_034 | 1 |
| `mine_figure_018` | `2005.11401` | chunk_001, chunk_002, chunk_003, chunk_004, chunk_005, chunk_006, chunk_007, chunk_008, chunk_009, chunk_017, chunk_018, chunk_019 | chunk_005, chunk_007, chunk_006, chunk_018, chunk_009 | 1 |
| `mine_figure_019` | `2302.13971` | chunk_007 | chunk_007, chunk_004, chunk_008, chunk_006, chunk_010 | 1 |
| `mine_figure_020` | `1512.03385` | chunk_006, chunk_007, chunk_011, chunk_012 | chunk_006, chunk_007, chunk_004, chunk_012, chunk_005 | 1 |
| `mine_figure_021` | `1706.03762` | chunk_001, chunk_009, chunk_010 | chunk_009, chunk_010, chunk_003, chunk_001, chunk_002 | 1 |
| `mine_figure_022` | `2005.11401` | chunk_006, chunk_007, chunk_008, chunk_017 | chunk_006, chunk_008, chunk_007, chunk_019, chunk_004 | 1 |
| `mine_figure_023` | `2010.11929` | chunk_019 | chunk_008, chunk_009, chunk_020, chunk_015, chunk_019 | 5 |
| `mine_figure_024` | `2512.24601` | chunk_001, chunk_006, chunk_040 | chunk_040, chunk_017, chunk_030, chunk_006, chunk_027 | 1 |
| `mine_figure_025` | `2302.13971` | chunk_002, chunk_006, chunk_013, chunk_014 | chunk_002, chunk_006, chunk_007, chunk_010, chunk_014 | 1 |

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
