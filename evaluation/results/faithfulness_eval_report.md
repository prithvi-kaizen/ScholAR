# ScholAR Citation Faithfulness Score (CFS) Evaluation Report

Generated: 2026-06-16 11:47:57
Approach:  Oracle-Claim Lexical Faithfulness (local, zero-dependency)
Cases:     20

---

## Metric Definitions

| Metric | Formula | Meaning |
|---|---|---|
| CTR | |claim_tokens ∩ chunk_tokens| / |claim_tokens| | Fraction of claim content-words found in retrieved chunks |
| KFP | found_key_facts / total_key_facts | Fraction of numbers and technical terms from claim found in chunks |
| CFS | 0.6 × CTR + 0.4 × KFP | Citation Faithfulness Score (main metric) |
| SCHR@K | 1 if supporting chunk ∈ top-K | Supporting Chunk Hit Rate at rank K |

---

## Summary

| Metric | Value |
|---|---:|
| Mean CFS | **0.914** |
| Mean CTR | 0.862 |
| Mean KFP | 0.993 |
| SCHR@1 | 0.300 |
| SCHR@3 | 0.550 |
| SCHR@5 | 0.900 |
| FAITHFUL (CFS ≥ 0.70) | 20 / 20 |
| PARTIAL  (0.45 ≤ CFS < 0.70) | 0 / 20 |
| UNFAITHFUL (CFS < 0.45) | 0 / 20 |

---

## Per-Case Results

| Case | Paper | Claim Type | CTR | KFP | CFS | Support Rank | Label |
|---|---|---|---:|---:|---:|---:|---|
| `attn_softmax_claim` | `1706.03762` | formula | 0.556 | 1.0 | **0.733** | miss | FAITHFUL |
| `attn_heads_claim` | `1706.03762` | architecture_detail | 1.0 | 1.0 | **1.0** | 4 | FAITHFUL |
| `attn_bleu_ende` | `1706.03762` | result_number | 1.0 | 1.0 | **1.0** | 1 | FAITHFUL |
| `attn_bleu_enfr` | `1706.03762` | result_number | 1.0 | 1.0 | **1.0** | 1 | FAITHFUL |
| `attn_training_steps` | `1706.03762` | training_detail | 1.0 | 1.0 | **1.0** | 1 | FAITHFUL |
| `attn_dropout` | `1706.03762` | training_detail | 1.0 | 1.0 | **1.0** | 2 | FAITHFUL |
| `attn_complexity` | `1706.03762` | technical_claim | 0.909 | 1.0 | **0.945** | 1 | FAITHFUL |
| `rag_parametric_nonparametric` | `2005.11401` | conceptual_claim | 0.769 | 1.0 | **0.862** | 1 | FAITHFUL |
| `rag_retriever` | `2005.11401` | technical_claim | 0.692 | 1.0 | **0.815** | 3 | FAITHFUL |
| `rag_nq_score` | `2005.11401` | result_number | 1.0 | 1.0 | **1.0** | 5 | FAITHFUL |
| `rag_trivia_score` | `2005.11401` | result_number | 1.0 | 1.0 | **1.0** | 5 | FAITHFUL |
| `rag_human_factuality` | `2005.11401` | human_eval | 0.867 | 1.0 | **0.92** | 3 | FAITHFUL |
| `rag_generator` | `2005.11401` | technical_claim | 1.0 | 1.0 | **1.0** | miss | FAITHFUL |
| `llama_training_tokens` | `2302.13971` | training_detail | 1.0 | 1.0 | **1.0** | 5 | FAITHFUL |
| `llama_public_data` | `2302.13971` | training_detail | 0.846 | 0.857 | **0.851** | 3 | FAITHFUL |
| `llama_sizes` | `2302.13971` | architecture_detail | 0.6 | 1.0 | **0.76** | 4 | FAITHFUL |
| `llama_hellaswag` | `2302.13971` | result_number | 0.667 | 1.0 | **0.8** | 4 | FAITHFUL |
| `llama_boolq` | `2302.13971` | result_number | 0.75 | 1.0 | **0.85** | 4 | FAITHFUL |
| `llama_carbon` | `2302.13971` | environmental_claim | 0.7 | 1.0 | **0.82** | 1 | FAITHFUL |
| `llama_architecture_norm` | `2302.13971` | architecture_detail | 0.889 | 1.0 | **0.933** | 2 | FAITHFUL |

---

## Ablation 1 — CFS vs. Retrieval Depth (Top-K)

Does retrieving more chunks improve faithfulness coverage?

| Chunks Used | N | Mean CFS |
|---|---:|---:|
| Top-1 | 20 | 0.731 |
| Top-2 | 20 | 0.815 |
| Top-3 | 20 | 0.853 |
| Top-4 | 20 | 0.893 |
| Top-5 | 20 | 0.914 |

---

## Ablation 2 — CFS by Claim Type

| Claim Type | N | Mean CFS | Mean CTR | Mean KFP | SCHR@5 |
|---|---:|---:|---:|---:|---:|
| `architecture_detail` | 3 | 0.898 | 0.830 | 1.000 | 1.000 |
| `conceptual_claim` | 1 | 0.862 | 0.769 | 1.000 | 1.000 |
| `environmental_claim` | 1 | 0.820 | 0.700 | 1.000 | 1.000 |
| `formula` | 1 | 0.733 | 0.556 | 1.000 | 0.000 |
| `human_eval` | 1 | 0.920 | 0.867 | 1.000 | 1.000 |
| `result_number` | 6 | 0.942 | 0.903 | 1.000 | 1.000 |
| `technical_claim` | 3 | 0.920 | 0.867 | 1.000 | 0.667 |
| `training_detail` | 4 | 0.963 | 0.962 | 0.964 | 1.000 |

---

## Interpretation

- A CFS of **0.914** means that on average, ~91% of each
  ground-truth answer's content tokens and key facts are recoverable from ScholAR's
  top-5 retrieved chunks.
- SCHR@5 of **0.900** indicates that in 90% of cases
  the manually labelled supporting chunk appeared in the top-5 retrieval list.
- The rank ablation shows how CFS grows as we use more retrieved chunks,
  revealing the marginal value of each successive rank position.

---

## Files

- Benchmark cases: `evaluation/faithfulness_cases.json`
- Raw results:     `evaluation/results/faithfulness_eval_results.json`
- This report:     `evaluation/results/faithfulness_eval_report.md`
