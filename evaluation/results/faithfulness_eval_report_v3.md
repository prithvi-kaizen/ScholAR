# ScholAR NLI-Based Citation Faithfulness Report (v3)

**Generated:** 2026-07-04 13:35
**Cases:** 51
**Method:** NLI-CFS = 0.50×NLI + 0.30×SCR + 0.20×KFP
**NLI:** Token-level entailment (SummaC/AlignScore-style)
**SCR:** MiniLM cosine similarity
**KFP:** Numeric and technical term precision

---

## System Comparison

| System | Mean NLI-CFS | SCHR@5 | Faithful | Partial | Unfaithful |
|---|---:|---:|---:|---:|---:|
| BM25-primary (ScholAR baseline) | **0.809** | 0.824 | 48/51 | 1/51 | 2/51 |
| Hybrid BM25+Dense+RRF          | **0.829** | 0.922 | 49/51 | 1/51 | 1/51 |

## Component Scores (BM25)

| Metric | Value |
|---|---:|
| Mean NLI-CFS (Tier 1) | 0.941 |
| Mean SCR (Tier 2)     | 0.484 |
| Mean KFP (Tier 3)     | 0.966 |
| Mean Combined NLI-CFS | **0.809** |
| SCHR@1 | 0.373 |
| SCHR@3 | 0.725 |
| SCHR@5 | 0.824 |
| Avg atoms/claim | 1.08 |

---

## Rank Ablation (NLI-CFS @ Top-K)

| Chunks Used | N | BM25 | Hybrid |
|---|---:|---:|---:|
| Top-1 | 51 | 0.658 | 0.669 |
| Top-2 | 51 | 0.757 | 0.776 |
| Top-3 | 51 | 0.799 | 0.796 |
| Top-4 | 51 | 0.791 | 0.817 |
| Top-5 | 51 | 0.809 | 0.829 |

---

## Claim-Type Breakdown (BM25)

| Claim Type | N | NLI-CFS | NLI | SCR | KFP | SCHR@5 |
|---|---:|---:|---:|---:|---:|---:|
| architecture_detail | 10 | 0.830 | 1.000 | 0.444 | 0.983 | 0.900 |
| conceptual_claim | 5 | 0.847 | 1.000 | 0.489 | 1.000 | 0.800 |
| environmental_claim | 1 | 0.918 | 1.000 | 0.727 | 1.000 | 1.000 |
| formula | 1 | 0.876 | 1.000 | 0.586 | 1.000 | 1.000 |
| human_eval | 2 | 0.882 | 1.000 | 0.607 | 1.000 | 1.000 |
| result_number | 13 | 0.811 | 0.923 | 0.514 | 0.974 | 0.923 |
| technical_claim | 11 | 0.844 | 1.000 | 0.525 | 0.932 | 0.636 |
| training_detail | 8 | 0.668 | 0.750 | 0.350 | 0.941 | 0.750 |

---

## Per-Case Results

| Case | Type | System | NLI | SCR | KFP | CFS | Rank | Label |
|---|---|---|---:|---:|---:|---:|---:|---|
| `attn_softmax_claim` | formula | BM25 | 1.000 | 0.586 | 1.000 | **0.876** | 1 | FAITHFUL |
| `attn_heads_claim` | architecture_detail | BM25 | 1.000 | 0.455 | 1.000 | **0.837** | 1 | FAITHFUL |
| `attn_bleu_ende` | result_number | BM25 | 1.000 | 0.666 | 1.000 | **0.900** | 1 | FAITHFUL |
| `attn_bleu_enfr` | result_number | BM25 | 1.000 | 0.663 | 1.000 | **0.899** | 1 | FAITHFUL |
| `attn_training_steps` | training_detail | BM25 | 0.000 | 0.362 | 0.667 | **0.242** | 1 | UNFAITHFUL |
| `attn_dropout` | training_detail | BM25 | 1.000 | 0.183 | 1.000 | **0.755** | 2 | FAITHFUL |
| `attn_complexity` | technical_claim | BM25 | 1.000 | 0.730 | 1.000 | **0.919** | 1 | FAITHFUL |
| `attn_encoder_decoder` | architecture_detail | BM25 | 1.000 | 0.877 | 1.000 | **0.963** | 1 | FAITHFUL |
| `attn_positional_encoding` | technical_claim | BM25 | 1.000 | 0.323 | 1.000 | **0.797** | 1 | FAITHFUL |
| `attn_warmup_steps` | training_detail | BM25 | 1.000 | 0.282 | 1.000 | **0.785** | 1 | FAITHFUL |
| `attn_label_smoothing` | training_detail | BM25 | 1.000 | 0.191 | 1.000 | **0.757** | 2 | FAITHFUL |
| `attn_parsing_result` | result_number | BM25 | 1.000 | 0.724 | 1.000 | **0.917** | 2 | FAITHFUL |
| `attn_multi_head_proj` | technical_claim | BM25 | 1.000 | 0.662 | 1.000 | **0.899** | 2 | FAITHFUL |
| `attn_recurrent_compare` | conceptual_claim | BM25 | 1.000 | 0.526 | 1.000 | **0.858** | miss | FAITHFUL |
| `attn_feed_forward` | architecture_detail | BM25 | 1.000 | 0.582 | 1.000 | **0.875** | 3 | FAITHFUL |
| `rag_parametric_nonparametric` | conceptual_claim | BM25 | 1.000 | 0.479 | 1.000 | **0.844** | 1 | FAITHFUL |
| `rag_retriever` | technical_claim | BM25 | 1.000 | 0.504 | 1.000 | **0.851** | 3 | FAITHFUL |
| `rag_nq_score` | result_number | BM25 | 1.000 | 0.400 | 1.000 | **0.820** | 5 | FAITHFUL |
| `rag_trivia_score` | result_number | BM25 | 0.000 | 0.340 | 0.667 | **0.235** | miss | UNFAITHFUL |
| `rag_human_factuality` | human_eval | BM25 | 1.000 | 0.741 | 1.000 | **0.922** | 3 | FAITHFUL |
| `rag_generator` | technical_claim | BM25 | 1.000 | 0.431 | 1.000 | **0.829** | miss | FAITHFUL |
| `rag_gradient_training` | training_detail | BM25 | 1.000 | 0.449 | 1.000 | **0.835** | miss | FAITHFUL |
| `rag_wikipedia_index` | technical_claim | BM25 | 1.000 | 0.444 | 1.000 | **0.833** | miss | FAITHFUL |
| `rag_msmarco` | result_number | BM25 | 1.000 | 0.672 | 1.000 | **0.902** | 1 | FAITHFUL |
| `rag_broader_impact` | conceptual_claim | BM25 | 1.000 | 0.633 | 1.000 | **0.890** | 2 | FAITHFUL |
| `rag_marginalization` | technical_claim | BM25 | 1.000 | 0.677 | 1.000 | **0.903** | 1 | FAITHFUL |
| `rag_appendix_details` | conceptual_claim | BM25 | 1.000 | 0.448 | 1.000 | **0.834** | 1 | FAITHFUL |
| `llama_training_tokens` | training_detail | BM25 | 0.000 | 0.521 | 1.000 | **0.356** | 4 | PARTIAL |
| `llama_public_data` | training_detail | BM25 | 1.000 | 0.293 | 0.857 | **0.759** | miss | FAITHFUL |
| `llama_sizes` | architecture_detail | BM25 | 1.000 | 0.353 | 1.000 | **0.806** | 5 | FAITHFUL |
| `llama_hellaswag` | result_number | BM25 | 1.000 | 0.308 | 1.000 | **0.792** | 3 | FAITHFUL |
| `llama_boolq` | result_number | BM25 | 1.000 | 0.343 | 1.000 | **0.803** | 1 | FAITHFUL |
| `llama_carbon` | environmental_claim | BM25 | 1.000 | 0.727 | 1.000 | **0.918** | 2 | FAITHFUL |
| `llama_architecture_norm` | architecture_detail | BM25 | 1.000 | 0.310 | 1.000 | **0.793** | 2 | FAITHFUL |
| `llama_activation_fn` | architecture_detail | BM25 | 1.000 | 0.345 | 1.000 | **0.804** | 1 | FAITHFUL |
| `llama_rotary_embedding` | architecture_detail | BM25 | 1.000 | 0.201 | 1.000 | **0.760** | 1 | FAITHFUL |
| `llama_learning_rate` | training_detail | BM25 | 1.000 | 0.521 | 1.000 | **0.856** | 1 | FAITHFUL |
| `llama_math_eval` | result_number | BM25 | 1.000 | 0.507 | 1.000 | **0.852** | 2 | FAITHFUL |
| `llama_mmlu` | result_number | BM25 | 1.000 | 0.524 | 1.000 | **0.857** | 1 | FAITHFUL |
| `llama_bias_gender` | human_eval | BM25 | 1.000 | 0.473 | 1.000 | **0.842** | 1 | FAITHFUL |
| `llama_triviaqa` | result_number | BM25 | 1.000 | 0.490 | 1.000 | **0.847** | 3 | FAITHFUL |
| `llama_approach_similar` | conceptual_claim | BM25 | 1.000 | 0.360 | 1.000 | **0.808** | 3 | FAITHFUL |
| `llama_tokenizer` | technical_claim | BM25 | 1.000 | 0.180 | 0.500 | **0.654** | miss | FAITHFUL |
| `llama_instruction_tuned` | technical_claim | BM25 | 1.000 | 0.552 | 0.750 | **0.816** | 2 | FAITHFUL |
| `attn_big_base_layer_count` | architecture_detail | BM25 | 1.000 | 0.486 | 1.000 | **0.846** | 5 | FAITHFUL |
| `rag_seq_vs_token` | technical_claim | BM25 | 1.000 | 0.481 | 1.000 | **0.844** | miss | FAITHFUL |
| `rag_fever_result` | result_number | BM25 | 1.000 | 0.493 | 1.000 | **0.848** | 5 | FAITHFUL |
| `llama_model_dim_65b` | architecture_detail | BM25 | 1.000 | 0.183 | 0.833 | **0.721** | miss | FAITHFUL |
| `llama_religion_bias` | result_number | BM25 | 1.000 | 0.555 | 1.000 | **0.867** | 2 | FAITHFUL |
| `attn_model_size_big` | architecture_detail | BM25 | 1.000 | 0.648 | 1.000 | **0.894** | 2 | FAITHFUL |
| `rag_qa_multiple_answers` | technical_claim | BM25 | 1.000 | 0.792 | 1.000 | **0.938** | 3 | FAITHFUL |
| `attn_softmax_claim` | formula | Hybrid | 1.000 | 0.586 | 1.000 | **0.876** | 1 | FAITHFUL |
| `attn_heads_claim` | architecture_detail | Hybrid | 1.000 | 0.455 | 1.000 | **0.837** | 1 | FAITHFUL |
| `attn_bleu_ende` | result_number | Hybrid | 1.000 | 0.666 | 1.000 | **0.900** | 1 | FAITHFUL |
| `attn_bleu_enfr` | result_number | Hybrid | 1.000 | 0.663 | 1.000 | **0.899** | 1 | FAITHFUL |
| `attn_training_steps` | training_detail | Hybrid | 0.000 | 0.362 | 0.667 | **0.242** | 2 | UNFAITHFUL |
| `attn_dropout` | training_detail | Hybrid | 1.000 | 0.183 | 1.000 | **0.755** | 5 | FAITHFUL |
| `attn_complexity` | technical_claim | Hybrid | 1.000 | 0.730 | 1.000 | **0.919** | 1 | FAITHFUL |
| `attn_encoder_decoder` | architecture_detail | Hybrid | 1.000 | 0.877 | 1.000 | **0.963** | 1 | FAITHFUL |
| `attn_positional_encoding` | technical_claim | Hybrid | 1.000 | 0.323 | 1.000 | **0.797** | 2 | FAITHFUL |
| `attn_warmup_steps` | training_detail | Hybrid | 1.000 | 0.282 | 1.000 | **0.785** | 2 | FAITHFUL |
| `attn_label_smoothing` | training_detail | Hybrid | 1.000 | 0.191 | 1.000 | **0.757** | 4 | FAITHFUL |
| `attn_parsing_result` | result_number | Hybrid | 1.000 | 0.749 | 1.000 | **0.925** | 1 | FAITHFUL |
| `attn_multi_head_proj` | technical_claim | Hybrid | 1.000 | 0.662 | 1.000 | **0.899** | 2 | FAITHFUL |
| `attn_recurrent_compare` | conceptual_claim | Hybrid | 1.000 | 0.642 | 1.000 | **0.893** | 4 | FAITHFUL |
| `attn_feed_forward` | architecture_detail | Hybrid | 1.000 | 0.582 | 1.000 | **0.875** | 4 | FAITHFUL |
| `rag_parametric_nonparametric` | conceptual_claim | Hybrid | 1.000 | 0.485 | 1.000 | **0.845** | 3 | FAITHFUL |
| `rag_retriever` | technical_claim | Hybrid | 1.000 | 0.600 | 1.000 | **0.880** | miss | FAITHFUL |
| `rag_nq_score` | result_number | Hybrid | 1.000 | 0.565 | 1.000 | **0.869** | 4 | FAITHFUL |
| `rag_trivia_score` | result_number | Hybrid | 1.000 | 0.340 | 1.000 | **0.802** | 5 | FAITHFUL |
| `rag_human_factuality` | human_eval | Hybrid | 1.000 | 0.741 | 1.000 | **0.922** | 2 | FAITHFUL |
| `rag_generator` | technical_claim | Hybrid | 1.000 | 0.431 | 1.000 | **0.829** | miss | FAITHFUL |
| `rag_gradient_training` | training_detail | Hybrid | 1.000 | 0.449 | 1.000 | **0.835** | miss | FAITHFUL |
| `rag_wikipedia_index` | technical_claim | Hybrid | 1.000 | 0.506 | 1.000 | **0.852** | miss | FAITHFUL |
| `rag_msmarco` | result_number | Hybrid | 1.000 | 0.672 | 1.000 | **0.902** | 1 | FAITHFUL |
| `rag_broader_impact` | conceptual_claim | Hybrid | 1.000 | 0.633 | 1.000 | **0.890** | 1 | FAITHFUL |
| `rag_marginalization` | technical_claim | Hybrid | 1.000 | 0.677 | 1.000 | **0.903** | 1 | FAITHFUL |
| `rag_appendix_details` | conceptual_claim | Hybrid | 1.000 | 0.448 | 1.000 | **0.834** | 1 | FAITHFUL |
| `llama_training_tokens` | training_detail | Hybrid | 0.000 | 0.521 | 1.000 | **0.356** | 4 | PARTIAL |
| `llama_public_data` | training_detail | Hybrid | 1.000 | 0.347 | 1.000 | **0.804** | 2 | FAITHFUL |
| `llama_sizes` | architecture_detail | Hybrid | 1.000 | 0.445 | 1.000 | **0.833** | 3 | FAITHFUL |
| `llama_hellaswag` | result_number | Hybrid | 1.000 | 0.325 | 1.000 | **0.798** | 2 | FAITHFUL |
| `llama_boolq` | result_number | Hybrid | 1.000 | 0.343 | 1.000 | **0.803** | 3 | FAITHFUL |
| `llama_carbon` | environmental_claim | Hybrid | 1.000 | 0.727 | 1.000 | **0.918** | 1 | FAITHFUL |
| `llama_architecture_norm` | architecture_detail | Hybrid | 1.000 | 0.310 | 1.000 | **0.793** | 1 | FAITHFUL |
| `llama_activation_fn` | architecture_detail | Hybrid | 1.000 | 0.345 | 1.000 | **0.804** | 3 | FAITHFUL |
| `llama_rotary_embedding` | architecture_detail | Hybrid | 1.000 | 0.239 | 1.000 | **0.772** | 5 | FAITHFUL |
| `llama_learning_rate` | training_detail | Hybrid | 1.000 | 0.521 | 1.000 | **0.856** | 1 | FAITHFUL |
| `llama_math_eval` | result_number | Hybrid | 1.000 | 0.507 | 1.000 | **0.852** | 1 | FAITHFUL |
| `llama_mmlu` | result_number | Hybrid | 1.000 | 0.554 | 1.000 | **0.866** | 4 | FAITHFUL |
| `llama_bias_gender` | human_eval | Hybrid | 1.000 | 0.473 | 1.000 | **0.842** | 1 | FAITHFUL |
| `llama_triviaqa` | result_number | Hybrid | 1.000 | 0.490 | 1.000 | **0.847** | 4 | FAITHFUL |
| `llama_approach_similar` | conceptual_claim | Hybrid | 1.000 | 0.368 | 1.000 | **0.810** | 1 | FAITHFUL |
| `llama_tokenizer` | technical_claim | Hybrid | 1.000 | 0.439 | 0.500 | **0.732** | 4 | FAITHFUL |
| `llama_instruction_tuned` | technical_claim | Hybrid | 1.000 | 0.552 | 0.750 | **0.816** | 1 | FAITHFUL |
| `attn_big_base_layer_count` | architecture_detail | Hybrid | 1.000 | 0.595 | 1.000 | **0.879** | 2 | FAITHFUL |
| `rag_seq_vs_token` | technical_claim | Hybrid | 1.000 | 0.515 | 1.000 | **0.855** | 4 | FAITHFUL |
| `rag_fever_result` | result_number | Hybrid | 1.000 | 0.493 | 1.000 | **0.848** | 1 | FAITHFUL |
| `llama_model_dim_65b` | architecture_detail | Hybrid | 1.000 | 0.330 | 1.000 | **0.799** | 2 | FAITHFUL |
| `llama_religion_bias` | result_number | Hybrid | 1.000 | 0.555 | 1.000 | **0.867** | 1 | FAITHFUL |
| `attn_model_size_big` | architecture_detail | Hybrid | 1.000 | 0.648 | 1.000 | **0.894** | 2 | FAITHFUL |
| `rag_qa_multiple_answers` | technical_claim | Hybrid | 1.000 | 0.792 | 1.000 | **0.938** | 2 | FAITHFUL |
