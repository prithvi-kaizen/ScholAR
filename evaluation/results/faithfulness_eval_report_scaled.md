# ScholAR NLI-Based Citation Faithfulness Report (v3)

**Generated:** 2026-07-11 13:42
**Cases:** 100
**Method:** NLI-CFS = 0.50×NLI + 0.30×SCR + 0.20×KFP
**NLI:** Sentence-level semantic entailment via MiniLM (SummaC/AlignScore-style)
**SCR:** MiniLM cosine similarity
**KFP:** Numeric and technical term precision

---

## System Comparison

| System | Mean NLI-CFS | SCHR@5 | Faithful | Partial | Unfaithful |
|---|---:|---:|---:|---:|---:|
| BM25-primary (ScholAR baseline) | **0.785** | 0.860 | 93/100 | 0/100 | 7/100 |
| Hybrid BM25+Dense+RRF          | **0.782** | 0.750 | 92/100 | 0/100 | 8/100 |

## Component Scores (BM25)

| Metric | Value |
|---|---:|
| Mean NLI-CFS (Tier 1) | 0.925 |
| Mean SCR (Tier 2)     | 0.433 |
| Mean KFP (Tier 3)     | 0.966 |
| Mean Combined NLI-CFS | **0.785** |
| SCHR@1 | 0.650 |
| SCHR@3 | 0.770 |
| SCHR@5 | 0.860 |
| Avg atoms/claim | 1.07 |

---

## Rank Ablation (NLI-CFS @ Top-K)

| Chunks Used | N | BM25 | Hybrid |
|---|---:|---:|---:|
| Top-1 | 100 | 0.663 | 0.634 |
| Top-2 | 100 | 0.743 | 0.723 |
| Top-3 | 100 | 0.767 | 0.762 |
| Top-4 | 100 | 0.772 | 0.769 |
| Top-5 | 100 | 0.785 | 0.782 |

---

## Claim-Type Breakdown (BM25)

| Claim Type | N | NLI-CFS | NLI | SCR | KFP | SCHR@5 |
|---|---:|---:|---:|---:|---:|---:|
| math | 25 | 0.758 | 0.920 | 0.340 | 0.980 | 0.920 |
| single_doc_text | 50 | 0.783 | 0.890 | 0.466 | 0.992 | 0.940 |
| visual | 25 | 0.818 | 1.000 | 0.459 | 0.900 | 0.640 |

---

## Per-Case Results

| Case | Type | System | NLI | SCR | KFP | CFS | Rank | Label |
|---|---|---|---:|---:|---:|---:|---:|---|
| `mine_text_001` | single_doc_text | BM25 | 0.000 | 0.249 | 1.000 | **0.275** | 1 | UNFAITHFUL |
| `mine_text_002` | single_doc_text | BM25 | 1.000 | 0.575 | 1.000 | **0.873** | 4 | FAITHFUL |
| `mine_text_003` | single_doc_text | BM25 | 1.000 | 0.594 | 1.000 | **0.878** | 1 | FAITHFUL |
| `mine_text_004` | single_doc_text | BM25 | 1.000 | 0.756 | 1.000 | **0.927** | 3 | FAITHFUL |
| `mine_text_005` | single_doc_text | BM25 | 1.000 | 0.485 | 1.000 | **0.845** | 2 | FAITHFUL |
| `mine_text_006` | single_doc_text | BM25 | 1.000 | 0.373 | 1.000 | **0.812** | 1 | FAITHFUL |
| `mine_text_007` | single_doc_text | BM25 | 1.000 | 0.470 | 0.750 | **0.791** | 1 | FAITHFUL |
| `mine_text_008` | single_doc_text | BM25 | 1.000 | 0.422 | 1.000 | **0.827** | 4 | FAITHFUL |
| `mine_text_009` | single_doc_text | BM25 | 1.000 | 0.497 | 1.000 | **0.849** | 1 | FAITHFUL |
| `mine_text_010` | single_doc_text | BM25 | 1.000 | 0.669 | 1.000 | **0.901** | 5 | FAITHFUL |
| `mine_text_011` | single_doc_text | BM25 | 1.000 | 0.214 | 1.000 | **0.764** | 1 | FAITHFUL |
| `mine_text_012` | single_doc_text | BM25 | 1.000 | 0.468 | 1.000 | **0.840** | 1 | FAITHFUL |
| `mine_text_013` | single_doc_text | BM25 | 1.000 | 0.424 | 1.000 | **0.827** | 1 | FAITHFUL |
| `mine_text_014` | single_doc_text | BM25 | 1.000 | 0.648 | 1.000 | **0.894** | 1 | FAITHFUL |
| `mine_text_015` | single_doc_text | BM25 | 1.000 | 0.327 | 1.000 | **0.798** | 1 | FAITHFUL |
| `mine_text_016` | single_doc_text | BM25 | 1.000 | 0.571 | 1.000 | **0.871** | 4 | FAITHFUL |
| `mine_text_017` | single_doc_text | BM25 | 1.000 | 0.589 | 1.000 | **0.877** | 1 | FAITHFUL |
| `mine_text_018` | single_doc_text | BM25 | 1.000 | 0.212 | 1.000 | **0.764** | 1 | FAITHFUL |
| `mine_text_019` | single_doc_text | BM25 | 1.000 | 0.345 | 1.000 | **0.804** | 1 | FAITHFUL |
| `mine_text_020` | single_doc_text | BM25 | 1.000 | 0.633 | 1.000 | **0.890** | 1 | FAITHFUL |
| `mine_text_021` | single_doc_text | BM25 | 1.000 | 0.529 | 1.000 | **0.859** | 1 | FAITHFUL |
| `mine_text_022` | single_doc_text | BM25 | 1.000 | 0.683 | 1.000 | **0.905** | 1 | FAITHFUL |
| `mine_text_023` | single_doc_text | BM25 | 1.000 | 0.553 | 1.000 | **0.866** | 1 | FAITHFUL |
| `mine_text_024` | single_doc_text | BM25 | 1.000 | 0.312 | 1.000 | **0.794** | 1 | FAITHFUL |
| `mine_text_025` | single_doc_text | BM25 | 1.000 | 0.765 | 1.000 | **0.929** | miss | FAITHFUL |
| `mine_text_026` | single_doc_text | BM25 | 1.000 | 0.578 | 1.000 | **0.873** | 5 | FAITHFUL |
| `mine_text_027` | single_doc_text | BM25 | 1.000 | 0.575 | 1.000 | **0.873** | 1 | FAITHFUL |
| `mine_text_028` | single_doc_text | BM25 | 0.000 | 0.022 | 1.000 | **0.207** | 1 | UNFAITHFUL |
| `mine_text_029` | single_doc_text | BM25 | 1.000 | 0.526 | 1.000 | **0.858** | 4 | FAITHFUL |
| `mine_text_030` | single_doc_text | BM25 | 1.000 | 0.641 | 1.000 | **0.892** | 2 | FAITHFUL |
| `mine_text_031` | single_doc_text | BM25 | 1.000 | 0.471 | 1.000 | **0.841** | 1 | FAITHFUL |
| `mine_text_032` | single_doc_text | BM25 | 0.500 | 0.463 | 1.000 | **0.589** | 1 | FAITHFUL |
| `mine_text_033` | single_doc_text | BM25 | 1.000 | 0.383 | 1.000 | **0.815** | 1 | FAITHFUL |
| `mine_text_034` | single_doc_text | BM25 | 1.000 | 0.257 | 1.000 | **0.777** | 1 | FAITHFUL |
| `mine_text_035` | single_doc_text | BM25 | 1.000 | 0.760 | 1.000 | **0.928** | 3 | FAITHFUL |
| `mine_text_036` | single_doc_text | BM25 | 1.000 | 0.314 | 1.000 | **0.794** | 1 | FAITHFUL |
| `mine_text_037` | single_doc_text | BM25 | 1.000 | 0.745 | 1.000 | **0.923** | 3 | FAITHFUL |
| `mine_text_038` | single_doc_text | BM25 | 1.000 | 0.534 | 1.000 | **0.860** | miss | FAITHFUL |
| `mine_text_039` | single_doc_text | BM25 | 1.000 | 0.643 | 1.000 | **0.893** | 1 | FAITHFUL |
| `mine_text_040` | single_doc_text | BM25 | 1.000 | 0.621 | 1.000 | **0.886** | 5 | FAITHFUL |
| `mine_text_041` | single_doc_text | BM25 | 1.000 | 0.492 | 1.000 | **0.848** | 1 | FAITHFUL |
| `mine_text_042` | single_doc_text | BM25 | 0.000 | 0.213 | 1.000 | **0.264** | 1 | UNFAITHFUL |
| `mine_text_043` | single_doc_text | BM25 | 1.000 | 0.296 | 1.000 | **0.789** | 1 | FAITHFUL |
| `mine_text_044` | single_doc_text | BM25 | 0.000 | 0.234 | 1.000 | **0.270** | miss | UNFAITHFUL |
| `mine_text_045` | single_doc_text | BM25 | 1.000 | 0.258 | 1.000 | **0.777** | 1 | FAITHFUL |
| `mine_text_046` | single_doc_text | BM25 | 1.000 | 0.532 | 1.000 | **0.860** | 2 | FAITHFUL |
| `mine_text_047` | single_doc_text | BM25 | 0.000 | 0.249 | 1.000 | **0.275** | 1 | UNFAITHFUL |
| `mine_text_048` | single_doc_text | BM25 | 1.000 | 0.396 | 1.000 | **0.819** | 1 | FAITHFUL |
| `mine_text_049` | single_doc_text | BM25 | 1.000 | 0.236 | 0.833 | **0.737** | 1 | FAITHFUL |
| `mine_text_050` | single_doc_text | BM25 | 1.000 | 0.476 | 1.000 | **0.843** | 1 | FAITHFUL |
| `mine_math_001` | math | BM25 | 1.000 | 0.574 | 1.000 | **0.872** | 1 | FAITHFUL |
| `mine_math_002` | math | BM25 | 1.000 | 0.336 | 1.000 | **0.801** | miss | FAITHFUL |
| `mine_math_003` | math | BM25 | 0.000 | 0.123 | 1.000 | **0.237** | 1 | UNFAITHFUL |
| `mine_math_004` | math | BM25 | 1.000 | 0.236 | 1.000 | **0.771** | 1 | FAITHFUL |
| `mine_math_005` | math | BM25 | 1.000 | 0.099 | 1.000 | **0.730** | 1 | FAITHFUL |
| `mine_math_006` | math | BM25 | 1.000 | 0.553 | 1.000 | **0.866** | 1 | FAITHFUL |
| `mine_math_007` | math | BM25 | 1.000 | 0.260 | 1.000 | **0.778** | 1 | FAITHFUL |
| `mine_math_008` | math | BM25 | 1.000 | 0.655 | 1.000 | **0.897** | 1 | FAITHFUL |
| `mine_math_009` | math | BM25 | 1.000 | 0.256 | 1.000 | **0.777** | 1 | FAITHFUL |
| `mine_math_010` | math | BM25 | 1.000 | 0.334 | 1.000 | **0.800** | 2 | FAITHFUL |
| `mine_math_011` | math | BM25 | 1.000 | 0.574 | 1.000 | **0.872** | 1 | FAITHFUL |
| `mine_math_012` | math | BM25 | 1.000 | 0.396 | 1.000 | **0.819** | 1 | FAITHFUL |
| `mine_math_013` | math | BM25 | 1.000 | 0.474 | 1.000 | **0.842** | 1 | FAITHFUL |
| `mine_math_014` | math | BM25 | 1.000 | 0.447 | 1.000 | **0.834** | 1 | FAITHFUL |
| `mine_math_015` | math | BM25 | 1.000 | 0.573 | 1.000 | **0.872** | 3 | FAITHFUL |
| `mine_math_016` | math | BM25 | 1.000 | 0.396 | 1.000 | **0.819** | 1 | FAITHFUL |
| `mine_math_017` | math | BM25 | 1.000 | 0.293 | 1.000 | **0.788** | 1 | FAITHFUL |
| `mine_math_018` | math | BM25 | 1.000 | 0.221 | 1.000 | **0.766** | 1 | FAITHFUL |
| `mine_math_019` | math | BM25 | 0.000 | 0.274 | 0.500 | **0.182** | miss | UNFAITHFUL |
| `mine_math_020` | math | BM25 | 1.000 | 0.239 | 1.000 | **0.772** | 1 | FAITHFUL |
| `mine_math_021` | math | BM25 | 1.000 | 0.314 | 1.000 | **0.794** | 1 | FAITHFUL |
| `mine_math_022` | math | BM25 | 1.000 | 0.387 | 1.000 | **0.816** | 1 | FAITHFUL |
| `mine_math_023` | math | BM25 | 1.000 | 0.102 | 1.000 | **0.731** | 1 | FAITHFUL |
| `mine_math_024` | math | BM25 | 1.000 | 0.169 | 1.000 | **0.751** | 1 | FAITHFUL |
| `mine_math_025` | math | BM25 | 1.000 | 0.221 | 1.000 | **0.766** | 2 | FAITHFUL |
| `mine_figure_001` | visual | BM25 | 1.000 | 0.206 | 1.000 | **0.762** | miss | FAITHFUL |
| `mine_figure_002` | visual | BM25 | 1.000 | 0.301 | 1.000 | **0.790** | 1 | FAITHFUL |
| `mine_figure_003` | visual | BM25 | 1.000 | 0.274 | 1.000 | **0.782** | miss | FAITHFUL |
| `mine_figure_004` | visual | BM25 | 1.000 | 0.651 | 1.000 | **0.895** | 1 | FAITHFUL |
| `mine_figure_005` | visual | BM25 | 1.000 | 0.535 | 0.000 | **0.660** | 1 | FAITHFUL |
| `mine_figure_006` | visual | BM25 | 1.000 | 0.483 | 1.000 | **0.845** | 3 | FAITHFUL |
| `mine_figure_007` | visual | BM25 | 1.000 | 0.379 | 1.000 | **0.814** | 1 | FAITHFUL |
| `mine_figure_008` | visual | BM25 | 1.000 | 0.497 | 1.000 | **0.849** | miss | FAITHFUL |
| `mine_figure_009` | visual | BM25 | 1.000 | 0.407 | 1.000 | **0.822** | 1 | FAITHFUL |
| `mine_figure_010` | visual | BM25 | 1.000 | 0.506 | 1.000 | **0.852** | 2 | FAITHFUL |
| `mine_figure_011` | visual | BM25 | 1.000 | 0.479 | 1.000 | **0.844** | 1 | FAITHFUL |
| `mine_figure_012` | visual | BM25 | 1.000 | 0.192 | 1.000 | **0.758** | miss | FAITHFUL |
| `mine_figure_013` | visual | BM25 | 1.000 | 0.466 | 0.000 | **0.640** | 2 | FAITHFUL |
| `mine_figure_014` | visual | BM25 | 1.000 | 0.432 | 0.500 | **0.730** | miss | FAITHFUL |
| `mine_figure_015` | visual | BM25 | 1.000 | 0.497 | 1.000 | **0.849** | 1 | FAITHFUL |
| `mine_figure_016` | visual | BM25 | 1.000 | 0.479 | 1.000 | **0.844** | miss | FAITHFUL |
| `mine_figure_017` | visual | BM25 | 1.000 | 0.559 | 1.000 | **0.868** | miss | FAITHFUL |
| `mine_figure_018` | visual | BM25 | 1.000 | 0.304 | 1.000 | **0.791** | miss | FAITHFUL |
| `mine_figure_019` | visual | BM25 | 1.000 | 0.391 | 1.000 | **0.817** | 1 | FAITHFUL |
| `mine_figure_020` | visual | BM25 | 1.000 | 0.463 | 1.000 | **0.839** | 1 | FAITHFUL |
| `mine_figure_021` | visual | BM25 | 1.000 | 0.634 | 1.000 | **0.890** | 4 | FAITHFUL |
| `mine_figure_022` | visual | BM25 | 1.000 | 0.650 | 1.000 | **0.895** | 1 | FAITHFUL |
| `mine_figure_023` | visual | BM25 | 1.000 | 0.506 | 1.000 | **0.852** | 5 | FAITHFUL |
| `mine_figure_024` | visual | BM25 | 1.000 | 0.695 | 1.000 | **0.909** | miss | FAITHFUL |
| `mine_figure_025` | visual | BM25 | 1.000 | 0.494 | 1.000 | **0.848** | 1 | FAITHFUL |
| `mine_text_001` | single_doc_text | Hybrid | 0.000 | 0.249 | 1.000 | **0.275** | 1 | UNFAITHFUL |
| `mine_text_002` | single_doc_text | Hybrid | 1.000 | 0.575 | 1.000 | **0.873** | 2 | FAITHFUL |
| `mine_text_003` | single_doc_text | Hybrid | 1.000 | 0.594 | 1.000 | **0.878** | 1 | FAITHFUL |
| `mine_text_004` | single_doc_text | Hybrid | 1.000 | 0.756 | 1.000 | **0.927** | 2 | FAITHFUL |
| `mine_text_005` | single_doc_text | Hybrid | 1.000 | 0.485 | 1.000 | **0.845** | 2 | FAITHFUL |
| `mine_text_006` | single_doc_text | Hybrid | 1.000 | 0.373 | 1.000 | **0.812** | 1 | FAITHFUL |
| `mine_text_007` | single_doc_text | Hybrid | 1.000 | 0.470 | 0.750 | **0.791** | 1 | FAITHFUL |
| `mine_text_008` | single_doc_text | Hybrid | 1.000 | 0.422 | 1.000 | **0.827** | 4 | FAITHFUL |
| `mine_text_009` | single_doc_text | Hybrid | 1.000 | 0.497 | 1.000 | **0.849** | 1 | FAITHFUL |
| `mine_text_010` | single_doc_text | Hybrid | 0.000 | 0.106 | 0.500 | **0.132** | miss | UNFAITHFUL |
| `mine_text_011` | single_doc_text | Hybrid | 1.000 | 0.214 | 1.000 | **0.764** | 1 | FAITHFUL |
| `mine_text_012` | single_doc_text | Hybrid | 1.000 | 0.468 | 1.000 | **0.840** | miss | FAITHFUL |
| `mine_text_013` | single_doc_text | Hybrid | 1.000 | 0.424 | 1.000 | **0.827** | 2 | FAITHFUL |
| `mine_text_014` | single_doc_text | Hybrid | 1.000 | 0.648 | 1.000 | **0.894** | 1 | FAITHFUL |
| `mine_text_015` | single_doc_text | Hybrid | 1.000 | 0.327 | 1.000 | **0.798** | 1 | FAITHFUL |
| `mine_text_016` | single_doc_text | Hybrid | 1.000 | 0.578 | 1.000 | **0.873** | 2 | FAITHFUL |
| `mine_text_017` | single_doc_text | Hybrid | 1.000 | 0.589 | 1.000 | **0.877** | 1 | FAITHFUL |
| `mine_text_018` | single_doc_text | Hybrid | 1.000 | 0.212 | 1.000 | **0.764** | 1 | FAITHFUL |
| `mine_text_019` | single_doc_text | Hybrid | 1.000 | 0.345 | 1.000 | **0.804** | 1 | FAITHFUL |
| `mine_text_020` | single_doc_text | Hybrid | 1.000 | 0.633 | 1.000 | **0.890** | miss | FAITHFUL |
| `mine_text_021` | single_doc_text | Hybrid | 1.000 | 0.529 | 1.000 | **0.859** | 1 | FAITHFUL |
| `mine_text_022` | single_doc_text | Hybrid | 1.000 | 0.683 | 1.000 | **0.905** | 1 | FAITHFUL |
| `mine_text_023` | single_doc_text | Hybrid | 1.000 | 0.553 | 1.000 | **0.866** | 1 | FAITHFUL |
| `mine_text_024` | single_doc_text | Hybrid | 0.000 | 0.314 | 1.000 | **0.294** | 1 | UNFAITHFUL |
| `mine_text_025` | single_doc_text | Hybrid | 1.000 | 0.765 | 1.000 | **0.929** | 5 | FAITHFUL |
| `mine_text_026` | single_doc_text | Hybrid | 1.000 | 0.578 | 1.000 | **0.873** | 2 | FAITHFUL |
| `mine_text_027` | single_doc_text | Hybrid | 1.000 | 0.575 | 1.000 | **0.873** | 1 | FAITHFUL |
| `mine_text_028` | single_doc_text | Hybrid | 0.000 | 0.022 | 1.000 | **0.207** | 1 | UNFAITHFUL |
| `mine_text_029` | single_doc_text | Hybrid | 1.000 | 0.526 | 1.000 | **0.858** | 1 | FAITHFUL |
| `mine_text_030` | single_doc_text | Hybrid | 1.000 | 0.641 | 1.000 | **0.892** | 1 | FAITHFUL |
| `mine_text_031` | single_doc_text | Hybrid | 1.000 | 0.471 | 1.000 | **0.841** | 5 | FAITHFUL |
| `mine_text_032` | single_doc_text | Hybrid | 0.500 | 0.463 | 1.000 | **0.589** | 1 | FAITHFUL |
| `mine_text_033` | single_doc_text | Hybrid | 1.000 | 0.383 | 1.000 | **0.815** | miss | FAITHFUL |
| `mine_text_034` | single_doc_text | Hybrid | 1.000 | 0.320 | 1.000 | **0.796** | 3 | FAITHFUL |
| `mine_text_035` | single_doc_text | Hybrid | 1.000 | 0.760 | 1.000 | **0.928** | 2 | FAITHFUL |
| `mine_text_036` | single_doc_text | Hybrid | 1.000 | 0.314 | 1.000 | **0.794** | 1 | FAITHFUL |
| `mine_text_037` | single_doc_text | Hybrid | 1.000 | 0.745 | 1.000 | **0.923** | 2 | FAITHFUL |
| `mine_text_038` | single_doc_text | Hybrid | 1.000 | 0.534 | 1.000 | **0.860** | miss | FAITHFUL |
| `mine_text_039` | single_doc_text | Hybrid | 1.000 | 0.643 | 1.000 | **0.893** | 1 | FAITHFUL |
| `mine_text_040` | single_doc_text | Hybrid | 1.000 | 0.621 | 1.000 | **0.886** | 2 | FAITHFUL |
| `mine_text_041` | single_doc_text | Hybrid | 1.000 | 0.492 | 1.000 | **0.848** | 1 | FAITHFUL |
| `mine_text_042` | single_doc_text | Hybrid | 0.000 | 0.178 | 1.000 | **0.253** | 3 | UNFAITHFUL |
| `mine_text_043` | single_doc_text | Hybrid | 1.000 | 0.296 | 1.000 | **0.789** | 1 | FAITHFUL |
| `mine_text_044` | single_doc_text | Hybrid | 0.000 | 0.249 | 1.000 | **0.275** | 3 | UNFAITHFUL |
| `mine_text_045` | single_doc_text | Hybrid | 1.000 | 0.323 | 1.000 | **0.797** | 1 | FAITHFUL |
| `mine_text_046` | single_doc_text | Hybrid | 1.000 | 0.532 | 1.000 | **0.860** | 1 | FAITHFUL |
| `mine_text_047` | single_doc_text | Hybrid | 0.000 | 0.249 | 1.000 | **0.275** | 1 | UNFAITHFUL |
| `mine_text_048` | single_doc_text | Hybrid | 1.000 | 0.583 | 1.000 | **0.875** | 5 | FAITHFUL |
| `mine_text_049` | single_doc_text | Hybrid | 1.000 | 0.236 | 0.833 | **0.737** | 1 | FAITHFUL |
| `mine_text_050` | single_doc_text | Hybrid | 1.000 | 0.476 | 1.000 | **0.843** | 1 | FAITHFUL |
| `mine_math_001` | math | Hybrid | 1.000 | 0.574 | 1.000 | **0.872** | 1 | FAITHFUL |
| `mine_math_002` | math | Hybrid | 1.000 | 0.336 | 1.000 | **0.801** | miss | FAITHFUL |
| `mine_math_003` | math | Hybrid | 0.000 | 0.151 | 1.000 | **0.245** | 2 | UNFAITHFUL |
| `mine_math_004` | math | Hybrid | 1.000 | 0.236 | 1.000 | **0.771** | 5 | FAITHFUL |
| `mine_math_005` | math | Hybrid | 1.000 | 0.213 | 1.000 | **0.764** | miss | FAITHFUL |
| `mine_math_006` | math | Hybrid | 1.000 | 0.553 | 1.000 | **0.866** | 1 | FAITHFUL |
| `mine_math_007` | math | Hybrid | 1.000 | 0.260 | 1.000 | **0.778** | 1 | FAITHFUL |
| `mine_math_008` | math | Hybrid | 1.000 | 0.655 | 1.000 | **0.897** | 1 | FAITHFUL |
| `mine_math_009` | math | Hybrid | 1.000 | 0.254 | 1.000 | **0.776** | miss | FAITHFUL |
| `mine_math_010` | math | Hybrid | 1.000 | 0.334 | 1.000 | **0.800** | 2 | FAITHFUL |
| `mine_math_011` | math | Hybrid | 1.000 | 0.574 | 1.000 | **0.872** | 1 | FAITHFUL |
| `mine_math_012` | math | Hybrid | 1.000 | 0.396 | 1.000 | **0.819** | 1 | FAITHFUL |
| `mine_math_013` | math | Hybrid | 1.000 | 0.474 | 1.000 | **0.842** | 2 | FAITHFUL |
| `mine_math_014` | math | Hybrid | 1.000 | 0.447 | 1.000 | **0.834** | 1 | FAITHFUL |
| `mine_math_015` | math | Hybrid | 1.000 | 0.573 | 1.000 | **0.872** | 3 | FAITHFUL |
| `mine_math_016` | math | Hybrid | 1.000 | 0.396 | 1.000 | **0.819** | 1 | FAITHFUL |
| `mine_math_017` | math | Hybrid | 1.000 | 0.293 | 1.000 | **0.788** | 1 | FAITHFUL |
| `mine_math_018` | math | Hybrid | 1.000 | 0.357 | 1.000 | **0.807** | 1 | FAITHFUL |
| `mine_math_019` | math | Hybrid | 1.000 | 0.426 | 1.000 | **0.828** | 2 | FAITHFUL |
| `mine_math_020` | math | Hybrid | 1.000 | 0.239 | 1.000 | **0.772** | miss | FAITHFUL |
| `mine_math_021` | math | Hybrid | 1.000 | 0.314 | 1.000 | **0.794** | 1 | FAITHFUL |
| `mine_math_022` | math | Hybrid | 1.000 | 0.387 | 1.000 | **0.816** | 1 | FAITHFUL |
| `mine_math_023` | math | Hybrid | 1.000 | 0.159 | 1.000 | **0.748** | 1 | FAITHFUL |
| `mine_math_024` | math | Hybrid | 1.000 | 0.169 | 1.000 | **0.751** | 5 | FAITHFUL |
| `mine_math_025` | math | Hybrid | 1.000 | 0.221 | 1.000 | **0.766** | 2 | FAITHFUL |
| `mine_figure_001` | visual | Hybrid | 1.000 | 0.277 | 1.000 | **0.783** | miss | FAITHFUL |
| `mine_figure_002` | visual | Hybrid | 1.000 | 0.414 | 1.000 | **0.824** | miss | FAITHFUL |
| `mine_figure_003` | visual | Hybrid | 1.000 | 0.382 | 1.000 | **0.815** | miss | FAITHFUL |
| `mine_figure_004` | visual | Hybrid | 1.000 | 0.757 | 1.000 | **0.927** | miss | FAITHFUL |
| `mine_figure_005` | visual | Hybrid | 1.000 | 0.535 | 0.000 | **0.660** | 1 | FAITHFUL |
| `mine_figure_006` | visual | Hybrid | 1.000 | 0.420 | 1.000 | **0.826** | miss | FAITHFUL |
| `mine_figure_007` | visual | Hybrid | 1.000 | 0.379 | 1.000 | **0.814** | 3 | FAITHFUL |
| `mine_figure_008` | visual | Hybrid | 1.000 | 0.497 | 1.000 | **0.849** | miss | FAITHFUL |
| `mine_figure_009` | visual | Hybrid | 1.000 | 0.412 | 1.000 | **0.824** | miss | FAITHFUL |
| `mine_figure_010` | visual | Hybrid | 1.000 | 0.506 | 1.000 | **0.852** | 1 | FAITHFUL |
| `mine_figure_011` | visual | Hybrid | 1.000 | 0.479 | 1.000 | **0.844** | 1 | FAITHFUL |
| `mine_figure_012` | visual | Hybrid | 1.000 | 0.192 | 1.000 | **0.758** | miss | FAITHFUL |
| `mine_figure_013` | visual | Hybrid | 1.000 | 0.466 | 0.000 | **0.640** | miss | FAITHFUL |
| `mine_figure_014` | visual | Hybrid | 1.000 | 0.432 | 0.500 | **0.730** | miss | FAITHFUL |
| `mine_figure_015` | visual | Hybrid | 1.000 | 0.557 | 1.000 | **0.867** | 4 | FAITHFUL |
| `mine_figure_016` | visual | Hybrid | 1.000 | 0.500 | 1.000 | **0.850** | 5 | FAITHFUL |
| `mine_figure_017` | visual | Hybrid | 1.000 | 0.559 | 1.000 | **0.868** | miss | FAITHFUL |
| `mine_figure_018` | visual | Hybrid | 1.000 | 0.304 | 1.000 | **0.791** | miss | FAITHFUL |
| `mine_figure_019` | visual | Hybrid | 1.000 | 0.391 | 1.000 | **0.817** | miss | FAITHFUL |
| `mine_figure_020` | visual | Hybrid | 1.000 | 0.463 | 1.000 | **0.839** | 1 | FAITHFUL |
| `mine_figure_021` | visual | Hybrid | 1.000 | 0.634 | 1.000 | **0.890** | miss | FAITHFUL |
| `mine_figure_022` | visual | Hybrid | 1.000 | 0.650 | 1.000 | **0.895** | 3 | FAITHFUL |
| `mine_figure_023` | visual | Hybrid | 1.000 | 0.422 | 1.000 | **0.827** | miss | FAITHFUL |
| `mine_figure_024` | visual | Hybrid | 1.000 | 0.695 | 1.000 | **0.909** | miss | FAITHFUL |
| `mine_figure_025` | visual | Hybrid | 1.000 | 0.494 | 1.000 | **0.848** | 1 | FAITHFUL |
