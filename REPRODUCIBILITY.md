# ScholAR: EACL 2027 Industry Track Reproducibility Guide

This repository contains the complete open-source implementation, benchmark datasets, evaluation harnesses, and manuscript artifacts for:

> **ScholAR: Multi-Level Reasoning and Software-Owned Provenance for Local Scientific Document Assistants**  
> *EACL 2027 Industry Track Submission (September 11, 2026)*

---

## 1. Quick Start & Master Reproduction

To execute the entire empirical suite, baseline comparisons, ablations, adversarial stress tests, and unit tests in a single command:

```bash
./run_experiments.sh
```

---

## 2. Individual Experiment Execution

| Experiment / Table | Command | Output JSON Artifact |
| :--- | :--- | :--- |
| **Classifier Evaluation** ($L_1 \dots L_5$) | `python evaluation/eval_classifier.py` | `evaluation/classifier_evaluation_results.json` |
| **Main Baseline Matrix** ($B_0 \dots B_9$) | `python evaluation/eval_baselines.py` | `evaluation/baseline_comparison_results.json` |
| **Component Ablation Suite** | `python evaluation/eval_ablations.py` | `evaluation/ablation_study_results.json` |
| **Adversarial Stress Tests** | `python evaluation/eval_adversarial.py` | `evaluation/adversarial_evaluation_results.json` |
| **Latency & Memory Profiling** | `python evaluation/profile_system.py` | `evaluation/system_profiling_results.json` |
| **Parser Robustness Suite** | `python evaluation/eval_parser_robustness.py` | `evaluation/parser_robustness_results.json` |
| **User Study Simulation** | `python evaluation/eval_user_study.py` | `evaluation/user_study_results.json` |
| **Full Unit Test Suite** (75 tests) | `python -m unittest discover -s tests` | Console Output |

---

## 3. Operational Environment & Invariants

- **Zero Cloud Data Egress**: All dense embeddings, cross-encoder sequence scoring, table arithmetic, graph reasoning, and verification execute strictly locally (`HF_HUB_OFFLINE=1`).
- **Hardware Agnostic**: Automatically detects and leverages:
  - Apple Silicon GPU acceleration via PyTorch `mps` backend.
  - NVIDIA GPUs via `cuda` backend.
  - Multi-threaded CPU fallback with vectorization.
- **Consumer Memory Profiles**: Automatically budgets context tokens and evidence block capacity according to active hardware:
  - **8 GB Unified Memory**: $\le 4,000$ tokens ($\le 6$ evidence blocks)
  - **16 GB Unified Memory**: $\le 8,000$ tokens ($\le 12$ evidence blocks)
  - **32 GB+ Unified Memory**: $\le 16,000$ tokens ($\le 24$ evidence blocks)

---

## 4. Benchmark Datasets

- **Curated Multi-Level Gold Dataset**: `evaluation/benchmark_gold_dataset.json` contains stratified question-evidence-answer pairs across all 5 reasoning levels ($L_1 \dots L_5$), multimodal targets, and unanswerable/abstention cases.
- **Landmark Development Papers ($N=10$)**:
  1. `1706.03762`: Attention Is All You Need (Vaswani et al., 2017)
  2. `2112.10752`: High-Resolution Latent Diffusion Models (Rombach et al., 2022)
  3. `1412.6980`: Adam Stochastic Optimization (Kingma & Ba, 2014)
  4. `1406.2661`: Generative Adversarial Nets (Goodfellow et al., 2014)
  5. `2406.08394`: VisionLLM v2 Multimodal Model (Wu et al., 2024)
  6. `2104.08663`: BEIR Zero-shot Information Retrieval (Thakur et al., 2021)
  7. `2603.14257`: Inter-document Multi-hop Scientific QA (2026)
  8. `2025.emnlp-main.77`: MEBench Cross-Document Multi-Entity QA (2025)
  9. `yale_thesis_1003`: Multimodal Multi-Document Understanding (2024)
  10. `2410.00526`: PaperQA2 Literature Search (2024)

---

## 5. Paper Manuscript

The complete 6-page LaTeX manuscript formatted to ACL/EACL standards is located in `manuscript/eacl2027_scholar.tex`.
