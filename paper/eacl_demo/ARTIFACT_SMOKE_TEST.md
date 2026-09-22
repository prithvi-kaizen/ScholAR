# ScholAR: EACL 2027 Demo Artifact Smoke Test Log

This document records the end-to-end smoke test executed on a clean environment to ensure full reproducibility of the ScholAR system and demonstration deliverables.

## 1. System Environment
- **Date Tested:** 2026-09-18 / 2026-09-19
- **Platform:** macOS (Darwin Kernel 24.0.0, Apple Silicon M3 Pro)
- **Host Memory:** 18 GB Unified Memory
- **Python Version:** 3.12.4 (Virtualenv `.venv312`)
- **Node.js Version:** 20.18.0
- **LaTeX Engine:** pdfTeX 3.141592653-2.6-1.40.26 (TeX Live 2025)
- **Local Model Backend:** Ollama v0.5.12 (`qwen3.5:9b`, 4-bit Q4_K_M)

## 2. Ingestion & Storage Smoke Test
```bash
# Ingestion test on sample computer science paper
PYTHONPATH=. ./.venv312/bin/pytest tests/test_paper_finalize_service.py tests/test_ingestion_routes.py
```
- **Result:** PASS (12 passed)
- **Verified:** Layout-aware AST parsing, dual-engine fallback, chunk bounding box normalization $[x_0, y_0, x_1, y_1]$, and atomic SQLite publication.

## 3. Strict-Local Network Policy Test
```bash
PYTHONPATH=. ./.venv312/bin/pytest tests/test_network_policy.py tests/test_offline_strict.py
```
- **Result:** PASS (8 passed)
- **Verified:** Rejection of non-loopback outbound socket connections; loopback communication (`127.0.0.1`) to local model servers permitted.

## 4. Hybrid Retrieval & Answer Generation Test
```bash
PYTHONPATH=. ./.venv312/bin/pytest tests/test_retrieval_hybrid.py tests/test_answer_pipeline.py
```
- **Result:** PASS (16 passed)
- **Verified:** Fused BM25 + dense search via RRF, evidence budgeting under RAM constraints, citation extraction, and non-streaming answer synthesis.

## 5. Verifier and Repair Test
```bash
PYTHONPATH=. ./.venv312/bin/pytest tests/test_verifier_and_repair.py tests/test_atomic_verifier.py
```
- **Result:** PASS (10 passed)
- **Verified:** Deterministic n-gram lexical overlap, numerical consistency checking, and citation remapping without hallucinated references.

## 6. Offline Human Evaluation Schema Test
```bash
PYTHONPATH=. ./.venv312/bin/pytest tests/test_offline_50_human_study.py tests/test_notion_ground_truth.py
```
- **Result:** PASS (6 passed)
- **Verified:** Support for `offline_50_v2` database schema, rater assignment validation, and progress backup import.

## 7. 150-Case Luna Complement Evaluation Test
```bash
PYTHONPATH=. ./.venv312/bin/pytest tests/test_eacl_demo_150_luna_judge.py
```
- **Result:** PASS (6 passed)
- **Verified:** Complement math ($200 - 50 = 150$, overlap $= 0$), reference eligibility partitioning ($113$ eligible vs $37$ ineligible), dry-run runner, aggregator fail-closed rules, batch export/import, and visual evidence crop evaluation (141 crops rendered across 90 visual cases).


## 8. Paper Build and Compliance Test
```bash
cd paper/eacl_demo
pdflatex -interaction=nonstopmode -halt-on-error main.tex
BSTINPUTS=style: bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```
- **Exit Code:** 0 (Clean compilation)
- **Content Page Count:** Exactly 6 pages (`\label{content:end}` on Page 6)
- **Undefined References:** 0
- **Overfull Hboxes in Paper:** 0
- **Authors Displayed:** Single-blind format confirmed.
