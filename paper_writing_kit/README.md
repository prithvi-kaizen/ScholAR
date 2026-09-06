# ScholAR: Paper Write-Up & Research Collaboration Kit

Welcome to the **ScholAR Research Paper Collaboration Kit**. This package consolidates all architectural documentation, empirical benchmarks, visual figures, ready-to-use LaTeX tables, reference bibliographies, and paper drafts into a single self-contained workspace for the writing team.

---

## 📁 Kit Directory Structure

```text
paper_writing_kit/
├── README.md                           # <--- You are here (Team onboarding & paper blueprint)
├── documents/                          # In-depth architectural guides & QA documentation
│   ├── ScholAR_Methodology_and_Paper_Guide.pdf  # 18-page visual guide with diagrams & pipeline walk-throughs
│   ├── ScholAR_Methodology_and_Paper_Guide.tex  # LaTeX source for methodology guide
│   ├── ScholAR_QA_Deep_Dive.pdf        # 16-page deep dive answering critical reviewer/architectural questions
│   ├── ScholAR_QA_Deep_Dive.md         # Markdown version of QA deep dive
│   ├── ScholAR_QA_Deep_Dive.tex        # LaTeX source of QA deep dive
│   ├── COMPREHENSIVE_WRITEUP.md        # Technical systems reference
│   └── RAG_explanation.md              # RAG pipeline reference and failure-mode analysis
├── references/                         # Master bibliography and citation guide
│   ├── scholar_master_references.bib   # Complete BibTeX file (ColPali, Nougat, Self-RAG, QASPER, etc.)
│   └── BIBLIOGRAPHY_GUIDE.md           # Guide mapping each citation to its section and role
├── figures/                            # Publication-ready charts, diagrams, and figures
│   ├── multilevel_reasoning_radar.pdf  # Vector radar chart comparing ScholAR against baselines
│   ├── pipeline_latency_breakdown.pdf  # Vector bar chart of latency breakdown across pipeline stages
│   ├── generate_paper_figures.py       # Python matplotlib script to regenerate/customize figures
│   └── *.png                           # High-resolution raster previews of pipeline & architecture
├── tables_and_results/                 # Pre-formatted LaTeX tables & experimental logs
│   ├── ready_to_use_tables.tex         # Copy-paste LaTeX tables (Main Results, Ablation, Routing, Deployment)
│   ├── query_demo_results.json         # Real-world query execution traces on Attention Is All You Need
│   └── claim_map.json                  # Claim verification audit matrix
└── manuscript_template/                # Official EACL / ACL LaTeX submission template
    ├── main.tex                        # Master document importing all sections
    ├── sections/                       # Modular section files (introduction, method, results, etc.)
    ├── style/                          # Official ACL LaTeX style files (acl.sty, acl_natbib.bst)
    └── Makefile                        # One-command compilation (`make`)
```

---

## 🎯 Paper Positioning & Target Venue

* **Target Venue**: **EACL 2027 / ACL System Demonstrations & Industry Track**
* **Paper Title Options**:
  1. *ScholAR: An Air-Gapped, Multimodal Scientific Document Understanding and Verification System*
  2. *Hierarchical Evidence Trees and Adaptive Routing for Trustworthy Scientific Document QA*
  3. *Beyond Flat Text RAG: Multimodal Late-Interaction and Mathematical Decoupling for Academic PDFs*
* **Core Value Proposition**:
  Standard RAG systems treat complex scientific PDFs as flat text strings, discarding typographic formulas, multi-column tables, visual charts, and document hierarchy. ScholAR resolves this through **5-channel hybrid retrieval** (combining sparse lexical, dense semantic, visual crops, and ColPali late-interaction), **hierarchical Evidence AST scoping**, **5-tier adaptive complexity routing**, and **deterministic mathematical decoupling**—running 100% locally on consumer hardware without cloud API egress.

---

## 📝 Master Paper Outline & Section-by-Section Guide

### Section 1: Introduction (Target: ~1.5 pages)
* **Problem**: Academic PDFs are hostile to standard LLMs: multi-column flows, embedded LaTeX equations, multi-hop citations, and high-density tables cause severe hallucinations ($>25\%$ in vanilla RAG).
* **Current Limitations**: Vision-only models (ColPali) lack section-level scoping; text-only models (BGE/BM25) miss charts and table geometries.
* **Our Solution & Contributions**:
  1. **Canonical Ingestion & Evidence AST**: 9 immutable artifacts compiled per paper.
  2. **5-Channel Hybrid Multimodal Retrieval**: Fusing lexical, dense semantic, visual crop vectors, ColPali page late-interaction, and structural AST priors.
  3. **Adaptive Complexity Routing ($L_1$–$L_5$)**: From direct factoids to cross-paper multi-hop synthesis.
  4. **Conservative Verification & Math Decoupling**: Deterministic AST-calculated arithmetic + citation-grounded self-critique.

### Section 2: Related Work (Target: ~1.0 page)
* **Visual Document Understanding (VDU)**: ColPali (`faysse2024colpali`), Nougat (`blecher2023nougat`), LayoutLM.
* **Retrieval-Augmented Generation**: ColBERT (`khattab2020colbert`), Dense Passage Retrieval, Hybrid RRF.
* **Scientific Document QA**: QASPER (`dasigi2021qasper`), SciFact (`wadden2020scifact`), M3SciQA (`li2024m3sciqa`).
* **Adaptive Retrieval & Verification**: Adaptive-RAG (`jeong2024adaptiverag`), Self-RAG (`asai2023selfrag`), CRAG (`yan2024crag`).

### Section 3: ScholAR System Architecture (Target: ~2.5 pages)
* **Ingestion Pipeline**: Compilation of the 9 canonical artifacts (`paper.pdf`, `evidence_ast.json`, `pages.json`, `chunks.json`, `figures.json`, `visual_units.json`, `metadata.json`, `document.db`, `index_cache`).
* **The 5 Retrieval Channels**: Formally define Reciprocal Rank Fusion (RRF) over all 5 modalities.
* **Hierarchical AST Scoping**: Preserving parent-child relationships (Document $\to$ Section $\to$ Subsection $\to$ Paragraph $\to$ Table/Figure).
* **Mathematical & Tabular Decoupling**: Extracting tables as structured matrices and evaluating formulas via deterministic symbolic execution instead of autoregressive token guessing.

### Section 4: Multi-Level Reasoning & Verification Engine (Target: ~1.5 pages)
* **Adaptive Routing Matrix**: Define criteria for $L_1$ (Factoid), $L_2$ (Local), $L_3$ (Multimodal/Table), $L_4$ (Cross-Section Multi-Hop), and $L_5$ (Cross-Document Synthesis).
* **Active Cross-Modal Graph Expansion**: Trigger conditions (when initial confidence $< 0.70$ or query requires table $\leftrightarrow$ section binding).
* **Conservative Verifier**: Four-point strictness check:
  1. Atomic claim premise entailment.
  2. Numerical consistency check against raw tables.
  3. Visual grounding validation (bounding-box check).
  4. Strict ungrounded hallucination penalty.

### Section 5: Experimental Evaluation Setup (Target: ~1.0 page)
* **Datasets**:
  * QASPER (Information-seeking questions on NLP papers).
  * SciFact (Scientific claim verification and rationale extraction).
  * M3SciQA (Multimodal multi-hop questions over charts and tables).
* **Baselines**: Lexical (BM25), Dense (BGE-M3), Visual-Only (ColPali), Naive Hybrid RAG.
* **Evaluation Metrics**: F1 Answer Score, Evidence Recall, FActScore atomic precision, Hallucination Rate, End-to-End Latency.

### Section 6: Empirical Results & Ablation (Target: ~1.5 pages)
* **Main Results**: Reference `\ref{tab:main_results}` from `tables_and_results/ready_to_use_tables.tex`.
* **Ablation Study**: Reference `\ref{tab:ablation}` showing quantitative drops when removing visual late interaction, AST scoping, and math decoupling.
* **Case Study**: Detailed walk-through of *Attention Is All You Need* (Table 3 ablation query vs. Section 3.2 multi-head attention formula).

### Section 7: Industrial Deployment & Hardware Footprint (Target: ~1.0 page)
* **Air-Gapped Local Architecture**: 100% loopback network policy; zero outbound telemetric leakage.
* **Hardware Benchmarks**: Performance on 1$\times$ Apple Silicon / RTX 4090 (Qwen-2.5-9B) vs. Edge profile (Llama-3.2-3B). Reference `\ref{tab:deployment_specs}`.

### Section 8: Conclusion & Limitations (Target: ~0.5 page)
* Summary of contributions and future exploration of agentic multi-paper hypothesis generation.

---

## 👥 Suggested Team Task Distribution

| Team Member / Role | Focus Sections | Key Deliverables | Associated Files |
| :--- | :--- | :--- | :--- |
| **Lead / Systems Author** | Section 3 (Architecture) & Section 7 (Deployment) | Ingestion pipeline description, 5-channel formulas, hardware latency table | `documents/ScholAR_Methodology_and_Paper_Guide.pdf`, `manuscript_template/sections/system.tex`, `deployment.tex` |
| **Reasoning & QA Author** | Section 4 (Multi-Level Reasoning & Verification) | $L_1$–$L_5$ router logic, graph traversal algorithm, verifier rules | `documents/ScholAR_QA_Deep_Dive.pdf`, `manuscript_template/sections/method.tex` |
| **Evaluation & Benchmark Author** | Section 5 (Setup) & Section 6 (Results & Ablation) | Fill benchmark tables, generate radar & latency plots, ablation analysis | `tables_and_results/ready_to_use_tables.tex`, `figures/generate_paper_figures.py`, `manuscript_template/sections/results.tex` |
| **Intro & Literature Author** | Section 1 (Intro) & Section 2 (Related Work) | Framing, motivation, failure mode examples, bibliography mapping | `references/scholar_master_references.bib`, `references/BIBLIOGRAPHY_GUIDE.md`, `manuscript_template/sections/introduction.tex` |

---

## 🚀 How to Compile the Paper Draft

Navigate to the `manuscript_template/` folder and run:
```bash
cd paper_writing_kit/manuscript_template
make
# Or manually:
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```
This will compile `main.pdf` using the official ACL styling rules.
