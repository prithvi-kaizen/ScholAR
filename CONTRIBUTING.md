# ScholAR: Technical Architecture, Methodology & Contributing Guide

Welcome to the **ScholAR** project! This document outlines the core technical architecture, state-of-the-art (SOTA) methodologies, citation verification protocols, and guidelines for contributing to the repository.

---

## 1. System Overview & Core Philosophy

ScholAR is an offline, multi-modal, pedagogical research assistant designed to run 100% locally on consumer edge hardware (optimized for Apple Silicon unified memory).

Unlike naive RAG systems that flatten research papers into unstructured character chunks and blindly accept LLM outputs, ScholAR implements a **4-Module Grounded RAG Architecture** with **AST-Guided Structural Ingestion**, **10-Archetype Adaptive Routing**, **Cross-Paper Multi-Document Graph Retrieval**, and **Runtime Sentence-Level Claim Verification with SOTA Citation Realignment & Pruning**.

---

## 2. Deep Technical Methodology Breakdown

### 🧱 Module 1: Hierarchical AST & Multi-View Ingestion
- **Abstract Syntax Tree (AST) Document Model**: PyMuPDF parses the PDF layout, font size hierarchy, and bold weights to build an explicit section tree (`Section > Subsection > Paragraph`).
- **3× Vector Region Clipping**: Tables and figures (including complex vector architectures and LaTeX `booktabs`) are clipped as high-resolution PNGs at $3\times$ zoom with normalized bounding boxes:
  $$\text{bbox}_{\text{norm}} = \left( \frac{x_0}{W}, \frac{y_0}{H}, \frac{x_1}{W}, \frac{y_1}{H} \right)$$
- **Relational Multi-View Storage**: Persists document hierarchy, paragraphs, and visual metadata in a unified local SQLite and JSON cache.

### ⚡ Module 2: 10-Archetype Adaptive Routing & Query Decomposition
- **Multi-Hop Query Decomposer**: Parses complex comparative and multi-entity user prompts into atomic subqueries to retrieve disjoint evidence paths (e.g. comparing Figure 1 and Figure 2).
- **Dynamic Modality Budgeting**: Classifies queries into 10 specialized route archetypes:
  1. `DIRECT_LOOKUP`
  2. `EXPLANATION`
  3. `COMPARISON`
  4. `MULTI_SECTION`
  5. `TABLE_NUMERIC`
  6. `FIGURE_VISUAL`
  7. `CHART_NUMERIC`
  8. `MIXED_TEXT_VISUAL`
  9. `CODE_ALGORITHM`
  10. `METHODOLOGY_DEEP_DIVE`
- Allocates specific text chunk budgets ($k_{\text{text}}$), visual image counts ($k_{\text{vis}}$), and native multimodal vision passes.

### 🌐 Module 3: Multi-Document Citation Graph & Cross-Paper Ingestion
- Extracts paper bibliography and resolves external citations through Semantic Scholar API.
- Enables 1-click on-demand ingestion of referenced papers, creating a multi-document retrieval pool with deterministic provenance tagging (`source_paper_id`).

### 🛡️ Module 4: SOTA Citation Realignment, Auto-Remapping & Pruning Engine
Grounded in recent peer-reviewed literature (**ALCE**, Gao et al. 2023; **AGREE**, Li et al. 2023; **Self-RAG**, Asai et al. 2023; **Corrective RAG**, Yan et al. 2024):

#### A. Attribution Disentanglement
- Explicit negative prompting prevents the LLM from appending citation tags to disclaimers, negative assertions (*"The paper does not mention X"*), or general conversational transitions.

#### B. Post-Hoc Entailment & Dynamic Auto-Remapping
- Evaluates every emitted citation $[k]$ against the cited chunk using token overlap and numeric consistency.
- If an LLM mis-indexes a citation (e.g. citing chunk 3 instead of chunk 1), the engine scans the candidate evidence pool and **automatically remaps** $[3] \rightarrow [1]$.

#### C. Deterministic Citation Pruning for Disclaimers
- If a sentence is an out-of-scope disclaimer or deductive jump without direct paper support, the engine **prunes and strips the citation marker** from that sentence.
- This eliminates false red `Contradicted` badges on disclaimers, ensuring that only genuinely verified facts carry citation badges.

#### D. 1-Step Active Entity & Metric Self-Repair
- If a sentence has high conceptual entailment with a cited chunk but minor phrasing or numeric drift, the verifier actively aligns the sentence with the exact evidence text.

---

## 3. Repository Structure

```
ScholAR/
├── backend/
│   ├── main.py                  # FastAPI application & API endpoints
│   ├── schemas/                 # Pydantic models (capabilities, AST, routing)
│   └── services/
│       ├── chunking_service.py   # AST-aware section chunking
│       ├── ollama_service.py     # Local LLM generation & 4-phase study goals
│       ├── pdf_service.py        # PyMuPDF parser & 3× visual region clipper
│       ├── reference_service.py  # Semantic Scholar enrichment & cross-paper ingestion
│       ├── retrieval_service.py  # BM25 & hybrid multi-subquery retrieval
│       ├── routing_service.py    # 10-archetype router & query decomposer
│       ├── storage_service.py    # Multi-view SQLite storage
│       ├── verifier_service.py   # Claim verification, auto-remapping & citation pruning
│       └── vision_service.py     # Multimodal vision question answering
├── frontend/
│   ├── app/                     # Next.js 14 App Router
│   ├── components/              # React UI components (PdfViewer, ChatBox, StudyGoals, etc.)
│   └── types/                   # TypeScript interfaces
├── tests/                       # Unit and integration test suite
├── ragarchitectureguide.md      # In-depth architectural & methodology guide
└── CONTRIBUTING.md              # Technical standards and methodology documentation
```

---

## 4. Contributing & Development Workflow

### Prerequisites
- Python 3.10+ with virtual environment (`.venv`)
- Node.js 18+ & npm
- Ollama with local model (e.g. `gemma4:12b` or `qwen3.5:9b`)

### Running the Test Suite
Before submitting any pull request or code change, run the test suite and frontend typecheck:

```bash
# Run Python backend unit tests (all 37+ tests must pass)
.venv/bin/python -m unittest discover -s tests

# Run Frontend TypeScript verification (0 errors required)
cd frontend && npm run typecheck
```

### Style & Engineering Guidelines
1. **Typography & Aesthetics**: Avoid emojis and raw em-dashes in user-facing UI text. Use clean typography and Lucide vectors.
2. **Mathematical Rendering**: Mathematical expressions and formulas must be wrapped in standard KaTeX syntax (`$x_t$`), while model names should render naturally (`BERT-Base`).
3. **Citation Integrity**: All RAG answers must pass through `ClaimVerifierService.verify_and_repair_answer` to maintain strict grounding.
