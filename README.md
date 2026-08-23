# ScholAR — Smart Companion for Holistic Organization, Literature Analysis & Research

ScholAR is a **local-first, privacy-preserving RAG system** for deep reading and Q&A over scientific PDFs. It combines page-grounded citation retrieval, multi-modal visual grounding, and multi-document reasoning into a unified research assistant.


---

## What ScholAR Does

- **Upload or search any arXiv paper** → instant ingestion, chunking, and indexing
- **Ask questions in natural language** → page-grounded answers with source citations
- **Visual grounding** → queries about figures/tables route to a vision-LLM that reads the actual image
- **Multi-document mode** → load cited references and ask cross-paper questions
- **100% local inference** → runs entirely on-device via Ollama, no cloud API required

---

## Key Technical Contributions

### 1. Page-Preserving Chunking
Every text chunk maps to a **single PDF page** — no cross-page merging. Implemented as a sliding window over per-page tokens (`target_words=1400`, `overlap=120`) with regex-based section title detection. This is the prerequisite for faithful, pinpoint citations.

### 2. BM25-Primary Retrieval with Heuristic Reranking
The production retrieval layer uses BM25 as the primary lexical signal:

| Boost | Trigger | Magnitude |
|---|---|---|
| Query expansion | Research-specific synonyms (e.g., `result → accuracy, BLEU, score`) | vocabulary |
| Page hint | User query mentions a page number | +1.25 |
| Section hint | Query implies Abstract / Method / Results / Limitation | +0.5–1.0 |
| Visual cue | Query mentions figure, table, chart, plot | +1.5 |
| Phrase boost | `"we introduce"`, `"we present"`, `"outperforms"` | +0.3–0.5 |

### 3. Hybrid BM25 + Dense + RRF Retrieval (Research Evaluation)
For the faithfulness evaluation pipeline:
- **Dense ranking:** `all-MiniLM-L6-v2` embeddings (pure PyTorch, no sentence-transformers required)
- **Reciprocal Rank Fusion (RRF, k=60):** Cormack et al. SIGIR 2009 formula
- **Post-fusion page boost:** +0.05 for page-hinted chunks

### 4. Visual Grounding (Figure & Table QA)
When a query targets a figure, table, or chart:
1. Visual-cue retrieval boosts figure chunks (+1.5) to rank them first
2. The figure image (rendered page region) is base64-encoded and sent to the local **Ollama `qwen3.5:9b`** model (natively multimodal) with the question
3. Caption-only fallback if the image is unavailable, too small, or Ollama is unreachable
4. The frontend renders a teal-bordered thumbnail of the cited figure inline in the answer

### 5. Multi-Document Extension
- **arXiv papers:** References resolved via Semantic Scholar API by arXiv ID (full bibliography, up to 500 refs)
- **Uploaded PDFs:** References resolved via **S2 title search** (new) — finds the paper on S2 by title, then fetches its reference list, enabling multi-doc mode for any PDF
- **Regex coverage:** Handles both numbered `[N]` and author-year `(Vaswani et al., 2017)` reference styles; scans last 6 pages
- **Reference ingestion:** Each cited paper can be individually downloaded, chunked, and added to the session — answers then draw from both the anchor paper and loaded references

### 6. NLI-Based Citation Faithfulness Metric (NLI-CFS)
Three-tier faithfulness scorer aligned with 2024–2026 SOTA practices:

| Tier | Method | Weight |
|---|---|---|
| SummaC-ZS | Sentence-level cosine entailment via MiniLM embeddings | 50% |
| SCR | Whole-claim vs. best-chunk cosine similarity | 30% |
| KFP | Exact recall of numbers and technical terms | 20% |

**CFS = 0.50 × SummaC-ZS + 0.30 × SCR + 0.20 × KFP**

Claims are labeled FAITHFUL (CFS ≥ 0.55), PARTIAL (≥ 0.35), or UNFAITHFUL.

### 7. Fully Local Inference
- **Text and vision, one model:** Ollama with `qwen3.5:9b` — natively multimodal, fully offline, private
- **No cloud dependency:** every answer, study goal, and figure/table QA call runs on-device

---

## Evaluation Results

### Retrieval Benchmark (14 annotated cases, 3 papers)

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| `keyword_overlap` | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 0.714 | 0.857 | 0.929 | 0.788 |
| `bm25_primary_no_page_hints` | 0.714 | 0.857 | 0.929 | 0.788 |
| `bm25_primary_with_page_hints` | 0.714 | 0.857 | 0.929 | 0.788 |
| `dense_only` (single-pass MiniLM cosine, no BM25/rerank/page-hints) | **0.786** | **1.000** | **1.000** | **0.881** |

Dense-only beats every lexical config on this 14-case benchmark — a genuine, small-N finding (see the paper's Discussion). It doesn't change the production choice: on the larger 51-case faithfulness benchmark below, hybrid (not dense-only) wins.

### Faithfulness Benchmark (51 oracle-claim cases, 3 papers, 8 claim types)

| System | Combined CFS | SCHR@5 | Faithful |
|---|---:|---:|---:|
| BM25-primary | 0.809 | 0.824 | 48 / 51 |
| Hybrid BM25 + Dense + RRF | 0.829 | 0.922 | 49 / 51 |

### Baseline Scope

Retrieval/faithfulness baselines above are internal ablations (lexical → dense → hybrid), matching the BEIR-standard comparison framework. We do **not** benchmark against OpenScholar, PaperQA2, or SciRAG head-to-head: all three depend on cloud-hosted frontier LLM backends (and, for OpenScholar, a tens-of-millions-of-papers datastore), incompatible with ScholAR's local-only, single-laptop design constraint. See the paper's Discussion section for the full reasoning.

Spans *Attention Is All You Need*, *RAG*, and *LLaMA* across 8 claim types: `result_number` (13), `technical_claim` (11), `architecture_detail` (10), `training_detail` (8), `conceptual_claim` (5), `human_eval` (2), `formula` (1), `environmental_claim` (1).

### Visual Grounding Benchmark (18 figure-grounded cases, 4 reasoning types)

| Metric | Score |
|---|---:|
| Retrieval R@5 | **18 / 18 = 1.000** |
| Caption fallback rate | 0 / 18 (0%) |

### Multi-Document Locality Benchmark (10 `locality_arxiv` cases, M3SciQA-style)

| Metric | Score |
|---|---:|
| Locality R@1 | 0.00 |
| Locality R@5 | 0.50 |
| MRR | 0.183 |
| Random-guess floor (R@5) | 0.625 |

A chunk-ID collision bug (fixed) had inflated an earlier version of these numbers (R@5 0.80, MRR 0.356) — the corrected result is at or below the random-guessing floor on this small benchmark. See the paper's Discussion for the full writeup and the oracle-bound analysis.

### Human Evaluation Pipeline (built, ready to run)

A model-agnostic, citation-grounded human evaluation lives in `evaluation/human_eval/`, grounded in the methodology of OpenScholar, SciRAG, and PaperQA2. Each of 100 curated questions (40 single-document, 20 visual, 20 multi-document, 20 hard-retrieval) is answered by four local models running the same ScholAR pipeline — only the generation model changes. Expert evaluators score each answer on four anchored 1-5 dimensions (Relevance, Coverage, Faithfulness, Usefulness) plus per-citation Supported/Partial/Unsupported grading, through a self-contained offline HTML interface. The analysis reports per-model quality and citation precision/recall/F1, a Friedman test for model-agnostic grounding, inter-annotator agreement, and the correlation between human faithfulness and the automated NLI-CFS metric. See `evaluation/human_eval/README.md` to run it.

---

## Repository Structure

```text
ScholAR/
├── backend/
│   ├── main.py                     # FastAPI entry point + all API routes
│   ├── services/
│   │   ├── pdf_service.py          # PDF ingestion, page image rendering
│   │   ├── chunking_service.py     # Page-preserving chunking + figure chunks
│   │   ├── retrieval_service.py    # BM25 + visual-cue boosting
│   │   ├── ollama_service.py       # Local Ollama LLM routing (text + vision)
│   │   ├── vision_service.py       # Figure QA via local Ollama vision model
│   │   ├── reference_service.py    # Multi-doc: S2 API + title-search for uploads
│   │   └── arxiv_service.py        # arXiv search and paper metadata
│   ├── data/                       # Per-paper extracted data (gitignored)
│   └── .env                        # Local config (not committed)
├── frontend/
│   ├── app/
│   │   ├── page.tsx                # Home: paper search + upload
│   │   └── paper/[id]/page.tsx     # Study workspace
│   └── components/
│       ├── StudyWorkspace.tsx      # Resizable split layout (drag handle)
│       ├── StudyPanel.tsx          # Tabbed panel: Chat / Study Goals / References
│       ├── ChatBox.tsx             # Chat UI: typing dots, model badge, suggestions
│       ├── PdfViewer.tsx           # PDF renderer with page navigation + zoom
│       ├── ReferencesPanel.tsx     # Multi-doc: reference cards + batch ingest
│       └── StudyGoals.tsx          # AI-generated study goal cards
├── evaluation/
│   ├── embedder.py                 # Local MiniLM-L6-v2 (pure PyTorch)
│   ├── hybrid_retrieval.py         # BM25 + Dense + RRF pipeline
│   ├── nli_faithfulness.py         # SummaC-ZS faithfulness scorer
│   ├── run_retrieval_eval.py       # 14-case retrieval benchmark
│   ├── run_faithfulness_eval.py    # 51-case faithfulness benchmark
│   ├── run_visual_eval.py          # 18-case visual grounding benchmark
│   ├── run_multidoc_eval.py        # Multi-doc locality benchmark
│   ├── benchmark_cases.json        # Retrieval ground truth
│   ├── faithfulness_cases.json     # Faithfulness oracle claims
│   ├── visual_benchmark.json       # Figure-grounded QA cases
│   ├── human_eval/                 # Human-evaluation pipeline (see below)
│   │   ├── HUMAN_EVAL_DESIGN.md     # Instrument design, grounded in SciRAG/OpenScholar/PaperQA2
│   │   ├── rubric.md               # Evaluator guideline (Q1-Q7, anchored scales)
│   │   ├── cases.json              # 100 curated cases (4 capabilities)
│   │   ├── generate_answers.py     # Runs each case across 4 local models (one at a time)
│   │   ├── _build_score_sheet.py   # Builds the offline blinded scoring interface
│   │   └── compute_scores.py       # Per-model metrics, model-agnostic test, human-vs-NLI-CFS
│   └── results/                    # Evaluation reports
├── docs/
│   ├── reference_papers/           # M3SciQA, PaperQA, SciDQA, ScholarlyQA PDFs
│   └── architecture/               # System diagrams
├── paper/
│   ├── scholar_aaai27.tex          # AAAI-27 manuscript source
│   ├── scholar_aaai27.pdf          # Compiled draft
│   └── scholar_references.bib      # Bibliography
├── Makefile                        # `make backend` / `make frontend`
└── requirements.txt
```

---

## Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Node.js v18+
- [Ollama](https://ollama.ai) running locally with `qwen3.5:9b` pulled (`ollama pull qwen3.5:9b`) — text and vision both use this one model

### 1. Clone & configure

```bash
git clone https://github.com/prithvi-kaizen/ScholAR.git
cd ScholAR
```

`backend/.env`:
```
OLLAMA_MODEL=qwen3.5:9b
OLLAMA_BASE_URL=http://localhost:11434
```

### 2. Backend

```bash
# Always run from the ScholAR/ root
make backend
# or manually:
source .venv312/bin/activate
uvicorn backend.main:app --reload --reload-dir backend
```

API runs at **http://localhost:8000**

### 3. Frontend

```bash
make frontend
# or manually:
cd frontend && npm install && npm run dev
```

UI runs at **http://localhost:3000**

---

## Running the Evaluation Suite

```bash
# Retrieval benchmark (14 cases, ~5 sec)
python3 evaluation/run_retrieval_eval.py

# Faithfulness benchmark (51 oracle claims, ~2 min)
python3 evaluation/run_faithfulness_eval.py

# Visual grounding benchmark (18 figure cases)
python3 evaluation/run_visual_eval.py

# Multi-doc locality benchmark (10 clusters)
python3 evaluation/run_multidoc_eval.py
```

All results are written to `evaluation/results/`.

---

## Research Paper

The AAAI-27 draft is in [`paper/scholar_aaai27.pdf`](paper/scholar_aaai27.pdf). It covers:
- Page-preserving chunking design and motivation
- Indirect citation grounding (evidence ID system)
- NLI-CFS faithfulness pipeline and ablation
- Visual grounding architecture
- Multi-document extension and M3SciQA-style evaluation

**Target submission window:** AAAI-27 — abstract due July 21, 2026; full paper due July 28, 2026.

---

## Comparison to Related Work

| System | Multi-modal | Multi-doc | Local inference | Page-grounded citations |
|---|:---:|:---:|:---:|:---:|
| PaperQA2 | ❌ | ✅ | ❌ | ❌ |
| SciDQA | ❌ | ❌ | ❌ | ❌ |
| ScholarlyQA | ❌ | ✅ | ❌ | ❌ |
| M3SciQA (benchmark) | ✅ | ✅ | ❌ | ❌ |
| **ScholAR (ours)** | ✅ | ✅ | ✅ | ✅ |
