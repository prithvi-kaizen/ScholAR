# ScholAR — Smart Companion for Holistic Organization, Literature Analysis & Research

ScholAR is a local-first, privacy-preserving **Retrieval-Augmented Generation (RAG)** system for scientific document comprehension. It is built as a research tool for deep reading and Q&A over academic PDFs, with a focus on faithful, page-grounded citations, visual figure understanding, and multi-document reasoning.

This project is being developed as a thesis-track submission to **AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence).

---

## Key Technical Contributions

### 1. Page-Preserving Chunking
ScholAR segments PDFs by enforcing strict page boundaries. Every text chunk maps exactly to a single PDF page, enabling reliable visual citations. Implemented as a sliding window over per-page tokens (`target_words=1400`, `overlap=120`) with section title detection via regex.

### 2. BM25-Primary Retrieval with Heuristic Reranking
The retrieval layer uses BM25 as the primary lexical signal, supplemented by lightweight heuristic boosts:
- **Query Expansion:** Research-specific term expansion (e.g., `result` → `accuracy, table, score, BLEU`).
- **Page Hints:** +1.25 score boost when the query references a specific page number.
- **Section Hints:** Boosts for chunks whose `section_title` matches the semantic category inferred from the query (Abstract, Method, Results, Limitation).
- **Visual Cue Boosting:** Queries containing figure/table keywords boost figure chunks by +1.5.

### 3. Hybrid BM25 + Dense + RRF Retrieval
ScholAR implements a full hybrid retrieval pipeline for evaluation:
- **Dense Ranking:** `all-MiniLM-L6-v2` embeddings (pure PyTorch + safetensors, no `sentence-transformers` required).
- **Reciprocal Rank Fusion (RRF, k=60):** Combines BM25 and dense ranks (Cormack et al., SIGIR 2009).
- **Post-Fusion Page Boost:** +0.05 applied to chunks matching user-specified page hints.

### 4. Visual Grounding — Figure & Table QA
ScholAR extracts and indexes figures and tables directly from PDFs and answers figure-specific questions using a vision-language model:
- **Figure extraction:** Caption-anchor heuristic (regex `Figure|Table|Fig\.`) detects figures from text; page-region rendering clips the surrounding image area.
- **Visual-aware retrieval:** Figure chunks are stored with `is_figure=True`, `caption`, `label`, and `image_file` metadata. Visual-cue queries (containing "figure", "show", "plot", etc.) apply a +1.5 retrieval boost.
- **Vision-LLM answer path:** Detected figure queries are routed to `meta-llama/llama-4-scout-17b-16e-instruct` (Groq) via multimodal base64 image input. A caption-fallback path handles cases where image rendering fails.
- **Frontend:** Figure thumbnails rendered inline in the citation panel with teal-bordered cards; clicking jumps to the source page.

**Visual Grounding Evaluation (18 benchmark cases, 4 papers):**

| Question Type | R@5 | Pass@1 (vision) |
|---|---:|---:|
| Figure description | 1.000 | — |
| Table lookup | 1.000 | — |
| Cross-figure comparison | 1.000 | — |
| Architecture diagram | 1.000 | — |
| **Overall** | **1.000** | — |

### 5. Multi-Document Extension (Anchor Paper + Bibliography)
ScholAR supports multi-document QA over an anchor paper and its cited references:
- **Reference resolution:** For arXiv papers, resolved via Semantic Scholar API by arXiv ID. For **uploaded PDFs**, resolved via S2 title-search (`/paper/search?query=<title>`) — the first system to support S2 resolution for non-arXiv uploads.
- **Reference ingestion:** Users can load individual cited papers (PDF download + chunking + indexing) or batch-ingest all available references with a single click.
- **Cross-paper retrieval:** During chat, all ingested secondary paper chunks are included in retrieval; citations are tagged with `source_paper_id` and a `ref` badge in the UI.
- **Reference section parsing:** Supports both numbered (`[1] Vaswani...`) and author-year (`Vaswani et al. (2017)...`) bibliography formats; scans last 6 pages.

**Multi-Document Evaluation (locality benchmark, 51 questions, 8 paper clusters):**

| System | Locality R@5 | MRR |
|---|---:|---:|
| ScholAR multi-doc | 0.80 | 0.338 |

### 6. Indirect Citation Grounding
The backend assigns evidence IDs (`E1`, `E2`, ...) to retrieved chunks and prompts the LLM to cite only these IDs. The frontend resolves IDs back to PDF page coordinates and highlights the exact source text.

### 7. NLI-Based Citation Faithfulness Evaluation (NLI-CFS)
A three-tier faithfulness metric aligned with 2024–2026 SOTA practices:

| Tier | Method | Weight |
|---|---|---|
| Tier 1: SummaC-ZS | Sentence-level cosine entailment via MiniLM embeddings | 50% |
| Tier 2: SCR | Whole-claim vs. best-chunk cosine similarity | 30% |
| Tier 3: KFP | Exact recall of numbers and technical terms | 20% |

**Combined CFS = 0.50 × SummaC-ZS + 0.30 × SCR + 0.20 × KFP**

Claims are labeled FAITHFUL (CFS ≥ 0.55), PARTIAL (≥ 0.35), or UNFAITHFUL.

### 8. Hybrid Cloud-Local Inference
- **Local:** Ollama with `qwen3:8b` for offline, private analysis.
- **Cloud:** Groq API with `llama-3.3-70b-versatile` for fast text inference; `llama-4-scout-17b-16e-instruct` for vision (figure QA).
- **Graceful Fallback:** If Groq is unavailable, falls back to local Ollama without interruption.

---

## Evaluation Results

### Retrieval Benchmark (14 manually annotated cases, 3 papers)

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| `keyword_overlap` | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_no_page_hints` | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_with_page_hints` | 0.714 | 0.929 | 1.000 | 0.812 |

### Faithfulness Benchmark (51 oracle-claim cases, 3 papers, 8 claim types)

| System | Combined CFS | SCHR@5 | Faithful |
|---|---:|---:|---:|
| BM25-primary | 0.820 | 0.863 | 49 / 51 |
| Hybrid BM25 + Dense + RRF | 0.829 | 0.922 | 49 / 51 |

Spans *Attention Is All You Need*, *RAG*, and *LLaMA* across 8 claim types: `result_number` (13), `technical_claim` (11), `architecture_detail` (10), `training_detail` (8), `conceptual_claim` (5), `human_eval` (2), `formula` (1), `environmental_claim` (1).

### Visual Grounding Benchmark (18 cases, 4 papers)

| Metric | Score |
|---|---:|
| Retrieval R@5 | **18/18 = 1.000** |
| Question types covered | Figure description, table lookup, architecture diagram, cross-figure comparison |

---

## Repository Structure

```text
ScholAR/
├── backend/                        # Python/FastAPI backend
│   ├── data/                       # Extracted papers, page images, metadata
│   ├── services/
│   │   ├── chunking_service.py     # Page-preserving chunking + figure chunking
│   │   ├── retrieval_service.py    # BM25 + visual-cue boosting + hybrid RRF
│   │   ├── reference_service.py    # Multi-doc reference resolution (S2 + arXiv)
│   │   ├── vision_service.py       # Visual grounding orchestration
│   │   ├── groq_service.py         # Groq text + vision inference
│   │   ├── pdf_service.py          # PDF parsing, page rendering, figure extraction
│   │   └── arxiv_service.py        # arXiv search and metadata fetch
│   └── main.py                     # FastAPI entry point and all API routes
├── frontend/                       # Next.js 15 / TypeScript / Tailwind CSS
│   ├── app/                        # Application routing and layout
│   └── components/
│       ├── StudyWorkspace.tsx      # Resizable split-panel layout (drag handle)
│       ├── StudyPanel.tsx          # Tabbed right panel (Chat / Goals / References)
│       ├── ChatBox.tsx             # Chat UI with typing animation, model badge, suggestions
│       ├── PdfViewer.tsx           # PDF page viewer with citation highlight
│       ├── ReferencesPanel.tsx     # Multi-doc reference browser with ingest progress
│       └── StudyGoals.tsx          # AI-generated study goals
├── evaluation/                     # Research evaluation suite
│   ├── embedder.py                 # Local MiniLM-L6-v2 embedder (pure PyTorch)
│   ├── hybrid_retrieval.py         # Hybrid BM25 + Dense + RRF retrieval
│   ├── nli_faithfulness.py         # SummaC-ZS faithfulness scorer
│   ├── run_faithfulness_eval.py    # 51-case faithfulness evaluation runner
│   ├── run_visual_eval.py          # 18-case visual grounding evaluation runner
│   ├── visual_benchmark.json       # Visual grounding benchmark cases
│   ├── benchmark_cases.json        # Retrieval benchmark (14 annotated cases)
│   ├── faithfulness_cases.json     # Faithfulness benchmark (51 oracle-claim cases)
│   └── results/                    # Evaluation reports and JSON outputs
├── paper/                          # AAAI-27 LaTeX manuscript
│   ├── scholar_aaai27.tex          # Main paper source
│   └── scholar_references.bib      # Bibliography
├── docs/                           # Documentation and research references
│   ├── reference_papers/           # Related papers (PaperQA, SCIDQA, M3SciQA, etc.)
│   └── domain_note.md              # Domain problem analysis
├── RESEARCH_ROADMAP.md             # Thesis checklist and future directions
├── Makefile                        # `make backend` shortcut
└── requirements.txt                # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Node.js v18+
- Ollama running locally with `qwen3:8b` loaded
- Groq API key (optional, for cloud inference and vision QA)

### 1. Backend
```bash
# Always run from ScholAR/ root, never from inside backend/
make backend
```

Or manually:
```bash
source .venv312/bin/activate
uvicorn backend.main:app --reload --reload-dir backend
```

Configure in `backend/.env`:
```env
GROQ_API_KEY=your_key_here
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
OLLAMA_MODEL=qwen3:8b
```

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

The interface runs at **http://localhost:3000** and the API at **http://127.0.0.1:8000**.

---

## Running Evaluations

```bash
# Retrieval benchmark (14 cases)
python3 evaluation/run_retrieval_eval.py

# Faithfulness benchmark (51 cases — ~2 min)
python3 evaluation/run_faithfulness_eval.py

# Visual grounding benchmark (18 cases)
python3 evaluation/run_visual_eval.py
```

Results are written to `evaluation/results/`.

---

## Research Paper

The AAAI-27 manuscript is in `paper/scholar_aaai27.tex`. It covers:
- Page-preserving chunking design
- Indirect citation grounding
- Visual figure grounding pipeline
- Multi-document reference extension
- NLI-CFS faithfulness metric and ablation studies

**Target submission window:** AAAI-27, August 2026.
