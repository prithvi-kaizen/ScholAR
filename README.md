# ScholAR — Smart Companion for Holistic Organization, Literature Analysis & Research

ScholAR is a **local-first, privacy-preserving RAG system** for deep reading and Q&A over scientific PDFs. It combines page-grounded citation retrieval, multi-modal visual grounding, and multi-document reasoning into a unified research assistant.

> **Thesis-track submission to AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence, August 2026).

---

## What ScholAR Does

- **Upload or search any arXiv paper** → instant ingestion, chunking, and indexing
- **Ask questions in natural language** → page-grounded answers with source citations
- **Visual grounding** → queries about figures/tables route to a vision-LLM that reads the actual image
- **Multi-document mode** → load cited references and ask cross-paper questions
- **Local or cloud inference** → Ollama (offline) or Groq (fast cloud), switchable mid-session

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
2. The figure image (rendered page region) is base64-encoded and sent to **Groq Llama 4 Scout** (vision-capable) with the question
3. Caption-only fallback if the image is unavailable or too small
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

### 7. Hybrid Cloud-Local Inference
- **Local:** Ollama with `qwen3:9b` — fully offline, private
- **Cloud:** Groq API with `llama-3.3-70b-versatile` — fast cloud inference
- **Vision:** Groq `meta-llama/llama-4-scout-17b-16e-instruct` — for figure/table QA
- **Graceful fallback:** Groq unavailable → automatic switch to local Ollama

---

## Evaluation Results

### Retrieval Benchmark (14 annotated cases, 3 papers)

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| `keyword_overlap` | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_with_page_hints` | 0.714 | 0.929 | 1.000 | 0.812 |

### Faithfulness Benchmark (51 oracle-claim cases, 3 papers, 8 claim types)

| System | Combined CFS | SCHR@5 | Faithful |
|---|---:|---:|---:|
| BM25-primary | 0.820 | 0.863 | 49 / 51 |
| Hybrid BM25 + Dense + RRF | 0.829 | 0.922 | 49 / 51 |

Spans *Attention Is All You Need*, *RAG*, and *LLaMA* across 8 claim types: `result_number` (13), `technical_claim` (11), `architecture_detail` (10), `training_detail` (8), `conceptual_claim` (5), `human_eval` (2), `formula` (1), `environmental_claim` (1).

### Visual Grounding Benchmark (18 figure-grounded cases, 4 reasoning types)

| Metric | Score |
|---|---:|
| Retrieval R@5 | **18 / 18 = 1.000** |
| Caption fallback rate | 2 / 18 (11%) |

### Multi-Document Locality Benchmark (10 paper clusters, M3SciQA-style)

| Metric | Score |
|---|---:|
| Locality R@5 | 0.80 |
| MRR | 0.338 |

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
│   │   ├── llm_service.py          # Groq / Ollama LLM routing
│   │   ├── vision_service.py       # Figure QA via Groq Llama 4 Scout vision
│   │   ├── reference_service.py    # Multi-doc: S2 API + title-search for uploads
│   │   └── arxiv_service.py        # arXiv search and paper metadata
│   ├── data/                       # Per-paper extracted data (gitignored)
│   └── .env                        # API keys (not committed)
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
- [Ollama](https://ollama.ai) running locally with `qwen3:9b` pulled
- Groq API key (optional, but recommended for speed)

### 1. Clone & configure

```bash
git clone https://github.com/prithvi-kaizen/ScholAR.git
cd ScholAR
cp backend/.env.example backend/.env   # then fill in your keys
```

`backend/.env`:
```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama-3.3-70b-versatile
GROQ_VISION_MODEL=meta-llama/llama-4-scout-17b-16e-instruct
OLLAMA_MODEL=qwen3:9b
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

**Target submission window:** AAAI-27, August 2026.

---

## Comparison to Related Work

| System | Multi-modal | Multi-doc | Local inference | Page-grounded citations |
|---|:---:|:---:|:---:|:---:|
| PaperQA2 | ❌ | ✅ | ❌ | ❌ |
| SciDQA | ❌ | ❌ | ❌ | ❌ |
| ScholarlyQA | ❌ | ✅ | ❌ | ❌ |
| M3SciQA (benchmark) | ✅ | ✅ | ❌ | ❌ |
| **ScholAR (ours)** | ✅ | ✅ | ✅ | ✅ |
