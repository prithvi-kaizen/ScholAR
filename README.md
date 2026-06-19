# ScholAR — Smart Companion for Holistic Organization, Literature Analysis & Research

ScholAR is a local-first, privacy-preserving **Retrieval-Augmented Generation (RAG)** system for scientific document comprehension. It is built as a research tool for deep reading and Q&A over academic PDFs, with a focus on faithful, page-grounded citations.

This project is being developed as a thesis-track submission to **AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence).

---

## Key Technical Contributions

### 1. Page-Preserving Chunking
ScholAR segments PDFs by enforcing strict page boundaries. Every text chunk maps exactly to a single PDF page, enabling reliable visual citations. This is implemented as a sliding window over per-page tokens (`target_words=1400`, `overlap=120`), with section title detection via regex.

### 2. BM25-Primary Retrieval with Heuristic Reranking
The retrieval layer uses BM25 as the primary lexical signal, supplemented by lightweight heuristic boosts:
- **Query Expansion:** A curated dictionary expands research-specific terms (e.g., `result` → `accuracy, table, score, BLEU`).
- **Page Hints:** +1.25 score boost when the user query references a specific page number.
- **Section Hints:** Boosts for chunks whose `section_title` matches the semantic category inferred from the query (Abstract, Method, Results, Limitation).
- **Phrase Boosts:** Small boosts for research-writing patterns (e.g., `"we introduce"`, `"we present"`).

### 3. Hybrid BM25 + Dense + RRF Retrieval (Research Evaluation)
For the AAAI-27 faithfulness evaluation, ScholAR implements a full hybrid retrieval pipeline:
- **Dense Ranking:** `all-MiniLM-L6-v2` embeddings loaded directly from the local HuggingFace cache (no `sentence-transformers` library required — pure PyTorch + safetensors).
- **Reciprocal Rank Fusion (RRF, k=60):** Combines BM25 and dense rank lists using the Cormack et al. (SIGIR 2009) formula.
- **Post-Fusion Page Boost:** +0.05 applied to chunks matching user-specified page hints.

### 4. Indirect Citation Grounding
The backend assigns evidence IDs (`E1`, `E2`, ...) to retrieved chunks and prompts the LLM to cite only these IDs — never raw page numbers. The frontend resolves IDs back to PDF bounding box coordinates and highlights the exact source text.

### 5. NLI-Based Citation Faithfulness Evaluation (NLI-CFS)
ScholAR includes a three-tier faithfulness metric aligned with 2024–2026 SOTA practices:

| Tier | Method | Weight |
|---|---|---|
| Tier 1: SummaC-ZS | Sentence-level cosine entailment via MiniLM embeddings | 50% |
| Tier 2: SCR | Whole-claim vs. best-chunk cosine similarity | 30% |
| Tier 3: KFP | Exact recall of numbers and technical terms | 20% |

**Combined CFS = 0.50 × SummaC-ZS + 0.30 × SCR + 0.20 × KFP**

Claims are labeled FAITHFUL (CFS ≥ 0.55), PARTIAL (≥ 0.35), or UNFAITHFUL.

### 6. Hybrid Cloud-Local Inference
- **Local:** Ollama with `qwen3.5:9b` for offline, private analysis.
- **Cloud:** Groq API with `llama-3.3-70b-versatile` for fast inference.
- **Graceful Fallback:** If the Groq API is unavailable, the system falls back to local Ollama without interruption.

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

The faithfulness benchmark spans *Attention Is All You Need*, *RAG*, and *LLaMA* across 8 claim types: `result_number` (13), `technical_claim` (11), `architecture_detail` (10), `training_detail` (8), `conceptual_claim` (5), `human_eval` (2), `formula` (1), `environmental_claim` (1).

---

## Repository Structure

```text
ScholAR/
├── backend/                        # Python/FastAPI backend
│   ├── data/                       # Extracted papers, page images, metadata
│   ├── services/                   # Chunking, retrieval, PDF parser, LLM services
│   └── main.py                     # FastAPI entry point
├── frontend/                       # Next.js / TypeScript / Tailwind CSS UI
│   ├── app/                        # Application routing and layout
│   └── components/                 # PDF viewer, Chat panel, Study panel
├── evaluation/                     # Research evaluation suite
│   ├── embedder.py                 # Local MiniLM-L6-v2 embedder (pure PyTorch)
│   ├── hybrid_retrieval.py         # Hybrid BM25 + Dense + RRF retrieval
│   ├── nli_faithfulness.py         # SummaC-ZS faithfulness scorer
│   ├── run_faithfulness_eval.py    # 51-case faithfulness evaluation runner
│   ├── faithfulness_cases.json     # Oracle-claim benchmark cases
│   └── results/                    # Evaluation reports and JSON outputs
├── paper/                          # AAAI-27 LaTeX manuscript
│   ├── scholar_aaai27.tex          # Main paper source
│   ├── scholar_aaai27.pdf          # Compiled PDF draft
│   └── scholar_references.bib     # Bibliography
├── docs/                           # Architecture diagrams and domain notes
├── .archive/                       # Legacy course project files
├── RESEARCH_ROADMAP.md             # Thesis checklist and future directions
├── run_backend.py                  # Backend launcher script
└── requirements.txt                # Python dependencies
```

---

## Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Node.js v18+
- Ollama running locally with `qwen3.5:9b` loaded

### 1. Backend
```bash
source .venv312/bin/activate
python run_backend.py
```
Configure `GROQ_API_KEY` and `OLLAMA_MODEL` in `backend/.env`.

### 2. Frontend
```bash
cd frontend
npm install
npm run dev
```

The interface runs at `http://localhost:3000` and the API at `http://127.0.0.1:8000`.

---

## Running the Evaluation

```bash
# Retrieval benchmark (14 cases)
python3 evaluation/run_retrieval_eval.py

# Faithfulness benchmark (51 cases — takes ~2 min)
python3 evaluation/run_faithfulness_eval.py
```

Results are written to `evaluation/results/`.

---

## Research Paper

The AAAI-27 draft is in `paper/scholar_aaai27.pdf`. It covers the page-preserving chunking design, indirect citation grounding, the NLI-CFS faithfulness pipeline, and ablation studies over retrieval depth and claim type.

**Target submission window:** AAAI-27, August 2026.
