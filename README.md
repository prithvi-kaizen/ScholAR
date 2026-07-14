# ScholAR — Smart Companion for Holistic Organization, Literature Analysis & Research

ScholAR is a **local-first, privacy-preserving RAG system** for deep reading and Q&A over scientific PDFs. It combines page-grounded citation retrieval, multi-modal visual grounding, and multi-document reasoning into a unified research assistant.

> **Thesis-track submission to AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence). Abstract due July 21, 2026; full paper due July 28, 2026.

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
2. The figure image (rendered page region, rendered at 3× zoom) is base64-encoded and sent to the configured local multimodal model (default `qwen3.5:9b`; `gemma4:12b` also supported) with the question
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

### 7. Fully Local, Model-Agnostic Inference
- **Local by default:** Ollama with `qwen3.5:9b` (natively multimodal) is the default text + vision model — fully offline and private
- **Model-agnostic:** the pipeline takes a per-request model override, and the grounding lives in retrieval + the evidence-ID citation layer, not in the model. The evaluation runs the **same** pipeline across **four local models** — `qwen3.5:9b`, `gemma4:12b`, `llama3.1:8b`, `mistral:7b` — so only the generation model changes (see the generation-faithfulness matrix and the human study)
- **No cloud dependency:** every answer, study goal, and figure/table QA call runs on-device; only public paper acquisition and reference resolution touch the network (arXiv / Semantic Scholar, by identifier)

---

## Evaluation Results

### Retrieval Benchmark — primary (100 cases, 25 papers, auto-labeled)

| System | Recall@1 | Recall@3 | Recall@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| `keyword_overlap` | 0.67 | 0.86 | 0.95 | 0.779 | 0.762 |
| `bm25_only` | 0.81 | 0.91 | 0.94 | 0.863 | 0.828 |
| `bm25_primary` | 0.81 | 0.91 | 0.93 | 0.861 | 0.824 |
| `dense_only` | 0.47 | 0.65 | 0.74 | 0.572 | 0.523 |

At scale **BM25 beats dense** (MRR 0.863 vs 0.572), reversing the 3-paper anchor below and vindicating the BM25-primary design (aligned with BEIR). Gold chunks here are auto-derived from mined facts (weaker than hand labels), and mined queries carry lexical overlap that likely flatters lexical retrieval, so the 3-paper hand-labeled set is kept as the higher-precision anchor.

### Retrieval Benchmark — anchor (14 annotated cases, 3 papers)

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| `keyword_overlap` | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 0.714 | 0.857 | 0.929 | 0.788 |
| `bm25_primary_with_page_hints` | 0.714 | 0.857 | 0.929 | 0.788 |
| `dense_only` (single-pass MiniLM cosine) | **0.786** | **1.000** | **1.000** | **0.881** |

Dense-only tops this tiny set — a genuine small-N artifact that the scaled result above overturns.

### Faithfulness Benchmark (oracle-claim cases, BM25 vs hybrid)

| System | Combined CFS | SCHR@5 | Faithful |
|---|---:|---:|---:|
| BM25-primary (scaled, 100 cases / 25 papers) | 0.785 | 0.860 | 93 / 100 |
| Hybrid BM25 + Dense + RRF (scaled) | 0.782 | 0.750 | 92 / 100 |
| BM25-primary (anchor, 51 cases / 3 papers) | 0.807 | 0.824 | 48 / 51 |
| Hybrid BM25 + Dense + RRF (anchor) | 0.827 | 0.922 | 49 / 51 |

On the scaled set BM25-primary and hybrid land within 0.003 (0.785 vs 0.782), so the reranking heuristics add nothing at scale — another reason ScholAR keeps BM25 primary.

### Baseline Scope

Retrieval/faithfulness baselines above are internal ablations (lexical → dense → hybrid), matching the BEIR-standard comparison framework. We do **not** benchmark against the *hosted* OpenScholar, PaperQA2, or SciRAG: all three depend on cloud-hosted frontier LLM backends (and, for OpenScholar, a tens-of-millions-of-papers datastore), incompatible with ScholAR's local-only constraint. Instead, `run_comparison_eval.py` runs a **resource-matched local comparison** — ScholAR vs long-context PDF-chat, vanilla RAG, and a local reimplementation of PaperQA2's rerank+summarize (RCS) step — all on the same local model (`qwen3.5:9b`) and 91 shared cases:

| System | Gen. Faithfulness | Answer Correctness | Citation F1 |
|---|---:|---:|---:|
| PDF-chat (long-context) | 0.801 | 0.267 | 0.742 |
| Vanilla RAG | 0.860 | 0.340 | 0.802 |
| PaperQA2-style (RCS, local) | 0.872 | 0.550 | 0.782 |
| **ScholAR** | **0.892** | **0.563** | 0.760 |

ScholAR leads on the two axes that matter most (faithfulness, correctness) with a simpler pipeline and page citations valid by construction; the terser baselines reach higher citation precision by citing less. This is a resource-matched local comparison, not a claim to frontier-scale quality. See the paper's Discussion for the full reasoning.

Spans *Attention Is All You Need*, *RAG*, and *LLaMA* across 8 claim types: `result_number` (13), `technical_claim` (11), `architecture_detail` (10), `training_detail` (8), `conceptual_claim` (5), `human_eval` (2), `formula` (1), `environmental_claim` (1).

### Visual Grounding Benchmark (18 figure-grounded cases, 4 reasoning types)

| Metric | Score |
|---|---:|
| Retrieval R@5 | **18 / 18 = 1.000** |
| Caption fallback rate | 0 / 18 (0%) |

### Multi-Document Localization — evaluated on **M3SciQA** (297 labeled locality cases)

M3SciQA's *locality* task is exactly ours: given an anchor paper and a question about one of its figures, rank the anchor's bibliography (mean 47.9 candidate references) to find the reference that answers it. Their gold labels and metric let us be measured directly against their published baselines.

| System | MRR | R@5 |
|---|---:|---:|
| Human expert † | 0.796 | — |
| GPT-4o † (cloud) | 0.500 | — |
| **ScholAR + local vision (`qwen3.5:9b`)** | **0.474** | **0.606** |
| **ScholAR + local vision (`gemma4:12b`)** | 0.455 | 0.572 |
| GPT-4V † (cloud) | 0.400 | — |
| text-embedding-3-large † (cloud) | 0.297 | — |
| ScholAR BM25-primary (text only) | 0.180 | 0.242 |
| Best open-source LMM † | 0.144 | — |
| BM25 † / Random † | 0.127 / 0.126 | — |

† published by M3SciQA; the rest measured here. Our empirical random floor is 0.121, matching their 0.126 — the setup reproduces theirs.

**The bottleneck was never retrieval.** From the question text alone, BM25 sits near chance (0.180 vs a 0.121 floor), matching their BM25 (0.127) and explaining our old 18-case result (0.183): the question names an entity that exists **only inside a figure**. Resolving that figure with the local multimodal model first lifts MRR to **0.474** — 3.3× the best open-source LMM they report, past GPT-4V, and within 0.026 of GPT-4o, on a laptop. Expert humans (0.796) remain far ahead, so the task is not solved.

Caveat: our pipeline *decomposes* the task (vision resolves the entity, then BM25 ranks the bibliography) whereas M3SciQA prompt their multimodal baselines to rank papers directly — a comparison of systems on a shared task and metric, not one identical protocol.

Run it: `python3 evaluation/m3sciqa/build_m3sciqa.py && python3 evaluation/m3sciqa/run_m3sciqa_eval.py --tier text`

### Abstention Benchmark (20 provably-unanswerable questions)

Each paper-specific question is posed against a *different* paper in which the queried fact is provably absent (exact-substring check), so the correct response is to decline. A grounded system should abstain, not fabricate.

| Model | Abstain | Fabricate |
|---|---:|---:|
| `qwen3.5:9b` | 1.00 | 0.00 |
| `llama3.1:8b` | 1.00 | 0.00 |
| `gemma4:12b` | 0.95 | 0.05 |
| `mistral:7b` | 0.90 | 0.10 |

The few non-abstentions are a benign overlap (the wrongly paired paper independently reports a minibatch size), not free invention. Abstention is scored by a refusal detector validated against a manual read of every output. Small and synthetic, so read it as a tendency, not a precise rate.

### Efficiency (per query, Apple Silicon 18 GB)

| Model | Mem (GB) | Tok/s | Latency mean / p95 (s) |
|---|---:|---:|---:|
| `mistral:7b` | 6.5 | 24.9 | 6.1 / 8.4 |
| `llama3.1:8b` | 6.9 | 26.9 | 6.3 / 17.8 |
| `qwen3.5:9b` | 6.1 | 21.9 | 11.4 / 14.6 |
| `gemma4:12b` | 8.1 | 16.3 | 8.8 / 14.2 |

BM25 retrieval is ~40 ms and model-independent, so latency is generation-bound. Every model fits the 18 GB budget and answers in single-digit to ~11 s — no cloud, no GPU cluster.

### Human Evaluation Pipeline (built, ready to run)

A model-agnostic, citation-grounded human evaluation lives in `evaluation/human_eval/`, grounded in the methodology of OpenScholar, SciRAG, and PaperQA2. It runs over a diverse 100-case benchmark (50 text, 25 mathematical, 25 figure/table questions across 25 papers), mined from the corpus and source-verified. Each question is answered by four local models (`qwen3.5:9b`, `gemma4:12b`, `llama3.1:8b`, `mistral:7b`) running the same ScholAR pipeline — only the generation model changes. Expert evaluators score each answer on four anchored 1-5 dimensions (Relevance, Coverage, Faithfulness, Usefulness) plus per-citation Supported/Partial/Unsupported grading, through a self-contained offline HTML interface. The analysis reports per-model quality and citation precision/recall/F1, a Friedman test for model-agnostic grounding, inter-annotator agreement, and the correlation between human faithfulness and the automated NLI-CFS metric. See `evaluation/human_eval/README.md` to run it.

The automated counterpart, `run_generation_faithfulness_matrix.py`, scores each of the four models' **generated answers** for grounding + citation support with no human involvement — the automated sibling of the human study.

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
│   ├── run_faithfulness_eval.py    # 51-case retrieval-support faithfulness benchmark
│   ├── run_generation_faithfulness_eval.py    # faithfulness of GENERATED answers (single model)
│   ├── run_generation_faithfulness_matrix.py  # 4-model automated faithfulness matrix
│   ├── run_comparison_eval.py      # ScholAR vs local baselines (pdfchat/vanilla-RAG/PaperQA2-RCS)
│   ├── run_efficiency_eval.py      # Per-model latency / throughput / memory footprint
│   ├── run_abstention_eval.py      # Abstention vs fabrication on unanswerable questions
│   ├── run_visual_eval.py          # 18-case visual grounding benchmark
│   ├── run_visual_caption_ablation.py         # caption-only vs full-vision ablation
│   ├── run_multidoc_eval.py        # Multi-doc locality benchmark
│   ├── run_multidoc_bounds_eval.py # Multi-doc oracle / random-floor bounds
│   ├── mine_cases.py               # Mines + source-verifies the diverse 100-case benchmark
│   ├── build_scaled_benchmark.py   # Auto-labels the 100-case set into scaled retrieval + faithfulness
│   ├── build_abstention_benchmark.py           # Builds 20 provably-unanswerable cross-paper negatives
│   ├── benchmark_cases.json        # Retrieval ground truth (3-paper anchor)
│   ├── benchmark_cases_scaled.json # Retrieval ground truth (25-paper, auto-labeled)
│   ├── faithfulness_cases.json     # Faithfulness oracle claims (3-paper anchor)
│   ├── faithfulness_cases_scaled.json          # Faithfulness claims (25-paper, auto-labeled)
│   ├── abstention_cases.json       # Provably-unanswerable negatives
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
│   ├── reference_papers/           # Domain/AAAI PDFs for tone + related work (gitignored): OpenScholar, PaperQA2, SciRAG, Pleias-RAG, ColPali, M3SciQA, SummaC, FActScore, RAGAS, ALCE, RGB
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
- [Ollama](https://ollama.ai) running locally with `qwen3.5:9b` pulled (`ollama pull qwen3.5:9b`) — the default text + vision model. The evaluation additionally uses `gemma4:12b`, `llama3.1:8b`, and `mistral:7b` (`ollama pull <model>`); any of them can serve the app via the `OLLAMA_MODEL` env var or a per-request override

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
# Retrieval benchmark (14 cases, ~5 sec) — no model needed
python3 evaluation/run_retrieval_eval.py

# Retrieval-support faithfulness (51 oracle claims, ~2 min) — no model needed
python3 evaluation/run_faithfulness_eval.py

# Multi-doc locality + oracle/floor bounds (10 arXiv-resolvable cases) — no model needed
python3 evaluation/run_multidoc_eval.py
python3 evaluation/run_multidoc_bounds_eval.py

# Visual grounding + caption ablation (18 figure cases) — needs a local multimodal model
python3 evaluation/run_visual_eval.py
python3 evaluation/run_visual_caption_ablation.py

# Generation faithfulness of actual answers — needs the backend + a model
python3 evaluation/run_generation_faithfulness_eval.py                 # single model
python3 evaluation/run_generation_faithfulness_matrix.py --models qwen3.5:9b   # per model; accumulates a 4-model matrix

# ScholAR vs local baselines (pdfchat / vanilla-RAG / PaperQA2-RCS), one system at a time
python3 evaluation/run_comparison_eval.py --systems scholar --cases both --limit 40
```

All results are written to `evaluation/results/`. The scripts that generate answers use the local model(s) via Ollama; the retrieval/faithfulness/multi-doc scripts run standalone.

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

| System | Multi-modal | Multi-doc | Local inference | Deterministic page-grounded citations¹ |
|---|:---:|:---:|:---:|:---:|
| PaperQA2 | ❌ | ✅ | ❌ | ❌ |
| SciDQA | ❌ | ❌ | ❌ | ❌ |
| ScholarlyQA | ❌ | ✅ | ❌ | ❌ |
| M3SciQA (benchmark) | ✅ | ✅ | ❌ | ❌ |
| **ScholAR (ours)** | ✅ | ✅ | ✅ | ✅ |

¹ PaperQA2 emits model-written page citations (e.g. `(pages 3-4)`); ScholAR's citations are evidence IDs mapped to pages by the application, so the page reference cannot be a model guess. This is a feature comparison, not a head-to-head quality benchmark.
