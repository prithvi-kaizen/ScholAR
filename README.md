# ScholAR: Page-Preserving Retrieval-Augmented Grounding for Research Paper Comprehension

ScholAR is an academic research assistant designed to facilitate the comprehension of technical scientific documents. It implements a local-first, privacy-preserving **Retrieval-Augmented Generation (RAG)** pipeline optimized specifically for scientific layout structures (such as PDFs). 

By enforcing strict page boundaries during document segmentation and using an indirect citation-grounding mechanism, ScholAR provides reliable, clickable visual citations directly linked to PDF coordinate regions, bypassing the common hallucination issues of standard conversational LLMs.

---

## Key Technical Contributions

### 1. Page-Preserving Chunking
Standard RAG pipelines chunk text based on arbitrary character limits (e.g., recursive character split), which frequently cross page boundaries. In contrast, ScholAR restricts text segmentation boundaries to PDF page limits. This guarantees a strict 1-to-1 mapping between any retrieved text chunk and its visual page, enabling reliable page-level citations.

### 2. Lexical-Heuristic Retrieval (BM25-Primary)
Quantitative evaluations show that standard vector embeddings often fail to retrieve exact result tables and specialized terminology (e.g., BLEU scores, dataset names, model parameters). ScholAR implements:
- **BM25 Lexical Backbone:** The primary ranking signal to ensure exact keyword and number matching.
- **Lightweight Semantic overlap:** Cosine similarity over MD5-hashed term vectors (for lightweight, zero-dependency semantic matching).
- **Heuristic Reranking:** Small ranking boosts for user-specified page hints (e.g., "on pages 6 and 7"), section labels (Abstract, Method, Results), and research-specific phrases (e.g., "we introduce", "we present").

### 3. Indirect Citation Grounding
Instead of allowing the LLM to write inline page numbers (which leads to hallucinated page citations), ScholAR's backend maps retrieved chunks to unique evidence IDs (`E1`, `E2`, etc.) in the LLM prompt. The LLM is forced to cite claims using only these IDs. The backend then resolves the IDs back to the actual PDF coordinates, and the frontend highlights the exact text bounding boxes on the PDF page.

### 4. Hybrid Cloud-Local Inference
The system supports:
- **Local Qwen via Ollama:** For offline, private research paper analysis.
- **Llama 3 via Groq API:** For fast, high-performance reasoning.
- **Graceful Fallback:** If the Groq API hits rate limits, the UI warns the user and falls back to local Qwen, ensuring uninterrupted study sessions.

---

## Repository Structure

```text
ScholAR/
├── backend/                  # Python/FastAPI backend API
│   ├── data/                 # Extracted papers, page images, metadata, and goals
│   ├── services/             # Chunking, PDF parser, retrieval, and LLM APIs
│   └── main.py               # FastAPI entry point
├── frontend/                 # Next.js / TypeScript / Tailwind CSS web application
│   ├── app/                  # Application routing and layout
│   └── components/           # Navbar, PDF viewer, Chat panel, action modules
├── evaluation/               # Quantitative benchmarking suite
│   ├── results/              # Evaluation reports and output JSONs
│   ├── benchmark_cases.json  # Ground-truth queries and relevant chunk mappings
│   ├── run_retrieval_eval.py # Quantitative retrieval evaluator script
│   └── README.md             # Benchmark execution instructions
├── docs/                     # Documentation & Figures
│   ├── architecture/         # PDF and SVG system diagrams
│   └── domain_note.md        # Domain problem analysis & real-world use case
├── .archive/                 # Course project archive
│   └── course_report.md      # Legacy course project report
├── requirements.txt          # Python requirements
└── README.md                 # Main research README
```

---

## Getting Started

### Prerequisites
- Python 3.11 or 3.12
- Node.js (v18+)
- Ollama (running locally with `qwen2.5:7b` or `qwen3.5:9b` loaded)

### 1. Backend Setup
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the FastAPI server
python3 run_backend.py
```
*The API server runs at `http://127.0.0.1:8000`. You can configure environment keys (like `GROQ_API_KEY`) in `backend/.env`.*

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```
*The interface will start at `http://localhost:3000`.*

---

## Quantitative Evaluation & Benchmarking

ScholAR features an integrated retrieval evaluation framework to test the accuracy of the grounding layer. The benchmark runs over **14 manually annotated query-to-chunk test cases** across 3 landmark papers (*Attention Is All You Need*, *RAG*, and *LLaMA*).

### Benchmark Results
The table below compares ScholAR's BM25-primary retriever against key baselines:

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| `keyword_overlap` | 0.571 | 0.786 | 0.929 | 0.687 |
| `bm25_only` | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_no_page_hints` | 0.714 | 0.929 | 1.000 | 0.812 |
| `bm25_primary_with_page_hints` | 0.714 | 0.929 | 1.000 | 0.812 |

To reproduce these results, run the evaluation script:
```bash
python3 evaluation/run_retrieval_eval.py
```
The script outputs raw logs to `evaluation/results/retrieval_eval_results.json` and generates a markdown summary in `evaluation/results/retrieval_eval_report.md`.
