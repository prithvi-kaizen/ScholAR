# ScholAR: Research Paper Study Assistant

ScholAR is a GenAI research paper assistant for studying technical papers with grounded evidence. It lets you search arXiv papers, upload custom PDFs, view the paper side by side with an AI study panel, generate paper-specific study goals, ask questions, and inspect cited references from the paper.

The project is local-first, but it supports both local Ollama Qwen models and Groq API models. The final retrieval design uses BM25 as the primary grounding method because the project evaluation showed BM25 was the most reliable tested retriever.

## Current Scope

- arXiv search through the public Atom API
- Custom PDF upload
- Local PDF download and page-wise text extraction with PyMuPDF
- Page-preserving chunking and local JSON storage
- BM25-primary retrieval with lightweight reranking
- Paper-specific study goals with recursive subquestions
- Local Ollama Qwen chat
- Optional Groq API mode from the study panel toggle
- Groq limit warning and local Qwen fallback
- Numbered cited references linked to paper evidence
- Free web search for outside or current questions
- Bookmarks and recently viewed papers stored in localStorage
- Dark UI and matte beige light mode

Not included: login, deployment, similar papers, fine-tuning, knowledge graphs, advanced recommendations, or multi-user storage.

## Final Submission Files

The final submission package is in:

```text
final submission/
```

Important files:

- `final submission/reports/FINAL_WRITTEN_REPORT.md`
- `final submission/architecture/ScholAR_architecture_flow.png`
- `final submission/evaluation/retrieval_eval_report.md`
- `final submission/evaluation/retrieval_eval_results.json`
- `final submission/evaluation/benchmark_cases.json`
- `evaluation/run_retrieval_eval.py`

## Setup

```bash
cd project-paperbreakdown
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Use Python 3.11 or 3.12 for the pinned Milestone 1 dependencies. If `python3.11` is not installed, `python3.12 -m venv .venv` works as well.

Install the frontend dependencies:

```bash
cd frontend
npm install
```

Create optional environment files:

```bash
# backend/.env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3.5:9b
GROQ_API_KEY=
GROQ_MODEL=llama-3.3-70b-versatile
ENABLE_WEB_SEARCH=true
WEB_SEARCH_RESULTS=5

# frontend/.env.local
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

## Run Backend

```bash
cd project-paperbreakdown
source .venv/bin/activate
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/health
```

## Run Frontend

```bash
cd project-paperbreakdown/frontend
npm run dev
```

Open `http://localhost:3000`.

## Start Ollama

Install Ollama, then run:

```bash
ollama serve
ollama pull qwen3.5:9b
```

If your machine cannot run `qwen3.5:9b`, use:

```bash
ollama pull qwen2.5:7b
OLLAMA_MODEL=qwen2.5:7b uvicorn backend.main:app --reload --port 8000
```

## Example Search Query

Try:

```text
retrieval augmented generation survey
```

Flow: search arXiv, open a paper card, click **Study with AI**, wait for preparation, then ask a question in the study panel.

You can also use **Upload PDF** on the home page to study a custom paper.

## Local or Groq AI Toggle

On the paper study page, use the **Local / Groq** toggle in the right panel.

- **Local** uses Ollama through `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- **Groq** uses `GROQ_API_KEY` and `GROQ_MODEL`.
- If Groq is selected but `GROQ_API_KEY` is missing or invalid, ScholAR falls back to default study goals and extractive paper-context answers.

## Free Web Search Tool

Chat can use free DuckDuckGo web search when the question needs outside, recent, or general information. Paper context is still the first source. Paper claims use numbered cited references like `[1]`; web claims are cited as `[web:1]`.

Set `ENABLE_WEB_SEARCH=false` to turn this off.

## Retrieval and Evaluation

ScholAR originally used a more aggressive hybrid retrieval score. The quantitative evaluation showed that BM25 was more reliable for finding correct evidence chunks, especially for exact result and table questions. Because of that, the app now uses BM25 as the main retrieval signal and keeps semantic overlap, page hints, section hints, and phrase hints as small reranking boosts.

Run the evaluation:

```bash
python3 evaluation/run_retrieval_eval.py
```

Current evaluation summary on 14 manually checked retrieval cases:

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| keyword overlap | 0.571 | 0.786 | 0.929 | 0.687 |
| BM25 only | 0.714 | 0.929 | 1.000 | 0.812 |
| BM25-primary without page hints | 0.714 | 0.929 | 1.000 | 0.812 |
| BM25-primary with page hints | 0.714 | 0.929 | 1.000 | 0.812 |

## Known Limitations

- Retrieval is BM25-primary with lightweight reranking, not a learned embedding retriever.
- PDF extraction quality depends on the source PDF text layer.
- Citation highlighting can still fail when PDF text extraction differs from visual PDF text.
- Ollama calls can be slow on CPU-only machines.
- Groq mode requires an internet connection and a valid `GROQ_API_KEY`.
- Web search depends on public search-result pages, so it can be rate-limited or blocked by the network.
- The PDF toolbar is a Milestone 1 placeholder.
- Bookmarks and recently viewed papers are stored only in browser localStorage.
