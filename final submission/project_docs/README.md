# ScholAR: Local LLM Research Paper Assistant

ScholAR Milestone 1 is a local-first MVP inspired by paperbreakdown.com. It lets you search arXiv, select a paper, download and process the PDF, view it in a split study workspace, generate study goals, and ask grounded questions using either a local Ollama Qwen model or a Groq API model.

## Milestone 1 Scope

- arXiv search through the public Atom API
- Local PDF download and page-wise text extraction with PyMuPDF
- Simple word chunking and local JSON storage
- Local Ollama chat and study-goal generation
- Optional Groq API mode from the study panel toggle
- Fallback study goals and extractive responses when Ollama is unavailable
- Dark Next.js UI with search, paper modal, PDF viewer, study goals, chat, bookmarks, and recently viewed papers

Not included in this milestone: login, deployment, similar papers, fine-tuning, knowledge graphs, advanced recommendations, or multi-user storage.

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
OLLAMA_MODEL=qwen3:9b
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
ollama pull qwen3:9b
```

If your machine cannot run `qwen3:9b`, use:

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

## Local or Groq AI Toggle

On the paper study page, use the **Local / Groq** toggle in the right panel.

- **Local** uses Ollama through `OLLAMA_BASE_URL` and `OLLAMA_MODEL`.
- **Groq** uses `GROQ_API_KEY` and `GROQ_MODEL`.
- If Groq is selected but `GROQ_API_KEY` is missing or invalid, ScholAR falls back to default study goals and extractive paper-context answers.

## Free Web Search Tool

Chat can use free DuckDuckGo web search when the question needs outside, recent, or general information. Paper context is still the first source. Paper claims use numbered cited references like `[1]`; web claims are cited as `[web:1]`.

Set `ENABLE_WEB_SEARCH=false` to turn this off.

## Known Limitations

- Retrieval is BM25-primary with lightweight reranking, not a learned embedding retriever.
- PDF extraction quality depends on the source PDF text layer.
- Ollama calls can be slow on CPU-only machines.
- Groq mode requires an internet connection and a valid `GROQ_API_KEY`.
- Web search depends on public search-result pages, so it can be rate-limited or blocked by the network.
- The PDF toolbar is a Milestone 1 placeholder.
- Bookmarks and recently viewed papers are stored only in browser localStorage.
