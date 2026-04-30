# ScholAR: Local LLM Research Paper Assistant

ScholAR Milestone 1 is a local-first MVP inspired by paperbreakdown.com. It lets you search arXiv, select a paper, download and process the PDF, view it in a split study workspace, generate study goals, and ask grounded questions using a local Ollama Qwen model.

## Milestone 1 Scope

- arXiv search through the public Atom API
- Local PDF download and page-wise text extraction with PyMuPDF
- Simple word chunking and local JSON storage
- Local Ollama chat and study-goal generation
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

## Known Limitations

- Retrieval is simple keyword-overlap scoring, not embeddings.
- PDF extraction quality depends on the source PDF text layer.
- Ollama calls can be slow on CPU-only machines.
- The PDF toolbar is a Milestone 1 placeholder.
- Bookmarks and recently viewed papers are stored only in browser localStorage.
