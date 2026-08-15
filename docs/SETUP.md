# ScholAR Local Setup and First-Day Guide

This guide takes a new contributor from an empty machine to a working ScholAR installation. Follow it in order. You do not need any repository data, paid API key, or cloud model account.

## 1. What runs on your machine

ScholAR has three processes:

| Process | Default address | Purpose |
|---|---|---|
| Ollama | `http://localhost:11434` | Runs the local language and vision model |
| FastAPI backend | `http://localhost:8000` | Imports papers, retrieves evidence, calls Ollama, and serves the API |
| Next.js frontend | `http://localhost:3000` | Provides search, PDF reading, study goals, chat, and references |

The backend stores imported papers under `backend/data/papers/`. That directory, local environment files, model weights, and build directories are excluded from Git.

ScholAR uses the network for GitHub cloning, package installation, arXiv search and downloads, and Semantic Scholar reference resolution. PDF processing and model inference are local after a paper and model have been downloaded.

## 2. Prerequisites

Install these before cloning the repository:

- Git
- Python 3.11 or 3.12. CI uses Python 3.12.
- Node.js 18 or newer. Node.js 20 LTS is recommended and is used by CI.
- npm, which is included with Node.js
- Ollama
- GNU Make for the shortest macOS/Linux workflow

Confirm the tools that are already installed:

```bash
git --version
python3 --version
node --version
npm --version
make --version
ollama --version
```

On Windows, the recommended options are WSL2 with the Linux instructions below or native PowerShell with the commands in [Windows setup](#10-windows-powershell-setup).

## 3. Clone the repository

```bash
git clone https://github.com/prithvi-kaizen/ScholAR.git
cd ScholAR
```

All backend and evaluation commands must be run from this repository root. Running Uvicorn from inside `backend/` breaks package imports.

## 4. Fast setup for macOS and Linux

From the repository root:

```bash
make setup
```

This command:

1. creates `.venv/`;
2. installs the pinned Python runtime dependencies;
3. installs the locked frontend dependencies with `npm ci`;
4. copies both example environment files if local copies do not exist;
5. leaves existing local environment files unchanged.

It does not install Ollama or download a model. Pull the configured default model separately:

```bash
ollama pull qwen3.5:9b
```

If the Ollama desktop application is not already running, start its local service in a terminal:

```bash
ollama serve
```

Do not start a second `ollama serve` process when the desktop application already owns port `11434`.

Run the setup diagnosis:

```bash
make doctor
```

Warnings that the backend or frontend is not running are expected at this stage. Missing dependencies or a missing model include an exact repair command in the output.

## 5. Manual setup

Use these commands when Make is unavailable or when you want to see each installation step.

### Backend

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
cp backend/.env.example backend/.env
```

The project uses a root-level `requirements.txt`. `evaluation/requirements.txt` contains optional packages used only by some research evaluations and is not required to run the application.

### Frontend

```bash
cd frontend
npm ci
cp .env.local.example .env.local
cd ..
```

Use `npm ci`, not `npm install`, for a first setup. It installs the exact dependency tree recorded in `package-lock.json`.

### Ollama

```bash
ollama pull qwen3.5:9b
ollama list
```

The default model is configured in `backend/.env`. If the machine cannot run it comfortably, change `OLLAMA_MODEL` to another installed Ollama model. Text-only models can answer text questions, but figure and table questions require a model that accepts images. Record model changes when producing evaluation results.

## 6. Start ScholAR

Keep each long-running process in its own terminal so its logs stay visible.

### Terminal 1: Ollama

Skip this command if the Ollama desktop application is already running.

```bash
ollama serve
```

### Terminal 2: backend

From the repository root:

```bash
make backend
```

Without Make:

```bash
.venv/bin/python -m uvicorn backend.main:app --reload --reload-dir backend
```

The API and interactive OpenAPI page are available at:

- `http://localhost:8000`
- `http://localhost:8000/docs`

### Terminal 3: frontend

From the repository root:

```bash
make frontend
```

Without Make:

```bash
cd frontend
npm run dev
```

Open `http://localhost:3000`.

## 7. Verify the installation

Check the backend directly:

```bash
curl http://localhost:8000/health
```

A fully ready response looks like:

```json
{
  "status": "ok",
  "ollama_available": true,
  "model": "qwen3.5:9b"
}
```

Then run the repository diagnosis again:

```bash
make doctor
```

Finally, run the code checks:

```bash
make check
make frontend-build
```

`make check` compiles the Python modules and type-checks the frontend. `make frontend-build` verifies the complete production frontend build.

## 8. Import and study the first paper

### Import from arXiv

1. Open `http://localhost:3000`.
2. Search for a paper by title, author, keyword, or arXiv identifier.
3. Open a result and select **Study with AI**.
4. Wait while the backend downloads and prepares the PDF.
5. Ask a question and select a citation to jump to the supporting page.

### Import a local PDF

1. Select the PDF upload control on the home page.
2. Choose a file smaller than 50 MB with a valid PDF header.
3. Wait for text and figure extraction.
4. Open the prepared study workspace.

After either flow, inspect the generated directory:

```text
backend/data/papers/{paper_id}/
├── paper.pdf
├── metadata.json
├── pages.json
├── chunks.json
├── figures.json
└── figures/
```

Some files are created only when needed. Do not commit this directory. It can contain copyrighted or private papers and extracted text.

## 9. Configuration

Backend settings are read from `backend/.env`:

| Variable | Default | Meaning |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server address |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Default generation model |
| `OLLAMA_TOP_P` | `0.9` | Nucleus sampling threshold |
| `OLLAMA_NUM_CTX` | `16000` | Requested context window |
| `OLLAMA_NUM_PREDICT` | `1650` | Maximum generated tokens |
| `OLLAMA_TIMEOUT` | `240` | Request timeout in seconds |

The frontend reads `frontend/.env.local`:

| Variable | Default | Meaning |
|---|---|---|
| `NEXT_PUBLIC_BACKEND_URL` | `http://localhost:8000` | Browser-visible FastAPI origin |

The backend allows browser requests from local frontend ports `3000` and `3001`. If you change the frontend port or host, update the CORS origins in `backend/main.py` as well as the frontend environment file.

Restart the affected service after changing an environment file.

## 10. Windows PowerShell setup

From PowerShell:

```powershell
git clone https://github.com/prithvi-kaizen/ScholAR.git
Set-Location ScholAR

py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
Copy-Item backend\.env.example backend\.env

Set-Location frontend
npm ci
Copy-Item .env.local.example .env.local
Set-Location ..

ollama pull qwen3.5:9b
python scripts\doctor.py
```

Start the backend in one PowerShell window from the repository root:

```powershell
.\.venv\Scripts\Activate.ps1
python -m uvicorn backend.main:app --reload --reload-dir backend
```

Start the frontend in another:

```powershell
Set-Location frontend
npm run dev
```

The Ollama Windows application normally runs the model service in the background.

## 11. Common setup failures

| Symptom | Cause | Fix |
|---|---|---|
| `No module named backend` | Uvicorn was started inside `backend/` | Return to the repository root and run `make backend` |
| `uvicorn: command not found` | The virtual environment is not active | Use `make backend` or `.venv/bin/python -m uvicorn ...` |
| `npm ci` rejects the lockfile | Unsupported or stale Node/npm installation | Install Node.js 20 LTS, then rerun `npm ci` |
| Health says `ollama_available: false` | Ollama is stopped or has a different URL | Start Ollama and check `OLLAMA_BASE_URL` |
| Chat reports that the model is not loaded | Configured model is missing | Run `ollama pull qwen3.5:9b` or change `OLLAMA_MODEL` |
| Browser reports a network error | Backend is stopped or frontend URL is wrong | Check port `8000` and `NEXT_PUBLIC_BACKEND_URL` |
| Browser reports a CORS error | Frontend is using an unlisted origin | Use port `3000`/`3001` or update backend CORS settings |
| Paper search fails | arXiv is unavailable, rate-limited, or blocked | Check the backend log and network connection, then retry |
| PDF preparation fails | The PDF is invalid, too large, inaccessible, or extraction failed | Try a valid PDF under 50 MB and inspect backend logs |
| Figure answer falls back to text | No image was extracted or the model is text-only | Inspect `figures.json` and use a multimodal model |

For a concise machine-generated diagnosis, run:

```bash
make doctor
```

## 12. First contribution checklist

After the application works locally:

1. read `docs/PROJECT_GUIDE.md` for architecture and project constraints;
2. read `docs/EXPERIMENTS.md` before changing research claims or evaluation code;
3. inspect one imported paper's `metadata.json`, `pages.json`, and `chunks.json`;
4. trace one answer citation from the API response to the PDF page;
5. run `make check` and `make frontend-build` before editing;
6. create a branch for the change;
7. follow `CONTRIBUTING.md` and the pull-request template;
8. never commit local PDFs, environment files, evaluator identities, or model data.

You now have the same application boundary and verification commands expected in GitHub CI. Continue with the architecture and research handoff in `docs/PROJECT_GUIDE.md`.
