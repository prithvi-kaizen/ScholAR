# Local setup

Architecture is in [CODEBASE.md](CODEBASE.md); the exact data and answer path is in
[PIPELINE.md](PIPELINE.md).

## Requirements

- Python 3.12 exactly
- Node.js 20 and npm 10
- Make on macOS/Linux
- Ollama only for generated text or visual answers

## Install

```bash
bash scripts/quickstart.sh
```

This creates `.venv`, installs `requirements/locks/base-py312.txt`, runs `npm ci`, copies
missing environment templates, and runs the setup doctor. It does not silently download
models, papers, datasets, or Docling assets.

Equivalent explicit setup:

```bash
make setup
make doctor
```

Optional layers:

```bash
make setup-parser      # Docling packages
make setup-evaluation  # PyTorch/Transformers evaluation stack
make models            # interactive Ollama selection/acquisition
```

## Configuration

Important `backend/.env` defaults:

| Variable | Default | Meaning |
|---|---:|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Local model endpoint |
| `OLLAMA_MODEL` | `qwen3.5:9b` | Generation/vision model |
| `OLLAMA_TOP_P` | `0.9` | Nucleus sampling |
| `OLLAMA_NUM_CTX` | `16000` | Requested context |
| `OLLAMA_NUM_PREDICT` | `1650` | Maximum output tokens |
| `OLLAMA_TIMEOUT` | `240` | Request timeout seconds |
| `SCHOLAR_NETWORK_MODE` | `strict-local` | Runtime network boundary |
| `SCHOLAR_VISUAL_EMBEDDING_MODEL` | `openai/clip-vit-base-patch32` | Crop and legacy page visual encoder |
| `SCHOLAR_VISUAL_MIN_SIMILARITY` | `0.20` | Crop routing floor |
| `SCHOLAR_VISUAL_PAGE_MIN_SIMILARITY` | `0.12` | Page MaxSim floor |
| `SCHOLAR_VISUAL_PAGE_BACKEND` | `auto` | `colqwen2`, `clip`, `auto`, or `disabled` |
| `SCHOLAR_DOCUMENT_VISUAL_MODEL` | `vidore/colqwen2-v1.0-hf` | Document-trained page retriever snapshot |
| `SCHOLAR_DOCUMENT_VISUAL_DEVICE` | `auto` | ColQwen2 device selection |
| `SCHOLAR_COLQWEN_MIN_SCORE` | `-1000000000` | Uncalibrated MaxSim routing floor |
| `DOCLING_ARTIFACTS_PATH` | empty | Optional local parser assets |

`frontend/.env.local` normally contains:

```dotenv
NEXT_PUBLIC_BACKEND_URL=http://localhost:8000
```

`strict-local` permits prepared files, cached encoders, browser/backend loopback traffic,
and loopback Ollama. It blocks arXiv, reference, paper, model, and dataset acquisition.
Use `acquisition-enabled` temporarily for an explicit acquisition operation, then switch
back before analysis or evaluation.

Text retrieval uses cached `all-MiniLM-L6-v2` or its deterministic fallback. Crop
retrieval needs the configured CLIP snapshot. Full-page retrieval prefers the configured
ColQwen2 snapshot in `auto` mode and records a CLIP fallback when it is absent. Both model
paths are cache-only during analysis. After provisioning ColQwen2 separately, run
`make visual-index` to build page indexes before interactive use. Docling is used in
strict-local mode only when `DOCLING_ARTIFACTS_PATH` is already provisioned.

## Run

```bash
# optional, if Ollama is not already running
ollama serve

# terminal 2
make backend

# terminal 3
make frontend
```

- App: `http://localhost:3000`
- API: `http://localhost:8000`
- OpenAPI: `http://localhost:8000/docs`

Local upload works in either network mode. arXiv and reference acquisition require
`acquisition-enabled`. Prepared papers are published under
`backend/data/papers/{paper_id}/` and are ignored by Git.

Upgrade older papers with locally derived page images and visual metadata using:

```bash
.venv/bin/python scripts/migrate_visual_artifacts.py --help
```

## Validate

```bash
make doctor
make check
make test
make smoke
make frontend-build
make paper-verify
```

## Common failures

| Symptom | Resolution |
|---|---|
| Lock installation fails | Verify Python is exactly 3.12 |
| Chat uses extractive fallback | Start Ollama and install/configure the requested tag |
| Visual channel unavailable | Provision the configured CLIP snapshot locally |
| Pixel question abstains | Use a vision-capable model and migrate/reingest visual units |
| Docling falls back | Install parser layer and configure local assets, or accept PyMuPDF mode |
| arXiv action returns 409 | Enable acquisition only for that explicit operation |
| Frontend cannot reach API | Check `NEXT_PUBLIC_BACKEND_URL` and local CORS ports |
