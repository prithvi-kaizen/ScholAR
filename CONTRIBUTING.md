# Contributing to ScholAR

Thank you for improving ScholAR. This is a research codebase, so a good contribution must preserve both software behavior and the integrity of the evidence reported about that behavior.

Start with [docs/SETUP.md](docs/SETUP.md) to install and verify the application. Then read [docs/PROJECT_GUIDE.md](docs/PROJECT_GUIDE.md) and [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) before changing retrieval, prompts, citation handling, scorers, or benchmarks.

## Local setup

```bash
git clone https://github.com/prithvi-kaizen/ScholAR.git
cd ScholAR
make setup
ollama pull qwen3.5:9b
make doctor
```

Run the backend from the repository root:

```bash
make backend
```

Run the frontend in a second terminal:

```bash
make frontend
```

Open `http://localhost:3000` and import one paper before making a first change. Windows and manual setup commands are maintained in [docs/SETUP.md](docs/SETUP.md).

## Before opening a change

- Keep the change focused and explain the user or research problem it solves.
- Preserve backward-compatible API fields unless the same change updates every consumer.
- Do not commit `backend/data/papers/`, local PDFs, model weights, virtual environments, `.env` files, evaluator identities, or generated score sheets.
- Do not edit benchmark labels to make a system score better.
- Do not silently replace a committed result file after a rerun.
- Do not describe a valid citation page as proof that the page supports the claim.
- Do not use older cosine-proxy results as current faithfulness evidence.

## Validation

Run the smallest relevant set, plus all checks for shared infrastructure.

```bash
make check
make eval
make eval-scaled
.venv/bin/python evaluation/run_faithfulness_eval.py
.venv/bin/python evaluation/run_faithfulness_eval.py \
  --cases evaluation/faithfulness_cases_scaled.json --tag scaled
make frontend-build
```

Retrieval changes must run the hand-labeled and scaled cases. Generation changes must record the model, model digest if available, prompt version, generation settings, case subset, and timestamp.

## Pull-request notes

A reviewable change explains:

1. what changed and why;
2. which user or research flow is affected;
3. how it was tested;
4. whether any stored result changed;
5. limitations or follow-up work;
6. screenshots only when the interface changed materially.

If an experiment changes a conclusion, update `docs/EXPERIMENTS.md`, `RESEARCH_ROADMAP.md`, and the manuscript together. Negative results are valid outcomes and must remain visible.

## Code organization

- Keep API routing and request validation in `backend/main.py`.
- Put cohesive document, retrieval, generation, vision, or reference logic in `backend/services/`.
- Keep evaluation code independent of the live UI where practical.
- Keep generated artifacts out of source folders unless they are deliberate, traceable research results.
- Prefer typed request and response shapes and small pure functions that can be tested without Ollama or network access.

## Documentation style

Write for the next person, not for the current session. State whether work is implemented, executed, validated, or merely planned. Include exact commands and source paths for quantitative claims. Avoid venue-specific instructions until a venue has been selected.
