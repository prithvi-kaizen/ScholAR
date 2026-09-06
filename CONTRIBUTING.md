# Contributing to ScholAR

Read [docs/CODEBASE.md](docs/CODEBASE.md) and [docs/PIPELINE.md](docs/PIPELINE.md)
before changing the runtime. They document the current call path and the invariants
that tests enforce.

## Development workflow

1. Use Python 3.12 and the locked dependencies from `requirements/locks/`.
2. Keep `SCHOLAR_NETWORK_MODE=strict-local` while developing analysis features.
3. Put orchestration in `AnswerPipelineService`; the API and evaluation runner must not
   grow separate answer paths.
4. Preserve `source_paper_id`, page, chunk/evidence identity, and image-relative paths
   through every retrieval and citation transformation.
5. Make model or encoder degradation explicit in traces. A measured condition must fail
   closed when its required local asset is absent.
6. Add focused tests for the behavior and run the checks below.

```bash
make check
make test
make smoke
make frontend-build
```

## Repository hygiene

- Do not commit `backend/data/papers/`, `backend/data/traces/`, `.env` files, model
  caches, `frontend/.next/`, evaluator exports, or generated evaluation results.
- Do not add a second root-level architecture guide. Update `docs/CODEBASE.md`,
  `docs/PIPELINE.md`, and their SVGs together with code changes.
- Keep experimental code under `evaluation/`; production services belong under
  `backend/services/` only after the application uses them.
- Never copy evaluation numbers into the manuscript by hand. The release pipeline must
  produce tables from validated aggregates.

## Pull requests

Explain the user-visible change, affected pipeline stage, degradation behavior, and
tests run. If storage or trace schemas change, include a migration and backward-compatibility
note.
