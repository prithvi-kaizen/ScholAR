# ScholAR

ScholAR is a local-first scientific-paper assistant. It ingests PDFs into
source-scoped text, table, figure, and full-page visual evidence; retrieves across
those modalities; answers with a local Ollama model; and returns an auditable trace
with page-linked citations and claim verification.

The application is a research prototype. Evaluation output is generated locally and
is intentionally not committed as a product claim.

![ScholAR system overview](docs/diagrams/system-overview.svg)

## Start here

- [Codebase guide](docs/CODEBASE.md): directory layout, ownership, entry points,
  storage, APIs, and development rules.
- [Pipeline guide](docs/PIPELINE.md): exact ingestion, chunking, embedding,
  retrieval, visual understanding, generation, citation, and verification path.
- [Setup guide](docs/SETUP.md): install, configure, run, and validate locally.
- [Evaluation guide](evaluation/README.md): current runners, datasets, result
  lifecycle, and release tooling.

## Quick start

Requirements: Python 3.12, Node.js 20/npm 10, and optionally Ollama for generated
answers.

```bash
bash scripts/quickstart.sh
make backend
# in a second terminal
make frontend
```

The web app runs at `http://localhost:3000`; the FastAPI OpenAPI UI is at
`http://localhost:8000/docs`.

The setup installs the exact base dependency lock but does not silently download
models, papers, datasets, or Docling assets. Use `make setup-parser`,
`make setup-evaluation`, and `make models` only when those layers are needed.

## Runtime boundary

`SCHOLAR_NETWORK_MODE=strict-local` is the default. In this mode, prepared local
papers, cached encoders, and loopback Ollama calls are allowed; arXiv search/download,
reference resolution/download, and implicit model acquisition are rejected.
Temporarily use `acquisition-enabled` only for an explicit acquisition operation.

Local runtime data lives under `backend/data/` and is excluded from Git. Do not commit
PDFs, extracted paper text, encoder indexes, traces, environment files, or evaluator
exports.

## Verification

```bash
make check          # Python compilation + frontend typecheck
make test           # strict-local backend tests
make smoke          # evaluation selfchecks + tests
make frontend-build # production frontend build
make paper-verify   # manuscript provenance gates
```

`./run_experiments.sh` is the safe offline smoke profile. Other profiles print a plan
unless `--execute` is supplied; see [evaluation/README.md](evaluation/README.md).

## Main directories

```text
backend/       FastAPI API, schemas, pipeline services, local runtime data
frontend/      Next.js application and typed API client
evaluation/    benchmark inputs, runners, scoring, human-study and release tooling
paper/         canonical EACL Industry Track manuscript source
requirements/ layered direct specifications and Python 3.12 locks
scripts/       setup, diagnostics, ingestion, and artifact migration utilities
tests/         backend, pipeline, governance, and release regression tests
docs/          the maintained codebase, pipeline, and setup guides
```

## License

[MIT](LICENSE)
