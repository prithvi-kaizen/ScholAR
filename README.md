# ScholAR: Inspectable Local-First Scientific Paper Assistant

[![EACL 2027 Demo](https://img.shields.io/badge/EACL_2027-System_Demonstration-blue.svg)](paper/eacl_demo/main.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](requirements/locks/base-py312.txt)
[![Release](https://img.shields.io/badge/Release-v1.0.0--eacl--demo-orange.svg)](https://github.com/prithvi-kaizen/ScholAR/releases/tag/eacl-demo-2027-v1.0.0)
[![Privacy First](https://img.shields.io/badge/Privacy-100%25_Local_Offline-success.svg)](#runtime-boundary)

**ScholAR** is an inspectable, local-first scientific-document assistant designed for researchers, students, and peer reviewers. It ingests multi-column scientific PDFs into source-scoped text, table, figure, and layout-aware evidence; retrieves across those modalities using hybrid lexical and dense ranking; answers queries using local Small Language Models (SLMs) via Ollama; and provides **verifiable, bounding-box spatial grounding** where clicking any citation immediately jumps to and highlights the supporting evidence in the original PDF.

The entire system runs on consumer hardware (tested on Apple Silicon Macs, Linux workstations, and Windows) without sending sensitive, unpublished manuscripts to external cloud APIs.

---

![ScholAR system overview](docs/diagrams/system-overview.svg)

---

## EACL 2027 System Demonstration

ScholAR is presented at the **EACL 2027 System Demonstrations Track**:
> **ScholAR: Inspectable Local-First Question Answering for Scientific Papers**  
> *Prithviraj Patil*  
> 📄 **Paper:** [`paper/eacl_demo/main.pdf`](paper/eacl_demo/main.pdf)  
> 📦 **Release Package:** [`release/ScholAR_EACL2027_Demo_v1.0.0.zip`](release/ScholAR_EACL2027_Demo_v1.0.0.zip) (SHA-256: `ebb3a67286c35256ccb7ca78df66371953c4f4f8bc7b791a7b1ce4f83b0e825f`)  
> 📖 **Reviewer & Installation Guide:** [`DEMO_INSTALL.md`](DEMO_INSTALL.md)  
> 🎥 **Demonstration Screencast:** `ScholAR_Demo_Screencast.mp4` (available on GitHub Releases)

### Core Demonstration Features

1. **100% Local Inference & Privacy**: Executes entirely on local hardware using Ollama (`qwen3.5:9b`, `gemma4:12b`, or `qwen2.5:7b`). Manuscripts, notes, and queries remain on your machine.
2. **Multi-Modal Evidence AST**: Preserves page identity, multi-column geometry, and bounding boxes for text paragraphs, tables, and figures during ingestion.
3. **Hybrid Retrieval (BM25 + Dense RRF)**: Reciprocal Rank Fusion combines sparse token matching with dense transformer embeddings for robust evidence selection (Hit@1 of 0.85 across 100 questions / 25 papers).
4. **Interactive Spatial Grounding**: Every generated statement is paired with an inspectable citation badge (`[Verified]`, `[Uncertain]`). Clicking a badge instantly synchronizes the PDF viewer to that exact page and draws a highlight box around the supporting locus.
5. **Automated Study Goals**: Generates structured, pedagogical reading goals tailored to the paper's specific methods, datasets, and claims.
6. **Extracted References**: Automatically extracts and links the bibliography for one-click cross-document exploration and multi-document comparative analysis.
7. **Transparent Audit Trace**: Inspectable execution trace exposing raw chunk relevance scores, token latencies, verifier decisions, and export options (Markdown / LaTeX).

---

## Quick Start (Cross-Platform)

ScholAR includes a zero-dependency Python runner (`run_demo.py`) that operates identically on **macOS**, **Linux**, and **Windows** without requiring `make` or bash:

```bash
# 1. Automated setup: creates virtualenv, installs locked Python packages & frontend npm dependencies
python run_demo.py setup

# 2. Environment check: verifies Python version, dependencies, and hardware profile
python run_demo.py doctor

# 3. Fast smoke test: hermetic, model-free verification of parser, API, and verifier (~1 second)
python run_demo.py smoke

# 4. (Optional) Detect hardware accelerators and pull local Ollama model (e.g. qwen3.5:9b)
python run_demo.py model
```

### Running the System

Start the backend and frontend in two separate terminals:

```bash
# Terminal 1: FastAPI backend on http://localhost:8000 (OpenAPI docs at /docs)
python run_demo.py backend

# Terminal 2: Next.js interactive web interface on http://localhost:3000
python run_demo.py frontend
```

Open your browser to **`http://localhost:3000`** to access the application.

*(For detailed platform-specific steps and troubleshooting for macOS, Ubuntu/Debian Linux, and Windows PowerShell, see [`DEMO_INSTALL.md`](DEMO_INSTALL.md)).*

---

## Documentation & Guides

- [Codebase Guide](docs/CODEBASE.md): Directory structure, architecture layers, storage schema, and API contracts.
- [Pipeline Guide](docs/PIPELINE.md): Ingestion, chunking, embedding, hybrid retrieval, generation, citation, and verification flow.
- [Demo Installation Guide](DEMO_INSTALL.md): Complete multi-platform setup and evaluation guide for EACL 2027 reviewers.
- [Setup & Hardware Guide](docs/SETUP.md): In-depth hardware profiling, memory tiers, and local model selection.
- [Evaluation Guide](evaluation/README.md): Benchmark runners, datasets, public aggregates, and auditing protocols.

---

## Runtime Boundary & Data Governance

ScholAR enforces an explicit network boundary:
- **`strict-local` (Default):** Pre-ingested local papers, cached embeddings, and loopback Ollama calls are permitted. External arXiv search, reference scraping, and implicit network downloads are blocked.
- **`acquisition-enabled`:** Temporarily enabled only when explicitly searching arXiv or fetching new paper PDFs.

Local runtime data lives under `backend/data/` and is strictly excluded from Git.

---

## Verification & Reproducibility Suite

You can verify system correctness, package integrity, and manuscript numbers locally:

```bash
# 1. Model-free demo smoke test (fast verification)
python run_demo.py smoke

# 2. Comprehensive EACL 2027 demo release check (runs smoke, packaging, integrity, and paper checks)
make demo-release-check

# 3. Core codebase sanity checks
make check          # Python syntax + frontend typecheck
make test           # Strict-local backend unit & integration tests
make frontend-build # Production Next.js bundle verification
```

### Benchmark Summary

| Evaluation Aspect | Scope & Cohort | Key Finding | Ground Truth Source |
| :--- | :--- | :--- | :--- |
| **Page Retrieval Ablation** | 100 questions / 25 papers | BM25+Dense RRF reaches **Hit@1 = 0.85**, **MRR = 0.905** | [`evaluation/PHASE2_AGGREGATE_PUBLIC.json`](evaluation/PHASE2_AGGREGATE_PUBLIC.json) |
| **Deterministic Repair** | 30 paired answerable cases | Raises reference-unit F1 (**0.159 $\to$ 0.224**) | [`evaluation/PHASE2_RESULTS.md`](evaluation/PHASE2_RESULTS.md) |
| **Diagnostic LLM Audit** | 150 completed cases | Grounding supported in **74.7%**; Citation accurate in **73.3%** | [`evaluation/EACL_DEMO_150_LUNA_PUBLIC.json`](evaluation/EACL_DEMO_150_LUNA_PUBLIC.json) |

---

## Repository Structure

```text
├── backend/          # FastAPI server, Evidence AST schemas, hybrid retrieval & pipeline services
├── frontend/         # Next.js 15 web application (split-view reader, chat copilot, telemetry)
├── paper/
│   ├── eacl_demo/    # EACL 2027 System Demonstrations Track manuscript (LaTeX source & artifacts)
│   └── eacl_industry/# EACL 2027 Industry Track manuscript source
├── evaluation/       # Retrospective benchmarks, audit ledgers, evaluation protocols & aggregates
├── release/          # Frozen reproducible demo packages (ZIP archives & SHA-256 sidecars)
├── docs/             # Technical architecture, pipeline documentation, and setup guides
├── scripts/          # Turnkey CLI runners, hardware diagnostics, and packaging tools
├── requirements/     # Layered Python dependency specifications and frozen 3.12 locks
├── tests/            # Test suite: unit, integration, smoke, release package & submission integrity
├── DEMO_INSTALL.md   # Official EACL 2027 Demonstration installation and review guide
└── run_demo.py       # Cross-platform CLI runner for macOS, Linux, and Windows
```

---

## Citation

If you use ScholAR in your research or wish to refer to the demonstration:

```bibtex
@inproceedings{patil2027scholar,
  title     = {ScholAR: Inspectable Local-First Question Answering for Scientific Papers},
  author    = {Patil, Prithviraj},
  booktitle = {Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations (EACL 2027)},
  year      = {2027},
  publisher = {Association for Computational Linguistics}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).
