# ScholAR: EACL 2027 Demonstration Installation Guide

This guide describes how to install, configure, and evaluate the **ScholAR** system demonstration from the frozen release package (`ScholAR_EACL2027_Demo_v1.0.0.zip`).

---

## 1. Quickstart (Reviewer Fast-Path)

From a fresh terminal:

```bash
# 1. Unpack archive and enter directory
unzip ScholAR_EACL2027_Demo_v1.0.0.zip
cd ScholAR-EACL2027-Demo-v1.0.0

# 2. Install exact Python & Node dependencies (no silent model downloads)
make demo-setup

# 3. Verify environment prerequisites and hardware readiness
make demo-doctor

# 4. Explicitly acquire the configured local model via Ollama
make demo-model

# 5. Start the local backend (port 8000) and frontend (port 3000)
make demo-run
```

Once running:
- Open your browser to **`http://localhost:3000`** to access the interactive study workspace.
- The FastAPI backend documentation is accessible at **`http://localhost:8000/docs`**.

---

## 2. System Requirements & Tested Environments

### Reference Environment (Tested)
- **Operating System:** macOS 14 / 15 (Apple Silicon M-series: M1/M2/M3, $\ge 16$\,GB unified memory).
- **Python:** CPython 3.12 (3.12.0 – 3.12.6).
- **Node.js:** Node.js 18 or 20 LTS with npm 10+.
- **Local Inference Engine:** Ollama v0.5.0 or later (`https://ollama.com`).

### Linux Environment (Best-Effort)
- Ubuntu 22.04 LTS or newer.
- 16\,GB host RAM (CPU execution) or NVIDIA GPU with $\ge 8$\,GB VRAM (CUDA 12+).

---

## 3. Architecture & Model Acquisition Rationale

ScholAR operates under a **strict-local privacy boundary**: document ingestion, layout AST parsing, hybrid retrieval, answer generation, and deterministic verification run completely on the local host.

### Why Model Weights are Acquired Separately
Neural language model weights are large ($\sim 5.5$\,GB) and governed by separate model community licenses. To ensure the release package remains fast to download, strictly auditable, and self-contained in code, the package distributes **only source code and lockfiles**. 

Running:
```bash
make demo-model
```
executes `scripts/setup_models.py`, which detects your hardware accelerators and pulls the optimal model tier directly into your local Ollama daemon (default: `qwen2.5:7b` or `qwen3.5:9b`).

---

## 4. Model-Free Smoke Verification

To verify that the backend API, dual-engine parser, and deterministic verifier work properly *without* pulling model weights or running local LLMs:

```bash
make demo-smoke
```

This runs `pytest tests/test_demo_smoke.py`, checking:
- Ingestion and layout parsing of `examples/eacl_demo/sample_paper.pdf`.
- Coordinate bounding box extraction and normalization $[x_0, y_0, x_1, y_1]$.
- Deterministic lexical overlap and numerical consistency verifier logic.
- REST API route declarations and health checks.

---

## 5. Step-by-Step Interactive Demonstration Walkthrough

Follow these steps using the included redistributable scientific paper (`examples/eacl_demo/sample_paper.pdf`):

### Step 1: Ingest Sample Paper
1. On the home screen (`http://localhost:3000`), drag and drop `examples/eacl_demo/sample_paper.pdf` or click **Upload PDF**.
2. Watch the real-time ingestion telemetry:
   - AST layout hierarchy extraction;
   - Page rasterization and bounding-box indexing;
   - Figure and table crop generation (`sample_figure.png`, `Table 1`);
   - Atomic catalog commit.
3. The interface automatically transitions to the unified study workspace.

### Step 2: Ask the Three Verified Evaluation Questions
Refer to `examples/eacl_demo/QUESTIONS.md` for ground-truth facts:

1. **Textual Lookup Question:**
   > *"What optimizer, learning rate, and batch size are used during training?"*
   - **Verification:** ScholAR answers citing `Adam`, `beta1=0.9`, `beta2=0.999`, `2e-4`, and batch size `32` with citation marker `[1]`.
   - **Action:** Click `[1]`. The document viewer scrolls to **Page 1, Section 3.1** and renders a red semi-transparent bounding box over the paragraph.

2. **Quantitative Table Question:**
   > *"What F1 score and latency are reported for Dynamic Evidence Fusion in Table 1?"*
   - **Verification:** ScholAR reports `88.4%` F1 and `42 ms` retrieval latency citing Table 1.
   - **Action:** Click the citation chip. The viewer scrolls to **Table 1** and highlights the corresponding row.

3. **Visual Figure Question:**
   > *"According to Figure 1, which module connects the visual feature encoder to the ranker?"*
   - **Verification:** ScholAR identifies the `cross-modal projection layer`.
   - **Action:** Click the citation chip. The interface highlights **Figure 1** and displays the extracted high-resolution crop alongside the caption.

### Step 3: Inspect the Evidence Graph
1. Click the **Evidence Graph** button in the chat header.
2. The modal visualizes:
   - All candidate chunks retrieved by BM25 and dense embedding search;
   - Their fused Reciprocal Rank Fusion (RRF) scores;
   - Which chunks were selected under the RAM context budget;
   - Which chunks received citation attribution.

### Step 4: Export Audit Trace
1. Click **Export Trace** at the top right of the workspace.
2. Download the JSON trace file and inspect:
   - Exact query UUID and timestamp;
   - Raw model generation vs. verifier-repaired text;
   - Exact bounding boxes and latency breakdown.

---

## 6. Shutting Down & Troubleshooting

### Stopping the Services
Press `Ctrl+C` in the terminal running `make demo-run`.

### Common Issues
- **Port Conflict (8000 or 3000 in use):**  
  Ensure no existing uvicorn or next.js processes are active:
  ```bash
  lsof -ti:8000 | xargs kill -9
  lsof -ti:3000 | xargs kill -9
  ```
- **Ollama Connection Error:**  
  Verify Ollama is running locally:
  ```bash
  curl http://localhost:11434/api/tags
  ```
  If not running, launch `ollama serve` in a background terminal.
- **Python Version Mismatch:**  
  ScholAR dependencies are pinned against CPython 3.12. Ensure `python3 --version` outputs `Python 3.12.x`.
