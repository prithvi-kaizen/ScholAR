# ScholAR: EACL 2027 Demonstration Installation & Evaluation Guide

This guide provides step-by-step instructions to install, verify, and evaluate the **ScholAR** system demonstration from the frozen release package (`ScholAR_EACL2027_Demo_v1.0.0.zip`) across **macOS**, **Linux**, and **Windows**.

---

## 1. Quickstart (Cross-Platform Universal Fast-Path)

ScholAR includes a cross-platform demo runner (`run_demo.py`) that operates identically on macOS, Linux, and Windows using standard Python—no `make` or bash shell required:

```bash
# 1. Unpack the release archive and enter the root directory
unzip ScholAR_EACL2027_Demo_v1.0.0.zip
cd ScholAR-EACL2027-Demo-v1.0.0

# 2. Automated setup: creates .venv, installs locked packages, and installs frontend npm packages
python run_demo.py setup

# 3. Environment check: verifies Python version, dependencies, and hardware profile
python run_demo.py doctor

# 4. Instant smoke test: hermetic, model-free verification of parser, API, and verifier (1 second)
python run_demo.py smoke

# 5. (Optional) Detect hardware accelerators and pull local Ollama model (e.g. qwen3.5:9b)
python run_demo.py model
```

### Running the Live Interactive Demo
In two separate terminals:
- **Terminal 1 (Backend):**
  ```bash
  python run_demo.py backend
  ```
  *(Starts FastAPI backend on `http://localhost:8000` with Swagger docs at `http://localhost:8000/docs`)*
- **Terminal 2 (Frontend):**
  ```bash
  python run_demo.py frontend
  ```
  *(Starts Next.js frontend on `http://localhost:3000`)*

Open your browser to **`http://localhost:3000`** to access the interactive study workspace.

---

## 2. Dedicated Platform Guides

### 2.1 macOS (Apple Silicon M-Series & Intel)

#### System Requirements
- macOS 14 (Sonoma) or macOS 15 (Sequoia).
- Apple Silicon M-series (M1/M2/M3/M4) with $\ge 16$\,GB unified memory recommended.
- Python 3.11+ (Python 3.12 recommended).
- Node.js 18 or 20 LTS with npm (`brew install node`).
- [Ollama](https://ollama.com) installed for local LLM inference.

#### Execution via Makefile
```bash
unzip ScholAR_EACL2027_Demo_v1.0.0.zip
cd ScholAR-EACL2027-Demo-v1.0.0

make demo-setup     # Sets up virtualenv, installs locked requirements, runs npm ci
make demo-doctor    # Diagnoses environment prerequisites
make demo-smoke     # Runs model-free smoke test under strict-local mode
make demo-model     # Pulls local model via Ollama
make demo-run       # Displays launch commands
```
Then start the services:
- Terminal 1: `make backend`
- Terminal 2: `make frontend`

---

### 2.2 Linux (Ubuntu / Debian / Fedora / Arch)

#### System Requirements
- Ubuntu 22.04 LTS or newer.
- 16\,GB host RAM (CPU execution) or an NVIDIA GPU with $\ge 8$\,GB VRAM (CUDA 12+).
- Python 3.11+ and Node.js 18+ LTS.

#### Prerequisites Installation (Ubuntu / Debian)
```bash
sudo apt update && sudo apt install -y python3-venv python3-pip curl git

# If Node.js is not yet installed:
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
sudo apt install -y nodejs
```

#### Step-by-Step Setup
```bash
unzip ScholAR_EACL2027_Demo_v1.0.0.zip
cd ScholAR-EACL2027-Demo-v1.0.0

# Setup environment and dependencies
python3 run_demo.py setup

# Verify readiness
python3 run_demo.py doctor

# Run fast model-free smoke test
python3 run_demo.py smoke

# Start servers:
# Terminal 1:
python3 run_demo.py backend

# Terminal 2:
python3 run_demo.py frontend
```

---

### 2.3 Windows 10 / 11 (PowerShell & Command Prompt)

#### System Requirements
- Windows 10 (64-bit) or Windows 11.
- 16\,GB system RAM or an NVIDIA GPU with $\ge 8$\,GB VRAM.
- [Python 3.11 or 3.12](https://www.python.org/downloads/) (⚠️ **Important:** Check *"Add python.exe to PATH"* during installation).
- [Node.js 18 or 20 LTS](https://nodejs.org/) (64-bit installer).
- [Ollama for Windows](https://ollama.com/download/windows) (optional for local LLM answers).

#### PowerShell Turnkey Execution
Open **PowerShell** and run:

```powershell
# 1. Unpack release package
Expand-Archive ScholAR_EACL2027_Demo_v1.0.0.zip -DestinationPath .
cd ScholAR-EACL2027-Demo-v1.0.0

# 2. Automated setup (creates .venv, installs wheels, copies configs, runs npm install)
python run_demo.py setup

# 3. Verify environment
python run_demo.py doctor

# 4. Run model-free smoke verification (completes in ~1 second)
python run_demo.py smoke

# 5. Launch application
# In PowerShell Window 1:
python run_demo.py backend

# In PowerShell Window 2:
python run_demo.py frontend
```

#### Manual PowerShell Steps (Alternative)
If you prefer standard manual PowerShell commands:
```powershell
# Create & activate virtualenv
python -m venv .venv
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
.\.venv\Scripts\Activate.ps1

# Install locked dependencies
python -m pip install -r requirements\locks\base-py312.txt -r requirements\locks\test-py312.txt

# Create environment config files
Copy-Item backend\.env.example backend\.env
Copy-Item frontend\.env.local.example frontend\.env.local

# Install frontend dependencies
cd frontend
npm install
cd ..

# Run smoke test
$env:SCHOLAR_NETWORK_MODE="strict-local"
pytest tests\test_demo_smoke.py

# Launch backend
python -m uvicorn backend.main:app --port 8000 --reload
```

---

## 3. Architecture & Model Acquisition Rationale

ScholAR operates under a **strict-local privacy boundary**: document ingestion, layout AST parsing, hybrid retrieval, answer generation, and deterministic verification run completely on the local host.

### Why Model Weights are Acquired Separately
Neural language model weights are large ($\sim 5.5$\,GB) and governed by separate model community licenses. To ensure the release package remains fast to download (<1 MB), strictly auditable, and self-contained in code, the package distributes **source code, assets, and lockfiles**. 

Running:
```bash
python run_demo.py model   # or: make demo-model
```
executes `scripts/setup_models.py`, which detects your hardware accelerators and pulls the optimal model tier directly into your local Ollama daemon (default: `qwen2.5:7b` or `qwen3.5:9b`).

---

## 4. Model-Free Smoke Verification

To verify that the backend API, dual-engine parser, and deterministic verifier work properly *without* pulling model weights or running local LLMs:

```bash
python run_demo.py smoke   # or: make demo-smoke
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

## 6. Troubleshooting by Operating System

### Stopping the Services
Press `Ctrl+C` in each terminal running the backend or frontend.

### Port Conflicts (8000 or 3000 in use)
- **macOS / Linux:**
  ```bash
  lsof -ti:8000 | xargs kill -9
  lsof -ti:3000 | xargs kill -9
  ```
- **Windows (PowerShell):**
  ```powershell
  Get-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess | Stop-Process -Force
  Get-Process -Id (Get-NetTCPConnection -LocalPort 3000).OwningProcess | Stop-Process -Force
  ```

### Windows PowerShell Script Execution Policy
If PowerShell errors with `running scripts is disabled on this system`:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process
```

### Ollama Connection
Verify Ollama is running locally:
- **macOS / Linux / Windows:**
  ```bash
  curl http://localhost:11434/api/tags
  ```
  If not running, launch Ollama from your application menu or run `ollama serve`.
