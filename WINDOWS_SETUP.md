# 🪟 ScholAR: Windows Setup & Team Collaboration Guide

This guide enables any teammate with a **Windows 10 / 11 laptop** to set up, run, and test ScholAR locally with their own local models.

---

## 📋 1. Prerequisites (Install Once)

Before running the setup script, ensure you have these 4 tools installed on Windows:

| Prerequisite | Recommended Version | Download Link & Note |
| :--- | :--- | :--- |
| **Git for Windows** | Latest | [git-scm.com/download/win](https://git-scm.com/download/win) |
| **Python** | 3.11 or 3.12 | [python.org/downloads](https://www.python.org/downloads/) <br>⚠️ **CRITICAL**: Check the box **"Add python.exe to PATH"** on the first installer screen! |
| **Node.js** | 18 LTS or 20 LTS | [nodejs.org](https://nodejs.org/) (Choose LTS installer) |
| **Ollama for Windows** | Latest | [ollama.com/download](https://ollama.com/download) (Single-click `.exe` installer) |

---

## ⚡ 2. One-Click Automated Setup

1. Open **Command Prompt** (`cmd`) or **PowerShell** and clone the repository:
   ```cmd
   git clone https://github.com/prithvi-kaizen/ScholAR.git
   cd ScholAR
   git checkout feat/local-only-eval-and-paper-updates
   ```

2. Double-click **`setup_windows.bat`** (or run `.\setup_windows.bat` in Command Prompt).
   
   This script will automatically:
   - Verify Python, Node.js, and Ollama.
   - Create a clean virtual environment (`.venv`).
   - Install all backend dependencies via pre-compiled Windows wheels.
   - Install frontend packages (`npm install`).
   - Auto-detect your laptop's RAM and GPU to suggest the optimal model.

3. **Pick Your Local Model**:
   * **For Laptops with NVIDIA GPU or 16GB+ RAM** (Recommended):
     ```cmd
     ollama pull qwen3.5:9b
     ```
   * **For Thin & Light Laptops / Intel Iris Graphics / 8GB-16GB RAM** (Fastest, ~1.5s per query):
     ```cmd
     ollama pull llama3.2:3b
     ```

---

## 🚀 3. Starting ScholAR on Windows

Whenever you want to use ScholAR:

1. Double-click **`run_windows.bat`** (or run `.\run_windows.bat` in Command Prompt).
2. The script will automatically:
   - Check and start the Ollama service in the background.
   - Launch the FastAPI Backend on `http://localhost:8000`.
   - Launch the Next.js Frontend on `http://localhost:3000`.
   - Open your default web browser to **`http://localhost:3000`**!

To shut down ScholAR, simply close the two opened command terminal windows.

---

## 🔬 4. Testing Your Setup

1. **Upload or Search a Paper**:
   - In the web UI at `http://localhost:3000`, click **Upload PDF** to ingest any paper.
   - Or type any arXiv ID (e.g. `1706.03762`) or query in the search bar and click **Ingest**.
2. **Try a Multi-Level Reasoning Query**:
   Paste this into the chat drawer:
   > *"Table 3 reports nearly identical results when sinusoidal positional encoding is replaced by learned positional embeddings. How does this experimental result relate to the authors' reason for choosing sinusoidal positional encoding?"*
3. **Inspect Evidence & Telemetry**:
   - Toggle the right-hand **Evidence Graph & Telemetry** drawer to view:
     - Exact table and figure crops.
     - Section-level AST provenance.
     - Retrieval channels ($L_1$ to $L_5$ adaptive complexity routing).

---

## ⚙️ 5. Switching Models on Windows

To switch between models at any time, open `backend\.env` in Notepad and change the model name:

```env
# For high-fidelity reasoning:
OLLAMA_MODEL=qwen3.5:9b

# Or for ultra-fast, lightweight responses on thin laptops:
OLLAMA_MODEL=llama3.2:3b
```
Save the file and restart `run_windows.bat`.

---

## ❓ Frequently Asked Questions (FAQ)

### Q: Command Prompt says `'python' is not recognized`?
**A**: You forgot to check "Add python.exe to PATH" when installing Python. Re-run the Python installer, select **Modify**, and check **Add Python to environment variables**.

### Q: Port 8000 or 3000 is already in use?
**A**: Another app is using that port. In Command Prompt, run:
```cmd
netstat -ano | findstr :8000
taskkill /F /PID <PID_NUMBER>
```

### Q: Can I run ScholAR without an Internet connection?
**A**: **Yes!** After running `setup_windows.bat` and downloading your Ollama model once, ScholAR is **100% air-gapped and local**. No queries, embeddings, or documents ever leave your machine.
