# 🚀 ScholAR Quick-Start Guide for Windows

Welcome to **ScholAR**! Follow these quick steps to get ScholAR running locally on your Windows laptop.

---

## 📥 Step 1: Install Prerequisites (One-Time Setup)

Download and install these 4 free tools:

1. **Git**: [Download Git for Windows](https://git-scm.com/download/win)
2. **Python (3.11 or 3.12)**: [Download Python](https://www.python.org/downloads/)  
   ⚠️ **CRITICAL**: On the very first installation screen, check the box:  
   **☑️ "Add python.exe to PATH"** before clicking Install.
3. **Node.js (LTS)**: [Download Node.js](https://nodejs.org/) (Click the button that says *LTS Recommended For Most Users*)
4. **Ollama**: [Download Ollama for Windows](https://ollama.com/download) (Run the downloaded `.exe`)

---

## 💻 Step 2: Clone the Project & Run Setup

1. Open **Command Prompt** (press `Win + R`, type `cmd`, and press Enter).
2. Run these commands to clone the project:
   ```cmd
   git clone https://github.com/prithvi-kaizen/ScholAR.git
   cd ScholAR
   git checkout feat/local-only-eval-and-paper-updates
   ```
3. In that folder, double-click **`setup_windows.bat`** (or run `setup_windows.bat` in Command Prompt).  
   *The script will automatically configure your Python environment and install all packages.*

---

## 🧠 Step 3: Download Your Local Model

Open **Command Prompt** and download your preferred model:

* **If your laptop has 16GB RAM or an NVIDIA GPU (Recommended)**:
  ```cmd
  ollama pull qwen3.5:9b
  ```
* **If your laptop is thin-and-light or has 8GB–16GB RAM (Super fast, ~1.5s per answer)**:
  ```cmd
  ollama pull llama3.2:3b
  ```

*(Tip: You only need to download the model once. It runs 100% offline on your laptop).*

---

## ▶️ Step 4: Launch ScholAR!

Whenever you want to use the application:

1. Double-click **`run_windows.bat`**.
2. ScholAR will automatically start the backend, frontend, and open your browser to:  
   👉 **http://localhost:3000**

---

## 🔍 How to Use It

1. **Upload a Paper**: In the web UI, click **Upload PDF** to add any paper (or type an arXiv ID like `1706.03762` in the search bar and click Ingest).
2. **Ask Questions**: Ask technical, tabular, or conceptual questions in the chat drawer.
3. **Inspect Evidence**: Toggle the right-hand **Evidence & Telemetry** drawer to view the exact bounding boxes, figures, and citation sources.

---

## 🛑 How to Stop ScholAR

When you're done, simply close the two Command Prompt windows that opened when you launched `run_windows.bat`.

---

## ❓ Need Help?
- **Model is slow?** Open `backend\.env` in Notepad, change `OLLAMA_MODEL=qwen3.5:9b` to `OLLAMA_MODEL=llama3.2:3b`, save, and re-run `run_windows.bat`.
- **Any other issue?** Check `WINDOWS_SETUP.md` for troubleshooting steps!
