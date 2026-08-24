#!/usr/bin/env bash
# ==============================================================================
# ScholAR One-Click Quickstart Setup Script
# Automates environment creation, dependency installation, hardware-aware
# model selection, and verification across macOS, Linux, and Windows (WSL2).
# ==============================================================================

set -e

BOLD="\033[1m"
GREEN="\033[32m"
BLUE="\033[34m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

echo -e "\n${BOLD}${CYAN}======================================================${RESET}"
echo -e "${BOLD}${CYAN}              ScholAR Local Quickstart Setup          ${RESET}"
echo -e "${BOLD}${CYAN}======================================================${RESET}\n"

# 1. Check prerequisites
echo -e "${BOLD}Checking prerequisites...${RESET}"

if ! command -v python3 &> /dev/null; then
    echo -e "${YELLOW}Error: python3 is not installed. Please install Python 3.11 or 3.12.${RESET}"
    exit 1
fi

if ! command -v node &> /dev/null; then
    echo -e "${YELLOW}Error: node is not installed. Please install Node.js 18 or 20 LTS.${RESET}"
    exit 1
fi

if ! command -v npm &> /dev/null; then
    echo -e "${YELLOW}Error: npm is not installed.${RESET}"
    exit 1
fi

if ! command -v ollama &> /dev/null; then
    echo -e "${YELLOW}Warning: Ollama is not installed.${RESET}"
    echo -e "Please install Ollama from ${CYAN}https://ollama.com/download${RESET} to enable local model inference."
fi

# 2. Setup Python virtual environment
echo -e "\n${BOLD}Setting up Python virtual environment (.venv)...${RESET}"
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

# Activate virtualenv
if [ -f ".venv/bin/activate" ]; then
    source .venv/bin/activate
elif [ -f ".venv/Scripts/activate" ]; then
    source .venv/Scripts/activate
fi

echo -e "Installing Python dependencies from requirements.txt..."
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet

# 3. Setup Frontend dependencies
echo -e "\n${BOLD}Installing Frontend dependencies...${RESET}"
cd frontend
if [ -f "package-lock.json" ]; then
    npm ci --silent
else
    npm install --silent
fi
cd ..

# 4. Copy environment templates if missing
if [ ! -f "backend/.env" ]; then
    if [ -f "backend/.env.example" ]; then
        cp backend/.env.example backend/.env
    else
        echo "OLLAMA_BASE_URL=http://localhost:11434" > backend/.env
        echo "OLLAMA_MODEL=qwen3.5:9b" >> backend/.env
    fi
fi

if [ ! -f "frontend/.env.local" ]; then
    if [ -f "frontend/.env.example" ]; then
        cp frontend/.env.example frontend/.env.local
    else
        echo "NEXT_PUBLIC_BACKEND_URL=http://localhost:8000" > frontend/.env.local
    fi
fi

# 5. Run hardware-aware model configuration
python3 scripts/setup_models.py

# 6. Run environment doctor diagnostics
echo -e "\n${BOLD}Running system verification (doctor.py)...${RESET}"
python3 scripts/doctor.py || true

echo -e "\n${BOLD}${GREEN}======================================================${RESET}"
echo -e "${BOLD}${GREEN}            ScholAR Setup Complete!                   ${RESET}"
echo -e "${BOLD}${GREEN}======================================================${RESET}\n"
echo -e "To start ScholAR locally:"
echo -e "  ${BOLD}1.${RESET} In Terminal 1 (Backend): ${CYAN}make backend${RESET}   (or .venv/bin/python -m uvicorn backend.main:app --port 8000 --reload)"
echo -e "  ${BOLD}2.${RESET} In Terminal 2 (Frontend): ${CYAN}make frontend${RESET}  (or cd frontend && npm run dev)"
echo -e "  ${BOLD}3.${RESET} Open your browser at:     ${BOLD}${BLUE}http://localhost:3000${RESET}\n"
