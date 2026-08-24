# ScholAR — project task runner
# Run all commands from the project root: /path/to/ScholAR/
#
# Usage:
#   make quickstart    Interactive hardware auto-detection & 1-click installation
#   make setup         Install backend and frontend dependencies
#   make models        Configure local model based on machine hardware
#   make doctor        Check whether the local environment is ready
#   make backend       Start the backend (hot-reload)
#   make frontend      Start the Next.js frontend
#   make check         Run Python syntax and frontend type checks
#   make eval          Run the hand-labeled retrieval evaluation
#   make eval-scaled   Run the 100-case retrieval evaluation
#   make multidoc-eval Run multi-document evaluation (papers must be seeded first)
#   make seed          Seed the secondary benchmark papers (needs backend running)

.PHONY: quickstart setup models doctor backend frontend check frontend-build eval eval-scaled multidoc-eval seed help

PYTHON ?= .venv/bin/python

# ── One-Click Quickstart ──────────────────────────────────────────────────────
quickstart:
	bash scripts/quickstart.sh

# ── First-time setup ─────────────────────────────────────────────────────────
setup:
	python3 -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	cd frontend && npm ci
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; echo "Created backend/.env"; else echo "Keeping existing backend/.env"; fi
	@if [ ! -f frontend/.env.local ]; then cp frontend/.env.local.example frontend/.env.local; echo "Created frontend/.env.local"; else echo "Keeping existing frontend/.env.local"; fi
	@echo "Setup complete. Configure your model with: make models"
	@echo "Then run: make doctor"

models:
	$(PYTHON) scripts/setup_models.py

doctor:
	$(PYTHON) scripts/doctor.py

# ── Backend ──────────────────────────────────────────────────────────────────
backend:
	@echo "Starting backend on http://localhost:8000"
	@echo "NOTE: always run from the ScholAR/ root, never from inside backend/"
	$(PYTHON) -m uvicorn backend.main:app --reload --reload-dir backend

# ── Frontend ─────────────────────────────────────────────────────────────────
frontend:
	@echo "Starting frontend on http://localhost:3000"
	cd frontend && npm run dev

# ── Validation ───────────────────────────────────────────────────────────────
check:
	$(PYTHON) -m compileall -q backend evaluation
	cd frontend && npm run typecheck

frontend-build:
	cd frontend && npm run build

# ── Evaluation ───────────────────────────────────────────────────────────────
eval:
	$(PYTHON) evaluation/run_retrieval_eval.py

eval-scaled:
	$(PYTHON) evaluation/run_retrieval_eval.py \
		--cases evaluation/benchmark_cases_scaled.json --tag scaled

multidoc-eval:
	$(PYTHON) evaluation/run_multidoc_eval.py --no-ingest

# Seed secondary papers for the multi-doc eval.
# Requires the backend to be running in another terminal (make backend).
seed:
	$(PYTHON) evaluation/seed_eval_papers.py

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ScholAR make targets:"
	@echo "  make setup          Install dependencies and create local env files"
	@echo "  make doctor         Diagnose the local setup"
	@echo "  make backend        Start FastAPI backend (from project root)"
	@echo "  make frontend       Start Next.js frontend"
	@echo "  make check          Run Python syntax and frontend type checks"
	@echo "  make frontend-build Run the frontend production build"
	@echo "  make eval           Run single-doc retrieval eval (14 cases)"
	@echo "  make eval-scaled    Run scaled retrieval eval (100 cases)"
	@echo "  make seed           Seed secondary papers for multi-doc eval"
	@echo "  make multidoc-eval  Run multi-doc eval (seed first)"
	@echo ""
	@echo "IMPORTANT: Always run from ScholAR/ root, not from backend/"
	@echo "  WRONG:  cd backend && uvicorn main:app"
	@echo "  RIGHT:  make backend"
	@echo ""
