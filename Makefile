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

.PHONY: quickstart setup setup-parser setup-evaluation models visual-index visual-index-clip visual-index-colqwen corpus-plan corpus-migrate corpus-freeze corpus-check doctor backend frontend check test smoke ci release-artifact paper-verify frontend-build eval eval-scaled multidoc-eval spiqa-eval seed help

PYTHON ?= .venv/bin/python

# ── One-Click Quickstart ──────────────────────────────────────────────────────
quickstart:
	bash scripts/quickstart.sh

# ── First-time setup ─────────────────────────────────────────────────────────
setup:
	python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "ScholAR locked setup requires CPython 3.12"'
	python3 -m venv .venv
	$(PYTHON) -m pip install -r requirements/locks/base-py312.txt
	cd frontend && npm ci
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; echo "Created backend/.env"; else echo "Keeping existing backend/.env"; fi
	@if [ ! -f frontend/.env.local ]; then cp frontend/.env.local.example frontend/.env.local; echo "Created frontend/.env.local"; else echo "Keeping existing frontend/.env.local"; fi
	@echo "Base setup complete. Model/parser packages and assets are separate acquisition steps."
	@echo "Optional packages: make setup-parser or make setup-evaluation"
	@echo "Model assets: make models (acquisition-enabled only)"

setup-parser:
	$(PYTHON) -m pip install -r requirements/locks/parser-py312.txt

setup-evaluation:
	$(PYTHON) -m pip install -r requirements/locks/evaluation-py312.txt

models:
	$(PYTHON) scripts/setup_models.py

visual-index: visual-index-clip visual-index-colqwen

visual-index-clip:
	$(PYTHON) scripts/prebuild_visual_indexes.py --backend clip --selection evaluation/corpus/eacl_industry_v1_selection.json

visual-index-colqwen:
	$(PYTHON) scripts/prebuild_visual_indexes.py --backend colqwen2 --selection evaluation/corpus/eacl_industry_v1_selection.json

corpus-plan:
	$(PYTHON) scripts/migrate_visual_artifacts.py --selection evaluation/corpus/eacl_industry_v1_selection.json

corpus-migrate:
	SCHOLAR_NETWORK_MODE=strict-local $(PYTHON) scripts/migrate_visual_artifacts.py --selection evaluation/corpus/eacl_industry_v1_selection.json --apply --manifest-out evaluation/corpus/eacl_industry_v1_manifest.json --data-card-out evaluation/corpus/eacl_industry_v1_data_card.json

corpus-freeze:
	$(PYTHON) evaluation/corpus/build_manifest.py --selection evaluation/corpus/eacl_industry_v1_selection.json --output evaluation/corpus/eacl_industry_v1_manifest.json --data-card evaluation/corpus/eacl_industry_v1_data_card.json --require-index-manifest visual_page_embeddings_manifest.json --require-index-manifest colqwen_page_manifest.json

corpus-check:
	$(PYTHON) evaluation/corpus/build_manifest.py --selection evaluation/corpus/eacl_industry_v1_selection.json --output evaluation/corpus/eacl_industry_v1_manifest.json --data-card evaluation/corpus/eacl_industry_v1_data_card.json --check

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

test:
	SCHOLAR_NETWORK_MODE=strict-local HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m unittest discover -s tests

smoke:
	SCHOLAR_NETWORK_MODE=strict-local ./run_experiments.sh smoke

ci: check test
	$(PYTHON) evaluation/reproduce_release_fixture.py
	$(PYTHON) evaluation/validate_human_templates.py
	$(PYTHON) evaluation/validate_paper.py --paper-dir paper/eacl_industry
	cd frontend && npm run build
	$(PYTHON) -c "from backend.main import app; assert app.title == 'ScholAR API'"

release-artifact:
	SCHOLAR_NETWORK_MODE=strict-local $(PYTHON) evaluation/reproduce_release_fixture.py

paper-verify:
	$(PYTHON) evaluation/validate_paper.py --paper-dir paper/eacl_industry

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

spiqa-eval:
	$(PYTHON) evaluation/spiqa/run_spiqa_eval.py --tier retrieval

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
	@echo "  make visual-index   Build CLIP and ColQwen2 indexes for the frozen corpus"
	@echo "  make corpus-plan    Dry-run the frozen EACL corpus migration"
	@echo "  make corpus-migrate Transactionally build and freeze that corpus"
	@echo "  make corpus-freeze  Freeze corpus identity after both visual indexes exist"
	@echo "  make corpus-check   Validate every frozen corpus artifact and checksum"
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
