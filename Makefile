# ScholAR — project task runner
# Run all commands from the project root: /path/to/ScholAR/
#
# Usage:
#   make backend       — start the backend (hot-reload)
#   make frontend      — start the Next.js frontend
#   make eval          — run single-document retrieval eval
#   make multidoc-eval — run multi-document eval (papers must be seeded first)
#   make seed          — seed the 10 secondary benchmark papers (needs backend running)
#   make dev           — start backend + frontend together (requires tmux or run manually)

.PHONY: backend frontend eval multidoc-eval seed help

# ── Backend ──────────────────────────────────────────────────────────────────
backend:
	@echo "Starting backend on http://localhost:8000"
	@echo "NOTE: always run from the ScholAR/ root, never from inside backend/"
	uvicorn backend.main:app --reload --reload-dir backend

# ── Frontend ─────────────────────────────────────────────────────────────────
frontend:
	@echo "Starting frontend on http://localhost:3000"
	cd frontend && npm run dev

# ── Evaluation ───────────────────────────────────────────────────────────────
eval:
	python3 evaluation/run_retrieval_eval.py

multidoc-eval:
	python3 evaluation/run_multidoc_eval.py --no-ingest

# Seed secondary papers for the multi-doc eval.
# Requires the backend to be running in another terminal (make backend).
seed:
	python3 evaluation/seed_eval_papers.py

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ScholAR make targets:"
	@echo "  make backend        Start FastAPI backend (from project root)"
	@echo "  make frontend       Start Next.js frontend"
	@echo "  make eval           Run single-doc retrieval eval (14 cases)"
	@echo "  make seed           Seed secondary papers for multi-doc eval"
	@echo "  make multidoc-eval  Run multi-doc eval (seed first)"
	@echo ""
	@echo "IMPORTANT: Always run from ScholAR/ root, not from backend/"
	@echo "  WRONG:  cd backend && uvicorn main:app"
	@echo "  RIGHT:  uvicorn backend.main:app --reload"
	@echo ""
