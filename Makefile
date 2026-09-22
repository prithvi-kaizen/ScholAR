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

.PHONY: quickstart setup setup-test setup-parser setup-evaluation models visual-index visual-index-clip visual-index-colqwen corpus-plan corpus-migrate corpus-freeze corpus-check eacl-protocol-check eacl-heldout-audit eacl-heldout-ready evidence-check doctor backend frontend check test smoke ci ci-model release-artifact anonymous-artifact-check anonymous-artifact anonymous-artifact-verify submission-audit submission-ready paper-verify frontend-build reproduce-eacl eval eval-scaled multidoc-eval spiqa-eval seed help

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

setup-test:
	$(PYTHON) -m pip install -r requirements/locks/test-py312.txt

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

eacl-protocol-check:
	$(PYTHON) evaluation/protocol_governance.py

eacl-heldout-audit:
	$(PYTHON) evaluation/audit_heldout_candidate.py --output evaluation/benchmarks/two_hundred_questions_dataset.audit.json

eacl-heldout-ready:
	$(PYTHON) evaluation/protocol_governance.py --require-frozen
	$(PYTHON) evaluation/audit_heldout_candidate.py --output evaluation/benchmarks/two_hundred_questions_dataset.audit.json --require-ready

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
	$(PYTHON) evaluation/validate_result_claims.py
	$(PYTHON) evaluation/validate_phase3_paper_numbers.py
	cd frontend && npm run typecheck

evidence-check:
	$(PYTHON) evaluation/validate_result_claims.py
	$(PYTHON) evaluation/validate_phase3_paper_numbers.py
	$(PYTHON) evaluation/run_final_retrieval_ablation.py --selfcheck
	$(PYTHON) evaluation/run_final_machine_evaluation.py --selfcheck
	$(PYTHON) evaluation/run_efficiency_eval.py --selfcheck

test:
	SCHOLAR_NETWORK_MODE=strict-local HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1 $(PYTHON) -m pytest

smoke:
	SCHOLAR_NETWORK_MODE=strict-local ./run_experiments.sh smoke

ci: check test anonymous-artifact-check
	$(PYTHON) evaluation/reproduce_release_fixture.py
	$(PYTHON) evaluation/validate_human_templates.py
	$(PYTHON) evaluation/validate_paper.py --paper-dir paper/eacl_industry
	cd frontend && npm run build
	$(PYTHON) -c "from backend.main import app; assert app.title == 'ScholAR API'"

ci-model:
	test -n "$(SCHOLAR_PINNED_MODEL)"
	test -n "$(SCHOLAR_PINNED_MODEL_DIGEST)"
	test -n "$(SCHOLAR_PINNED_MODEL_QUANTIZATION)"
	SCHOLAR_NETWORK_MODE=strict-local $(PYTHON) evaluation/run_evaluation_profiles.py model-backed --execute --model "$(SCHOLAR_PINNED_MODEL)" --model-digest "$(SCHOLAR_PINNED_MODEL_DIGEST)" --quantization "$(SCHOLAR_PINNED_MODEL_QUANTIZATION)" --limit 5

release-artifact:
	SCHOLAR_NETWORK_MODE=strict-local $(PYTHON) evaluation/reproduce_release_fixture.py

anonymous-artifact-check:
	$(PYTHON) scripts/package_supplementary.py --validate-only

anonymous-artifact:
	$(PYTHON) scripts/package_supplementary.py

anonymous-artifact-verify: anonymous-artifact
	$(PYTHON) scripts/package_supplementary.py --verify-archive release/ScholAR_EACL2027_Anonymous_Artifact.zip

submission-audit:
	$(PYTHON) evaluation/final_submission_audit.py

submission-ready: anonymous-artifact-verify
	$(PYTHON) evaluation/final_submission_audit.py --require-ready

paper-verify:
	$(PYTHON) evaluation/validate_paper.py --paper-dir paper/eacl_industry

frontend-build:
	cd frontend && npm run build

reproduce-eacl:
	SCHOLAR_NETWORK_MODE=strict-local HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 $(PYTHON) evaluation/reproduce_eacl.py

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

# ── EACL 2027 Demonstration Targets ───────────────────────────────────────────
.PHONY: demo-setup demo-model demo-run demo-doctor demo-smoke demo-package demo-package-verify demo-release-check

demo-setup:
	python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "ScholAR locked setup requires CPython 3.12"'
	@if [ ! -d .venv ]; then python3 -m venv .venv; echo "Created .venv"; fi
	$(PYTHON) -m pip install -r requirements/locks/base-py312.txt
	cd frontend && npm ci
	@if [ ! -f backend/.env ]; then cp backend/.env.example backend/.env; echo "Created backend/.env"; else echo "Keeping existing backend/.env"; fi
	@if [ ! -f frontend/.env.local ]; then cp frontend/.env.local.example frontend/.env.local; echo "Created frontend/.env.local"; else echo "Keeping existing frontend/.env.local"; fi
	@echo "Demo setup complete. Run 'make demo-doctor' to verify or 'make demo-model' to acquire local LLM."

demo-model:
	$(PYTHON) scripts/setup_models.py

demo-run:
	@echo "Starting ScholAR system for interactive demonstration..."
	@echo "Backend starting on http://localhost:8000 and Frontend on http://localhost:3000"
	@echo "To run concurrently: start 'make backend' in terminal 1, and 'make frontend' in terminal 2."

demo-doctor:
	$(PYTHON) scripts/doctor.py

demo-smoke:
	SCHOLAR_NETWORK_MODE=strict-local $(PYTHON) -m pytest tests/test_demo_smoke.py

demo-package:
	$(PYTHON) scripts/package_demo_release.py

demo-package-verify:
	$(PYTHON) scripts/package_demo_release.py --verify-archive release/ScholAR_EACL2027_Demo_v1.0.0.zip

demo-release-check: demo-smoke demo-package demo-package-verify
	PYTHONPATH=. $(PYTHON) -m pytest tests/test_demo_release_package.py tests/test_demo_submission_integrity.py
	$(PYTHON) evaluation/validate_demo_paper.py
	@echo "All EACL 2027 demo release checks PASSED!"

# ── Help ─────────────────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "ScholAR make targets:"
	@echo "  make setup          Install dependencies and create local env files"
	@echo "  make setup-test     Install the pinned pytest environment"
	@echo "  make doctor         Diagnose the local setup"
	@echo "  make visual-index   Build CLIP and ColQwen2 indexes for the frozen corpus"
	@echo "  make corpus-plan    Dry-run the frozen EACL corpus migration"
	@echo "  make corpus-migrate Transactionally build and freeze that corpus"
	@echo "  make corpus-freeze  Freeze corpus identity after both visual indexes exist"
	@echo "  make corpus-check   Validate every frozen corpus artifact and checksum"
	@echo "  make eacl-protocol-check Validate the EACL protocol draft"
	@echo "  make eacl-heldout-audit Profile the candidate held-out benchmark"
	@echo "  make eacl-heldout-ready Require a frozen protocol and release-ready benchmark"
	@echo "  make anonymous-artifact-check Scan the anonymous artifact allowlist"
	@echo "  make anonymous-artifact Build the deterministic anonymous review ZIP"
	@echo "  make anonymous-artifact-verify Build and verify its manifest and checksums"
	@echo "  make submission-audit Produce the final READY/NO-GO report"
	@echo "  make submission-ready Require every paper, evidence, and artifact gate"
	@echo "  make backend        Start FastAPI backend (from project root)"
	@echo "  make frontend       Start Next.js frontend"
	@echo "  make check          Run Python syntax and frontend type checks"
	@echo "  make frontend-build Run the frontend production build"
	@echo "  make reproduce-eacl Rebuild and validate the frozen EACL artifact and paper"
	@echo "  make eval           Run single-doc retrieval eval (14 cases)"
	@echo "  make eval-scaled    Run scaled retrieval eval (100 cases)"
	@echo "  make seed           Seed secondary papers for multi-doc eval"
	@echo "  make multidoc-eval  Run multi-doc eval (seed first)"
	@echo ""
	@echo "IMPORTANT: Always run from ScholAR/ root, not from backend/"
	@echo "  WRONG:  cd backend && uvicorn main:app"
	@echo "  RIGHT:  make backend"
	@echo ""
