#!/usr/bin/env bash
# ScholAR: 1-Click Master Reproduction Script for EACL 2027 Industry Track
# Reproduces all paper benchmark tables, ablations, adversarial stress tests, and latency profiling.

set -e

echo "=========================================================================="
echo "  ScholAR v1: EACL 2027 Master Experimental Suite & Reproducibility Runner"
echo "=========================================================================="

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN=".venv/bin/python"
if [ ! -f "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "[1/8] Running Question Classifier Independent Evaluation..."
$PYTHON_BIN evaluation/eval_classifier.py

echo "[2/8] Running Full Baseline Matrix (B0 - B9)..."
$PYTHON_BIN evaluation/eval_baselines.py

echo "[3/8] Running Component & Representation Ablations..."
$PYTHON_BIN evaluation/eval_ablations.py

echo "[4/8] Running Adversarial Grounding & Perturbation Stress Tests..."
$PYTHON_BIN evaluation/eval_adversarial.py

echo "[5/8] Running End-to-End Deployed System Latency & Memory Profiler..."
$PYTHON_BIN evaluation/profile_system.py

echo "[6/8] Running Parser Robustness & Layout Hierarchy Evaluation..."
$PYTHON_BIN evaluation/eval_parser_robustness.py

echo "[7/8] Running User Study & Verification Efficiency Simulation..."
$PYTHON_BIN evaluation/eval_user_study.py

echo "[8/8] Executing Full Unit Test Suite (75 Tests)..."
$PYTHON_BIN -m unittest discover -s tests

echo "=========================================================================="
echo "  [SUCCESS] All EACL 2027 Experiments & Tests Completed Successfully!"
echo "  Artifacts saved to evaluation/*.json and manuscript/eacl2027_scholar.tex"
echo "=========================================================================="
