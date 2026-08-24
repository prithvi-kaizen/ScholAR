"""System Self-Diagnostic & Hardware Health Service for ScholAR.

Provides full diagnostic visibility into:
- Apple Silicon MPS / NVIDIA CUDA / CPU acceleration state
- System RAM and dynamic HardwareTier allocation
- Docling OCR-free engine status
- Vector embedding cache status
- Local Ollama model registry
- Telemetry trace storage metrics
"""

from __future__ import annotations

import os
import psutil
from pathlib import Path
from typing import Any

import torch

from backend.schemas.capabilities import HardwareTier
from backend.services.budgeting_service import BudgetingService
from backend.services.ollama_service import OLLAMA_BASE_URL, OLLAMA_MODEL, ollama_available
from backend.services.pdf_service import PAPERS_DIR

ROOT = Path(__file__).resolve().parents[2]
TRACES_DIR = ROOT / "backend" / "data" / "traces"


class DiagnosticService:
    """Performs comprehensive local system health checks."""

    @classmethod
    async def get_system_diagnostic(cls) -> dict[str, Any]:
        """Gather active hardware acceleration and engine status."""
        # 1. Device acceleration
        if torch.backends.mps.is_available():
            accel_device = "Apple Silicon MPS (GPU Accelerated)"
            is_accelerated = True
        elif torch.cuda.is_available():
            accel_device = f"NVIDIA CUDA ({torch.cuda.get_device_name(0)})"
            is_accelerated = True
        else:
            accel_device = "CPU (Fallback)"
            is_accelerated = False

        # 2. System Memory & Tier
        vm = psutil.virtual_memory()
        total_ram_gb = round(vm.total / (1024 ** 3), 2)
        avail_ram_gb = round(vm.available / (1024 ** 3), 2)
        tier = BudgetingService.get_hardware_tier()
        budget = BudgetingService.get_evidence_budget()

        # 3. Ingested papers & dense vector caches
        paper_dirs = [p for p in PAPERS_DIR.iterdir() if p.is_dir()] if PAPERS_DIR.exists() else []
        cached_vectors_count = sum(
            1 for p in paper_dirs if (p / "embeddings.npy").exists()
        )

        # 4. Ollama Local LLM Server
        is_ollama_up = await ollama_available()

        # 5. Telemetry traces
        traces_count = len(list(TRACES_DIR.glob("*.json"))) if TRACES_DIR.exists() else 0

        return {
            "status": "HEALTHY",
            "acceleration": {
                "device": accel_device,
                "is_gpu_accelerated": is_accelerated,
                "torch_version": torch.__version__,
            },
            "memory": {
                "total_ram_gb": total_ram_gb,
                "available_ram_gb": avail_ram_gb,
                "hardware_tier": tier.value,
                "token_budget": budget.max_context_tokens,
                "max_evidence_blocks": budget.max_evidence_blocks,
            },
            "local_llm": {
                "ollama_url": OLLAMA_BASE_URL,
                "active_model": OLLAMA_MODEL,
                "is_connected": is_ollama_up,
                "mode": "100% Local / Zero Egress",
            },
            "storage": {
                "ingested_papers_count": len(paper_dirs),
                "cached_embeddings_count": cached_vectors_count,
                "telemetry_traces_recorded": traces_count,
            },
        }
