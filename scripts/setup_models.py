#!/usr/bin/env python3
"""Hardware-aware model selection and configuration for ScholAR.

Detects system RAM, CPU/GPU accelerators, and recommends the best Ollama model
tier for the user's specific machine configuration.
"""

from __future__ import annotations

import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV = ROOT / "backend" / ".env"
FRONTEND_ENV = ROOT / "frontend" / ".env.local"

# ANSI Colors
BOLD = "\033[1m"
GREEN = "\033[32m"
BLUE = "\033[34m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
MAGENTA = "\033[35m"
RESET = "\033[0m"


def get_total_ram_gb() -> float:
    """Detect total system RAM in gigabytes across macOS, Linux, and Windows."""
    try:
        if sys.platform == "darwin":
            out = subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
            return round(int(out) / (1024**3), 1)
        elif sys.platform.startswith("linux"):
            with open("/proc/meminfo", "r") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        return round(kb / (1024**2), 1)
        elif sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            c_ulonglong = ctypes.c_ulonglong
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ('dwLength', ctypes.c_ulong),
                    ('dwMemoryLoad', ctypes.c_ulong),
                    ('ullTotalPhys', c_ulonglong),
                    ('ullAvailPhys', c_ulonglong),
                    ('ullTotalPageFile', c_ulonglong),
                    ('ullAvailPageFile', c_ulonglong),
                    ('ullTotalVirtual', c_ulonglong),
                    ('ullAvailVirtual', c_ulonglong),
                    ('sullAvailExtendedVirtual', c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            return round(stat.ullTotalPhys / (1024**3), 1)
    except Exception:
        pass
    return 16.0  # Fallback assumption


def detect_hardware_profile() -> tuple[str, str, float]:
    """Return (hardware_description, recommended_tier, ram_gb)."""
    ram_gb = get_total_ram_gb()
    is_apple_silicon = sys.platform == "darwin" and platform.machine() in ("arm64", "aarch64")

    has_nvidia = False
    gpu_desc = ""
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                text=True,
            ).strip()
            if out:
                has_nvidia = True
                gpu_desc = f"NVIDIA {out.splitlines()[0]}"
        except Exception:
            pass

    if is_apple_silicon:
        hw_desc = f"Apple Silicon Mac ({ram_gb} GB Unified Memory)"
    elif has_nvidia:
        hw_desc = f"{gpu_desc} + {ram_gb} GB RAM"
    else:
        hw_desc = f"{platform.system()} ({ram_gb} GB RAM, CPU-only)"

    if ram_gb < 12:
        rec_tier = "1"
    elif ram_gb < 24:
        rec_tier = "2" if not is_apple_silicon and not has_nvidia else "3"
    else:
        rec_tier = "3" if ram_gb < 48 else "4"

    return hw_desc, rec_tier, ram_gb


MODEL_TIERS = [
    {
        "tier": "1",
        "name": "Lightweight / Entry (8 GB RAM)",
        "model": "qwen2.5:7b",
        "size": "~4.7 GB",
        "multimodal": False,
        "desc": "Ultra-fast text reasoning, lowest RAM footprint. Ideal for 8GB laptops or CPU-only setups.",
    },
    {
        "tier": "2",
        "name": "Balanced Standard (16 GB RAM - Recommended)",
        "model": "qwen3.5:9b",
        "size": "~6.6 GB",
        "multimodal": True,
        "desc": "Balanced text and vision capabilities. Excellent reasoning over charts, equations, and paper prose.",
    },
    {
        "tier": "3",
        "name": "High-Precision Multimodal (16-32 GB RAM)",
        "model": "gemma4:12b",
        "size": "~8.5 GB",
        "multimodal": True,
        "desc": "State-of-the-art vision and mathematical grounding. Reads complex multi-panel figures and table grids.",
    },
    {
        "tier": "4",
        "name": "Power Workstation (32 GB+ RAM / RTX 3090/4090)",
        "model": "qwen2.5:14b",
        "size": "~9.0 GB",
        "multimodal": False,
        "desc": "Maximum textual precision and nuance across long academic papers and deep cross-document queries.",
    },
]


def check_ollama() -> bool:
    """Check if Ollama CLI is installed and running."""
    if not shutil.which("ollama"):
        print(f"\n{YELLOW}Warning:{RESET} 'ollama' CLI was not found on your PATH.")
        print("Please install Ollama from: https://ollama.com/download\n")
        return False
    return True


def get_installed_models() -> list[str]:
    """Return list of models currently downloaded in Ollama."""
    try:
        out = subprocess.check_output(["ollama", "list"], text=True)
        lines = out.strip().splitlines()[1:]
        return [line.split()[0] for line in lines if line]
    except Exception:
        return []


def update_env_file(model_name: str) -> None:
    """Write OLLAMA_MODEL to backend/.env."""
    BACKEND_ENV.parent.mkdir(parents=True, exist_ok=True)
    existing_lines = []
    if BACKEND_ENV.exists():
        existing_lines = BACKEND_ENV.read_text(encoding="utf-8").splitlines()

    updated = False
    new_lines = []
    for line in existing_lines:
        if line.startswith("OLLAMA_MODEL="):
            new_lines.append(f"OLLAMA_MODEL={model_name}")
            updated = True
        else:
            new_lines.append(line)

    if not updated:
        new_lines.append(f"OLLAMA_MODEL={model_name}")

    BACKEND_ENV.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
    print(f"Configured {GREEN}{BACKEND_ENV.relative_to(ROOT)}{RESET} with {BOLD}{model_name}{RESET}")


def main() -> None:
    print(f"\n{BOLD}{CYAN}======================================================{RESET}")
    print(f"{BOLD}{CYAN}       ScholAR Hardware & Model Configuration Tool     {RESET}")
    print(f"{BOLD}{CYAN}======================================================{RESET}\n")

    hw_desc, rec_tier, ram_gb = detect_hardware_profile()
    print(f"Detected System: {BOLD}{GREEN}{hw_desc}{RESET}")
    print(f"Available RAM:   {BOLD}{ram_gb} GB{RESET}\n")

    print(f"{BOLD}Available Model Tiers:{RESET}\n")
    for t in MODEL_TIERS:
        is_rec = t["tier"] == rec_tier
        tag = f" {GREEN}(Recommended for your machine){RESET}" if is_rec else ""
        multimodal_tag = f" {CYAN}[Vision Enabled]{RESET}" if t["multimodal"] else ""
        print(f"  {BOLD}[{t['tier']}]{RESET} {BOLD}{t['name']}{RESET}{tag}")
        print(f"      Model:       {MAGENTA}{t['model']}{RESET} ({t['size']}){multimodal_tag}")
        print(f"      Description: {t['desc']}\n")

    installed = get_installed_models() if check_ollama() else []
    if installed:
        print(f"Currently installed in Ollama: {', '.join(installed)}")

    try:
        choice = input(f"Select a tier [1-4] or press Enter for recommended [{rec_tier}]: ").strip()
    except (EOFError, KeyboardInterrupt):
        choice = rec_tier
        print(rec_tier)

    if not choice:
        choice = rec_tier

    selected_tier = next((t for t in MODEL_TIERS if t["tier"] == choice), None)
    if not selected_tier:
        print(f"{YELLOW}Invalid choice. Defaulting to tier {rec_tier}.{RESET}")
        selected_tier = next(t for t in MODEL_TIERS if t["tier"] == rec_tier)

    chosen_model = selected_tier["model"]
    print(f"\nTarget Model: {BOLD}{chosen_model}{RESET}")

    if check_ollama():
        if chosen_model not in installed and f"{chosen_model}:latest" not in installed:
            try:
                pull_prompt = input(f"Download '{chosen_model}' now via Ollama? [Y/n]: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                pull_prompt = "n"
                print("n")
            if pull_prompt in ("", "y", "yes"):
                print(f"\n{BLUE}Pulling {chosen_model} from Ollama...{RESET}")
                try:
                    subprocess.run(["ollama", "pull", chosen_model], check=True)
                    print(f"{GREEN}Model {chosen_model} downloaded successfully!{RESET}\n")
                except subprocess.CalledProcessError as err:
                    print(f"{YELLOW}Failed to pull model: {err}. You can run 'ollama pull {chosen_model}' manually.{RESET}")
        else:
            print(f"{GREEN}Model '{chosen_model}' is already available locally in Ollama.{RESET}")

    update_env_file(chosen_model)
    print(f"\n{GREEN}Hardware model setup complete!{RESET}\n")


if __name__ == "__main__":
    main()
