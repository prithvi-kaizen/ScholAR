#!/usr/bin/env python3
"""ScholAR Cross-Platform Demo Runner (macOS, Linux, Windows).

This utility provides turnkey setup, diagnostics, smoke verification,
and execution commands that work identically across macOS, Linux, and Windows
without requiring 'make' or bash.

Usage:
    python run_demo.py setup      # Create venv, install Python & npm dependencies
    python run_demo.py doctor     # Verify environment prerequisites and hardware
    python run_demo.py smoke      # Run model-free smoke test (offline, 1 second)
    python run_demo.py model      # Detect hardware & pull local Ollama model
    python run_demo.py backend    # Start the backend server on port 8000
    python run_demo.py frontend   # Start the frontend dev server on port 3000
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV_DIR = ROOT / ".venv"


def get_venv_python() -> Path:
    """Return the platform-specific path to the virtualenv python binary."""
    if sys.platform == "win32":
        candidate = VENV_DIR / "Scripts" / "python.exe"
    else:
        candidate = VENV_DIR / "bin" / "python"
    return candidate if candidate.is_file() else Path(sys.executable)


def get_npm_executable() -> str:
    """Find npm executable on Unix or Windows."""
    npm = shutil.which("npm.cmd") if sys.platform == "win32" else shutil.which("npm")
    if not npm:
        npm = "npm"
    return npm


def run_command(cmd: list[str], cwd: Path | None = None, env: dict[str, str] | None = None) -> int:
    """Run command with real-time streaming output."""
    work_dir = cwd or ROOT
    environment = os.environ.copy()
    if env:
        environment.update(env)
    print(f"\n[ScholAR] > {' '.join(cmd)} (cwd: {work_dir})")
    try:
        proc = subprocess.run(cmd, cwd=str(work_dir), env=environment, check=False)
        return proc.returncode
    except FileNotFoundError as exc:
        print(f"[ScholAR ERROR] Executable not found: {exc}")
        return 1
    except KeyboardInterrupt:
        print("\n[ScholAR] Interrupted by user.")
        return 130


def cmd_setup() -> int:
    """Set up Python venv, locked dependencies, environment files, and frontend."""
    print("=" * 60)
    print("ScholAR Demo Setup (Cross-Platform)")
    print("=" * 60)

    # 1. Check Python version
    v = sys.version_info
    print(f"Detected host Python: {v.major}.{v.minor}.{v.micro} ({sys.executable})")
    if v.major != 3 or v.minor < 11:
        print("[ERROR] ScholAR requires Python 3.11 or newer.")
        return 1

    # 2. Create virtual environment
    if not VENV_DIR.exists():
        print(f"\nCreating virtual environment in {VENV_DIR}...")
        code = run_command([sys.executable, "-m", "venv", str(VENV_DIR)])
        if code != 0:
            print("[ERROR] Failed to create virtual environment.")
            return code
    else:
        print(f"\nVirtual environment already exists at {VENV_DIR}")

    venv_py = get_venv_python()
    print(f"Using virtualenv python: {venv_py}")

    # 3. Upgrade pip and install requirements
    print("\nInstalling pinned Python dependencies...")
    run_command([str(venv_py), "-m", "pip", "install", "--upgrade", "pip"])

    req_base = ROOT / "requirements" / "locks" / "base-py312.txt"
    req_test = ROOT / "requirements" / "locks" / "test-py312.txt"
    req_compat = ROOT / "requirements.txt"

    if req_base.is_file() and req_test.is_file():
        code = run_command([str(venv_py), "-m", "pip", "install", "-r", str(req_base), "-r", str(req_test)])
    else:
        code = run_command([str(venv_py), "-m", "pip", "install", "-r", str(req_compat)])

    if code != 0:
        print("[ERROR] Python package installation failed.")
        return code

    # 4. Copy environment files if missing
    backend_env = ROOT / "backend" / ".env"
    backend_example = ROOT / "backend" / ".env.example"
    if not backend_env.exists() and backend_example.exists():
        shutil.copyfile(backend_example, backend_env)
        print("Created backend/.env from template.")

    frontend_env = ROOT / "frontend" / ".env.local"
    frontend_example = ROOT / "frontend" / ".env.local.example"
    if not frontend_env.exists() and frontend_example.exists():
        shutil.copyfile(frontend_example, frontend_env)
        print("Created frontend/.env.local from template.")

    # 5. Frontend dependencies
    frontend_dir = ROOT / "frontend"
    npm_exe = get_npm_executable()
    if frontend_dir.is_dir() and shutil.which(npm_exe):
        print("\nInstalling frontend npm dependencies...")
        code = run_command([npm_exe, "install"], cwd=frontend_dir)
        if code != 0:
            print("[WARN] npm install returned non-zero code. You may need to run 'npm install' manually inside frontend/.")
    else:
        print("\n[NOTE] Node.js or npm not detected in PATH. To run the web interface, please install Node.js 18+ LTS.")

    print("\n" + "=" * 60)
    print("Setup completed successfully!")
    print("Next steps:")
    print("  python run_demo.py doctor    # Check environment prerequisites")
    print("  python run_demo.py smoke     # Run quick model-free verification")
    print("  python run_demo.py backend   # Start backend on http://localhost:8000")
    print("  python run_demo.py frontend  # Start frontend on http://localhost:3000")
    print("=" * 60)
    return 0


def cmd_doctor() -> int:
    """Run setup doctor diagnostics."""
    venv_py = get_venv_python()
    doctor_script = ROOT / "scripts" / "doctor.py"
    return run_command([str(venv_py), str(doctor_script)])


def cmd_smoke() -> int:
    """Run model-free smoke test in strict-local network mode."""
    venv_py = get_venv_python()
    test_file = ROOT / "tests" / "test_demo_smoke.py"
    env = {"SCHOLAR_NETWORK_MODE": "strict-local"}
    return run_command([str(venv_py), "-m", "pytest", "-v", str(test_file)], env=env)


def cmd_model() -> int:
    """Detect hardware and acquire local model via Ollama."""
    venv_py = get_venv_python()
    model_script = ROOT / "scripts" / "setup_models.py"
    return run_command([str(venv_py), str(model_script)])


def cmd_backend() -> int:
    """Start the FastAPI backend server on port 8000."""
    venv_py = get_venv_python()
    print("Starting ScholAR backend on http://localhost:8000 ...")
    return run_command([str(venv_py), "-m", "uvicorn", "backend.main:app", "--port", "8000", "--reload", "--reload-dir", "backend"])


def cmd_frontend() -> int:
    """Start the Next.js frontend dev server on port 3000."""
    npm_exe = get_npm_executable()
    frontend_dir = ROOT / "frontend"
    print("Starting ScholAR frontend on http://localhost:3000 ...")
    return run_command([npm_exe, "run", "dev"], cwd=frontend_dir)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ScholAR Cross-Platform Demo Runner (macOS, Linux, Windows)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    subparsers.add_parser("setup", help="Create virtual environment, install dependencies, copy configs")
    subparsers.add_parser("doctor", help="Inspect and report environment and hardware readiness")
    subparsers.add_parser("smoke", help="Execute model-free smoke tests (offline verification)")
    subparsers.add_parser("model", help="Auto-detect hardware and acquire local Ollama model")
    subparsers.add_parser("backend", help="Start the FastAPI backend on port 8000")
    subparsers.add_parser("frontend", help="Start the Next.js frontend on port 3000")

    args = parser.parse_args()

    commands = {
        "setup": cmd_setup,
        "doctor": cmd_doctor,
        "smoke": cmd_smoke,
        "model": cmd_model,
        "backend": cmd_backend,
        "frontend": cmd_frontend,
    }

    if not args.command:
        parser.print_help()
        sys.exit(0)

    handler = commands.get(args.command)
    if handler:
        code = handler()
        sys.exit(code)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
