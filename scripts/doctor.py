#!/usr/bin/env python3
"""Check whether a local ScholAR development environment is ready."""

from __future__ import annotations

import importlib.util
import json
import platform
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


ROOT = Path(__file__).resolve().parents[1]
BACKEND_ENV = ROOT / "backend" / ".env"
FRONTEND_ENV = ROOT / "frontend" / ".env.local"


@dataclass(frozen=True)
class Result:
    status: str
    name: str
    detail: str


results: list[Result] = []


def record(status: str, name: str, detail: str) -> None:
    results.append(Result(status, name, detail))


def command_output(*command: str) -> str | None:
    executable = shutil.which(command[0])
    if not executable:
        return None
    try:
        completed = subprocess.run(
            [executable, *command[1:]],
            check=False,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    output = completed.stdout.strip() or completed.stderr.strip()
    return output.splitlines()[0] if output else None


def read_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def fetch_json(url: str) -> tuple[int, Any] | None:
    request = Request(url, headers={"User-Agent": "ScholAR-setup-doctor/1.0"})
    try:
        with urlopen(request, timeout=2) as response:
            body = response.read().decode("utf-8")
            return response.status, json.loads(body)
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, OSError):
        return None


def fetch_status(url: str) -> int | None:
    request = Request(url, headers={"User-Agent": "ScholAR-setup-doctor/1.0"})
    try:
        with urlopen(request, timeout=2) as response:
            return response.status
    except HTTPError as exc:
        return exc.code
    except (URLError, TimeoutError, OSError):
        return None


def check_python() -> None:
    version = sys.version_info
    rendered = f"{version.major}.{version.minor}.{version.micro} at {sys.executable}"
    if version < (3, 11):
        record("FAIL", "Python", f"{rendered}; install Python 3.11 or 3.12")
    elif version[:2] not in {(3, 11), (3, 12)}:
        record("WARN", "Python", f"{rendered}; supported versions are 3.11 and 3.12")
    else:
        record("PASS", "Python", rendered)

    venv_root = Path(sys.prefix).resolve()
    expected_venv = (ROOT / ".venv").resolve()
    if venv_root == expected_venv:
        record("PASS", "Virtual environment", str(expected_venv))
    else:
        record("WARN", "Virtual environment", f"active interpreter is not {expected_venv}")


def check_python_packages() -> None:
    packages = {
        "fastapi": "fastapi",
        "uvicorn": "uvicorn",
        "httpx": "httpx",
        "PyMuPDF": "fitz",
        "python-dotenv": "dotenv",
        "pydantic": "pydantic",
        "python-multipart": "multipart",
    }
    missing = [name for name, module in packages.items() if importlib.util.find_spec(module) is None]
    if missing:
        record("FAIL", "Python packages", f"missing {', '.join(missing)}; run make setup")
    else:
        record("PASS", "Python packages", "all runtime imports are available")


def major_version(output: str | None) -> int | None:
    if not output:
        return None
    match = re.search(r"v?(\d+)(?:\.\d+)+", output)
    return int(match.group(1)) if match else None


def check_javascript() -> None:
    node = command_output("node", "--version")
    node_major = major_version(node)
    if node_major is None:
        record("FAIL", "Node.js", "not found; install Node.js 20 LTS")
    elif node_major < 18:
        record("FAIL", "Node.js", f"{node}; version 18 or newer is required")
    elif node_major != 20:
        record("WARN", "Node.js", f"{node}; Node.js 20 LTS is the CI version")
    else:
        record("PASS", "Node.js", node or "version 20")

    npm = command_output("npm", "--version")
    if npm:
        record("PASS", "npm", npm)
    else:
        record("FAIL", "npm", "not found; install it with Node.js")

    node_modules = ROOT / "frontend" / "node_modules"
    if node_modules.is_dir():
        record("PASS", "Frontend packages", str(node_modules))
    else:
        record("FAIL", "Frontend packages", "missing; run make setup or cd frontend && npm ci")


def check_configuration() -> tuple[str, str]:
    if BACKEND_ENV.exists():
        record("PASS", "Backend environment", str(BACKEND_ENV))
    else:
        record("WARN", "Backend environment", "missing; copy backend/.env.example to backend/.env")

    if FRONTEND_ENV.exists():
        record("PASS", "Frontend environment", str(FRONTEND_ENV))
    else:
        record("WARN", "Frontend environment", "missing; copy frontend/.env.local.example to frontend/.env.local")

    data_dir = ROOT / "backend" / "data" / "papers"
    if data_dir.is_dir():
        record("PASS", "Paper store", str(data_dir))
    else:
        record("FAIL", "Paper store", f"missing directory: {data_dir}")

    env = read_env(BACKEND_ENV)
    base_url = env.get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
    model = env.get("OLLAMA_MODEL", "qwen3.5:9b")
    return base_url, model


def check_ollama(base_url: str, model: str) -> None:
    version = command_output("ollama", "--version")
    if version:
        record("PASS", "Ollama CLI", version)
    else:
        record("WARN", "Ollama CLI", "not found; install Ollama to enable generated answers")

    response = fetch_json(f"{base_url}/api/tags")
    if response is None:
        record("WARN", "Ollama service", f"not reachable at {base_url}; start Ollama")
        return

    status, payload = response
    if status != 200 or not isinstance(payload, dict):
        record("WARN", "Ollama service", f"unexpected response from {base_url}")
        return
    record("PASS", "Ollama service", base_url)

    available = {
        item.get("name")
        for item in payload.get("models", [])
        if isinstance(item, dict) and isinstance(item.get("name"), str)
    }
    normalized_available = {name.removesuffix(":latest") for name in available}
    if model in available or model.removesuffix(":latest") in normalized_available:
        record("PASS", "Configured model", model)
    else:
        record("WARN", "Configured model", f"{model} is not installed; run ollama pull {model}")


def check_running_services() -> None:
    backend = fetch_json("http://localhost:8000/health")
    if backend and backend[0] == 200 and isinstance(backend[1], dict):
        state = backend[1]
        detail = f"status={state.get('status')}, model={state.get('model')}, ollama={state.get('ollama_available')}"
        record("PASS", "FastAPI backend", detail)
    else:
        record("INFO", "FastAPI backend", "not running; start it with make backend")

    frontend_status = fetch_status("http://localhost:3000")
    if frontend_status is not None and frontend_status < 500:
        record("PASS", "Next.js frontend", f"http://localhost:3000 returned HTTP {frontend_status}")
    else:
        record("INFO", "Next.js frontend", "not running; start it with make frontend")


def print_report() -> None:
    labels = {
        "PASS": "[PASS]",
        "WARN": "[WARN]",
        "FAIL": "[FAIL]",
        "INFO": "[INFO]",
    }
    print("ScholAR setup doctor")
    print(f"Repository: {ROOT}")
    print(f"System: {platform.system()} {platform.release()} ({platform.machine()})")
    print()
    width = max(len(result.name) for result in results)
    for result in results:
        print(f"{labels[result.status]:6} {result.name:<{width}}  {result.detail}")

    failures = sum(result.status == "FAIL" for result in results)
    warnings = sum(result.status == "WARN" for result in results)
    print()
    if failures:
        print(f"Setup is incomplete: {failures} failure(s), {warnings} warning(s).")
        print("Fix the FAIL rows first, then run this command again.")
    elif warnings:
        print(f"Core dependencies are ready with {warnings} warning(s).")
        print("Generated AI answers require a reachable Ollama service and the configured model.")
    else:
        print("Everything checked is ready.")


def main() -> int:
    check_python()
    check_python_packages()
    check_javascript()
    base_url, model = check_configuration()
    check_ollama(base_url, model)
    check_running_services()
    print_report()
    return 1 if any(result.status == "FAIL" for result in results) else 0


if __name__ == "__main__":
    raise SystemExit(main())
