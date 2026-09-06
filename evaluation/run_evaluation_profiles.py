#!/usr/bin/env python3
"""Truthful profile dispatcher for ScholAR evaluation and reproduction."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import socket
import subprocess
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.network_policy_service import NetworkPolicyService  # noqa: E402


@dataclass(frozen=True)
class Command:
    label: str
    argv: tuple[str, ...]
    outputs: tuple[str, ...] = ()


def py_command(label: str, script: str, *args: str, outputs: tuple[str, ...] = ()) -> Command:
    return Command(label, (PYTHON, script, *args), outputs)


def smoke_commands() -> list[Command]:
    return [
        py_command("release-v1 artifact reproduction", "evaluation/reproduce_release_fixture.py"),
        py_command("human/ethics template safety", "evaluation/validate_human_templates.py"),
        py_command("EACL paper draft provenance", "evaluation/validate_paper.py", "--paper-dir", "paper/eacl_industry"),
        py_command("scaled benchmark builder", "evaluation/build_scaled_benchmark.py", "--selfcheck"),
        py_command("case mining helpers", "evaluation/mine_cases.py", "--selfcheck"),
        py_command("comparison evaluator", "evaluation/run_comparison_eval.py", "--selfcheck"),
        py_command("abstention evaluator", "evaluation/run_abstention_eval.py", "--selfcheck"),
        py_command("efficiency evaluator", "evaluation/run_efficiency_eval.py", "--selfcheck"),
        py_command("M3SciQA evaluator", "evaluation/m3sciqa/run_m3sciqa_eval.py", "--selfcheck"),
        Command("unit tests", (PYTHON, "-m", "unittest", "discover", "-s", "tests")),
    ]


def artifact_only_commands() -> list[Command]:
    return [
        py_command("reproduce release-v1 fixture", "evaluation/reproduce_release_fixture.py"),
        py_command("validate human/ethics templates", "evaluation/validate_human_templates.py"),
        py_command("validate EACL paper provenance", "evaluation/validate_paper.py", "--paper-dir", "paper/eacl_industry"),
    ]


def measured_retrieval_commands() -> list[Command]:
    return [
        py_command(
            "hand-labeled retrieval anchor",
            "evaluation/run_retrieval_eval.py",
            outputs=(
                "evaluation/results/retrieval_eval_results.json",
                "evaluation/results/retrieval_eval_report.md",
            ),
        ),
        py_command(
            "scaled retrieval benchmark",
            "evaluation/run_retrieval_eval.py",
            "--cases",
            "evaluation/benchmark_cases_scaled.json",
            "--tag",
            "scaled",
            outputs=(
                "evaluation/results/retrieval_eval_results_scaled.json",
                "evaluation/results/retrieval_eval_report_scaled.md",
            ),
        ),
    ]


def model_backed_commands(model: str, backend: str, limit: int) -> list[Command]:
    return [
        py_command(
            "generated-answer faithfulness",
            "evaluation/run_generation_faithfulness_eval.py",
            "--backend",
            backend,
            "--model",
            model,
            "--limit",
            str(limit),
            "--require-encoder",
            outputs=(
                "evaluation/results/generation_faithfulness_results.json",
                "evaluation/results/generation_faithfulness_report.md",
            ),
        ),
        py_command(
            "abstention generation",
            "evaluation/run_abstention_eval.py",
            "--model",
            model,
            "--backend",
            backend,
            outputs=("evaluation/results/abstention_results.json",),
        ),
        py_command(
            "single-machine efficiency",
            "evaluation/run_efficiency_eval.py",
            "--model",
            model,
            "--backend",
            backend,
            "--n",
            str(limit),
            outputs=("evaluation/results/efficiency_results.json",),
        ),
    ]


def prerequisites(profile: str) -> list[str]:
    common = ["Run from a repository checkout with the project Python dependencies installed."]
    if profile == "smoke":
        return common + [
            "No papers, model server, or network access is required.",
            "The profile sets Hugging Face/Transformers offline flags and writes no result artifacts.",
        ]
    if profile == "artifact-only":
        return common + [
            "The release fixture, human-study templates, and paper source must be present.",
            "This profile validates those artifacts without running models or rewriting measured results.",
            "No papers, model server, encoder, or network access is required.",
        ]
    if profile == "measured-retrieval":
        return common + [
            "Prepared local paper chunks and at least one extracted benchmark image must exist under backend/data/papers/.",
            "The all-MiniLM-L6-v2 and configured paired image/text snapshots must already exist in the local Hugging Face cache.",
            "The versioned four-channel condition fails closed rather than silently measuring a text-only degradation.",
            "No acquisition is performed and offline mode is enforced.",
        ]
    return common + [
        "Pass --model with an already-pulled Ollama model.",
        "A local backend and local Ollama must be running at loopback URLs.",
        "Prepared local paper chunks must exist under backend/data/papers/.",
        "The all-MiniLM-L6-v2 and configured paired image/text snapshots must already exist in the local Hugging Face cache.",
        "No model pulling, paper acquisition, Git clone, or other network acquisition is performed.",
    ]


def format_command(command: Command) -> str:
    return shlex.join(command.argv)


def ensure_loopback(url: str, label: str) -> tuple[str, int]:
    parsed = urlparse(url)
    if not NetworkPolicyService.is_loopback_url(url):
        raise SystemExit(f"{label} must use a loopback HTTP(S) URL, got {url!r}")
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    return parsed.hostname, port


def ensure_socket(url: str, label: str) -> None:
    host, port = ensure_loopback(url, label)
    try:
        with socket.create_connection((host, port), timeout=2):
            pass
    except OSError as exc:
        raise SystemExit(f"Cannot reach {label} at {url}: {exc}") from exc


def ensure_backend_policy(backend_url: str) -> None:
    ensure_socket(backend_url, "backend")
    try:
        with urllib.request.urlopen(
            f"{backend_url.rstrip('/')}/api/system/network-policy", timeout=5
        ) as response:
            policy = json.loads(response.read().decode())
    except Exception as exc:
        raise SystemExit(f"Cannot query backend network policy at {backend_url}: {exc}") from exc
    if policy.get("mode") != "strict-local" or policy.get("external_network_allowed") is not False:
        raise SystemExit(
            "Model-backed evaluation requires a backend running in SCHOLAR_NETWORK_MODE=strict-local"
        )


def ensure_local_model(ollama_url: str, model: str) -> None:
    ensure_socket(ollama_url, "Ollama")
    try:
        with urllib.request.urlopen(f"{ollama_url.rstrip('/')}/api/tags", timeout=5) as response:
            models = json.loads(response.read().decode()).get("models", [])
    except Exception as exc:
        raise SystemExit(f"Cannot query local Ollama models at {ollama_url}: {exc}") from exc
    names = {item.get("name") or item.get("model") for item in models}
    if model not in names:
        raise SystemExit(f"Ollama model {model!r} is not installed locally; available: {sorted(names)}")


def ensure_prepared_papers() -> None:
    if not any((ROOT / "backend" / "data" / "papers").glob("*/chunks.json")):
        raise SystemExit("No prepared paper chunks found under backend/data/papers/.")


def ensure_prepared_visuals() -> None:
    case_paths = (
        ROOT / "evaluation" / "benchmark_cases.json",
        ROOT / "evaluation" / "benchmark_cases_scaled.json",
    )
    paper_ids: set[str] = set()
    for case_path in case_paths:
        if not case_path.is_file():
            continue
        payload = json.loads(case_path.read_text(encoding="utf-8"))
        paper_ids.update(
            str(case.get("paper_id") or "")
            for case in payload
            if isinstance(case, dict) and case.get("paper_id")
        )

    usable_images = 0
    for paper_id in sorted(paper_ids):
        directory = ROOT / "backend" / "data" / "papers" / paper_id
        figures_path = directory / "figures.json"
        if not figures_path.is_file():
            continue
        figures = json.loads(figures_path.read_text(encoding="utf-8"))
        for figure in figures if isinstance(figures, list) else []:
            image_file = str(figure.get("image_file") or "") if isinstance(figure, dict) else ""
            if image_file and Path(image_file).name == image_file and (
                directory / "figures" / image_file
            ).is_file():
                usable_images += 1
    if usable_images == 0:
        raise SystemExit(
            "Measured four-channel retrieval requires at least one extracted benchmark image."
        )


def _cached_hf_snapshot_exists(model: str) -> bool:
    local_path = Path(model).expanduser()
    if local_path.is_dir():
        return any(path.is_file() for path in local_path.rglob("*"))
    cache_name = "models--" + model.replace("/", "--")
    snapshots = Path.home() / ".cache" / "huggingface" / "hub" / cache_name / "snapshots"
    return snapshots.is_dir() and any(path.is_dir() for path in snapshots.iterdir())


def ensure_cached_encoders() -> None:
    text_model = "sentence-transformers/all-MiniLM-L6-v2"
    if not _cached_hf_snapshot_exists(text_model):
        raise SystemExit(
            f"Local {text_model} cache is missing; measured retrieval will not acquire it."
        )
    visual_model = os.getenv(
        "SCHOLAR_VISUAL_EMBEDDING_MODEL",
        "openai/clip-vit-base-patch32",
    )
    if not _cached_hf_snapshot_exists(visual_model):
        raise SystemExit(
            f"Local {visual_model} cache is missing; the versioned four-channel "
            "condition will not run in a degraded state or acquire it."
        )

    from backend.services.visual_embedding_service import VisualEmbeddingService

    VisualEmbeddingService.initialize(visual_model)
    status = VisualEmbeddingService.status()
    if not status.get("model_loaded") or not status.get("encoder_fingerprint"):
        raise SystemExit(
            "The cached visual encoder failed load/identity preflight: "
            + str(status.get("fallback_reason") or "unknown failure")
        )
    VisualEmbeddingService.release()


def ensure_overwrite_ack(commands: list[Command], allowed: bool) -> None:
    existing = sorted(
        {output for command in commands for output in command.outputs if (ROOT / output).exists()}
    )
    if existing and not allowed:
        joined = "\n  - ".join(existing)
        raise SystemExit(
            "Execution would rewrite committed/current artifacts:\n  - "
            + joined
            + "\nRe-run with --allow-current-result-overwrite only after preserving or reviewing them."
        )


def run_commands(commands: list[Command], env: dict[str, str]) -> None:
    for index, command in enumerate(commands, 1):
        print(f"\n[{index}/{len(commands)}] {command.label}\n$ {format_command(command)}", flush=True)
        subprocess.run(command.argv, cwd=ROOT, env=env, check=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run deterministic smoke checks or explicitly selected evaluation profiles."
    )
    parser.add_argument(
        "profile",
        nargs="?",
        default="smoke",
        choices=["smoke", "artifact-only", "measured-retrieval", "model-backed", "full"],
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="execute a non-smoke profile; otherwise print its commands and prerequisites",
    )
    parser.add_argument(
        "--allow-current-result-overwrite",
        action="store_true",
        help="acknowledge that an executing non-smoke profile may rewrite current result artifacts",
    )
    parser.add_argument("--model", help="already-installed local Ollama model for model-backed/full")
    parser.add_argument("--backend", default="http://127.0.0.1:8000")
    parser.add_argument("--ollama", default=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"))
    parser.add_argument("--limit", type=int, default=5, help="positive smoke-sized model sample")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.limit < 1:
        raise SystemExit("--limit must be positive")

    model = args.model or "<required: --model>"
    if args.profile == "smoke":
        commands = smoke_commands()
    elif args.profile == "artifact-only":
        commands = artifact_only_commands()
    elif args.profile == "measured-retrieval":
        commands = measured_retrieval_commands()
    elif args.profile == "model-backed":
        commands = model_backed_commands(model, args.backend, args.limit)
    else:
        commands = smoke_commands() + measured_retrieval_commands()
        commands += model_backed_commands(model, args.backend, args.limit)

    print(f"ScholAR evaluation profile: {args.profile}")
    print("Prerequisites:")
    for item in prerequisites(args.profile):
        print(f"  - {item}")
    print("Commands:")
    for command in commands:
        print(f"  - {command.label}: {format_command(command)}")

    if args.profile != "smoke" and not args.execute:
        print("\nPlan only. Add --execute to run this explicit profile.")
        return 0

    if args.profile in {"model-backed", "full"}:
        if not args.model:
            raise SystemExit("--model is required when executing model-backed or full")
        ensure_backend_policy(args.backend)
        ensure_local_model(args.ollama, args.model)
        ensure_prepared_papers()
    if args.profile == "measured-retrieval":
        ensure_prepared_papers()
    if args.profile in {"measured-retrieval", "full"}:
        ensure_prepared_visuals()
    if args.profile in {"model-backed", "measured-retrieval", "full"}:
        ensure_cached_encoders()
    if args.profile != "smoke":
        ensure_overwrite_ack(commands, args.allow_current_result_overwrite)

    env = os.environ.copy()
    env.update(
        {
            "PYTHONHASHSEED": "0",
            "SCHOLAR_NETWORK_MODE": "strict-local",
            "HF_HUB_OFFLINE": "1",
            "HF_DATASETS_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "OLLAMA_BASE_URL": (
                args.ollama if args.profile in {"model-backed", "full"} else "http://127.0.0.1:9"
            ),
        }
    )
    run_commands(commands, env)
    print(f"\nProfile {args.profile!r} completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
