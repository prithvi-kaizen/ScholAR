"""Small entry point for the current ScholAR evaluation profiles."""

from __future__ import annotations

from evaluation.run_evaluation_profiles import main as run_profiles


def main() -> int:
    """Run the safe profile dispatcher (offline deterministic smoke by default)."""
    return run_profiles()


if __name__ == "__main__":
    raise SystemExit(main())
