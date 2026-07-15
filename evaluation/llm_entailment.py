"""
llm_entailment.py
A local LLM-as-judge entailment classifier for faithfulness scoring.

Why this exists: the previous scorer (nli_faithfulness.py, "semantic" mode) labeled an atom
ENTAILMENT / NEUTRAL / CONTRADICTION by sentence-embedding COSINE similarity against the
retrieved chunk. Cosine similarity cannot detect contradiction: a claim and its negation are
near neighbours in embedding space, so the "contradiction" bin only ever caught topical
mismatch, never an actual conflicting fact. This module replaces that with a real entailment
decision made by a local instruction model (Ollama), which CAN separate contradiction from
"not enough information". It is fully local, so the local-only design is preserved.

The judge model is fixed and configurable (default qwen3.5:9b) so that scoring is reproducible
and independent of which model generated the answer being scored.

Usage:
    from llm_entailment import classify
    label, score = classify("X used a minibatch of 80", evidence_text)   # -> ("ENTAILMENT", 1.0)
Environment:
    FAITH_JUDGE_MODEL   judge model tag (default qwen3.5:9b)
    OLLAMA_BASE_URL     default http://localhost:11434
"""
from __future__ import annotations

import os
import sys

import httpx

OLLAMA = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
JUDGE_MODEL = os.getenv("FAITH_JUDGE_MODEL", "qwen3.5:9b")

_PROMPT = (
    "You are a strict scientific fact-checker. Decide how the EVIDENCE relates to the CLAIM.\n"
    "Reply with exactly ONE word:\n"
    "ENTAILMENT if the evidence states or directly implies the claim.\n"
    "CONTRADICTION if the evidence asserts something that conflicts with the claim "
    "(a different number, an opposite relationship, a negation).\n"
    "NEUTRAL if the evidence neither supports nor conflicts (not enough information).\n\n"
    "EVIDENCE:\n{premise}\n\nCLAIM: {hypothesis}\n\nOne word:"
)

_LABELS = ("ENTAILMENT", "CONTRADICTION", "NEUTRAL")
_SCORE = {"ENTAILMENT": 1.0, "NEUTRAL": 0.5, "CONTRADICTION": 0.0}


def _parse(text: str) -> str:
    up = text.upper()
    # exact tokens first, contradiction before entailment (an answer may contain both words)
    for lab in ("CONTRADICTION", "ENTAILMENT", "NEUTRAL"):
        if lab in up:
            return lab
    if "CONTRADICT" in up or "CONFLICT" in up:
        return "CONTRADICTION"
    if "ENTAIL" in up or "SUPPORT" in up:
        return "ENTAILMENT"
    return "NEUTRAL"


def classify(hypothesis: str, premise: str, model: str | None = None) -> tuple[str, float]:
    """Return (label, score) for whether `premise` entails `hypothesis`. Local Ollama call."""
    payload = {
        "model": model or JUDGE_MODEL,
        "prompt": _PROMPT.format(premise=str(premise)[:4000], hypothesis=str(hypothesis)[:800]),
        "stream": False, "think": False,
        "options": {"temperature": 0.0, "num_predict": 8},
    }
    try:
        with httpx.Client(timeout=120.0, trust_env=False) as c:
            r = c.post(f"{OLLAMA}/api/generate", json=payload)
            r.raise_for_status()
            label = _parse(r.json().get("response", ""))
    except Exception:
        label = "NEUTRAL"  # a failed judge call must not fabricate support
    return label, _SCORE[label]


def _selfcheck() -> None:
    # parser (deterministic, no network)
    assert _parse("ENTAILMENT") == "ENTAILMENT"
    assert _parse("The answer is CONTRADICTION.") == "CONTRADICTION"
    assert _parse("clearly contradicts") == "CONTRADICTION"
    assert _parse("not enough info") == "NEUTRAL"
    print("parser selfcheck OK")
    # live smoke (only if Ollama reachable)
    try:
        httpx.Client(timeout=3.0, trust_env=False).get(f"{OLLAMA}/api/tags")
    except Exception:
        print("(ollama not reachable; skipping live smoke)")
        return
    ent, _ = classify("The minibatch size was 80 sentences.",
                      "Each SGD update uses a minibatch of 80 sentences.")
    con, _ = classify("The minibatch size was 512 sentences.",
                      "Each SGD update uses a minibatch of 80 sentences.")
    neu, _ = classify("The model was trained for 5 days.",
                      "Each SGD update uses a minibatch of 80 sentences.")
    print(f"live: entail->{ent}  contra(80 vs 512)->{con}  neutral->{neu}")
    assert ent == "ENTAILMENT", ent
    assert con == "CONTRADICTION", f"judge failed to catch a number conflict: {con}"
    print("live selfcheck OK")


if __name__ == "__main__":
    _selfcheck() if "--selfcheck" in sys.argv else print(classify(sys.argv[1], sys.argv[2]))
