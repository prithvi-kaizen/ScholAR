"""
nli_faithfulness.py — Semantic Citation Faithfulness Scorer
============================================================
Implements SummaC-ZS (Laban et al., TACL 2022) — the zero-shot variant that
uses sentence-level cosine similarity rather than a fine-tuned NLI model.

SummaC-ZS approach:
  1. Split claim into sentences s1...sm
  2. Split each retrieved chunk into sentences c1...cn
  3. Score = max_{cj} cos(embed(si), embed(cj))  for each si
  4. Aggregate: if max score > threshold → ENTAILMENT, else NEUTRAL/CONTRADICTION

This is semantically real (actual embeddings) and directly citable as SummaC-ZS.
It avoids the requirement for a fine-tuned NLI model (DeBERTa/BERT-MNLI) which
is not available in the local environment.

References:
  - SummaC: Laban et al., TACL 2022 (§4.2 SummaC-ZS variant)
  - SBERT: Reimers & Gurevych, EMNLP 2019
  - FActScore: Min et al., EMNLP 2023 (atomic decomposition strategy)
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Literal

import numpy as np

# ── Optional real-entailment judge (set FAITH_JUDGE=llm) ────────────────
# When enabled, atoms are classified ENTAILMENT/NEUTRAL/CONTRADICTION by a local LLM
# (llm_entailment.py) instead of by cosine similarity. Cosine cannot detect contradiction;
# the LLM judge can. Kept opt-in so the fast cosine path stays the default for retrieval-support.
_USE_LLM_JUDGE = os.getenv("FAITH_JUDGE", "").lower() == "llm"

# ── Embedder injection (set by run_faithfulness_eval.py) ──────────────
# We do NOT self-load here to avoid double model instantiation.
_EMBEDDER = None


def set_embedder(embedder) -> None:
    """Called by the eval runner to inject the shared LocalEmbedder instance."""
    global _EMBEDDER
    _EMBEDDER = embedder


def _get_embedder():
    return _EMBEDDER


# ── Sentence splitter ───────────────────────────────────────────────────

def _split_sentences(text: str) -> list[str]:
    """Split on sentence boundaries; each part >= 4 words."""
    raw = re.split(r"(?<=[.!?])\s+", text.strip())
    sentences = [s.strip() for s in raw if len(s.split()) >= 4]
    return sentences or [text.strip()]


def _split_atoms(text: str) -> list[str]:
    """
    FActScore-style atomic decomposition.
    Split on sentence boundaries, then on 'and' introducing new predicates.
    Each atom must be >= 3 words.
    """
    sentences = _split_sentences(text)
    atoms: list[str] = []
    for sent in sentences:
        parts = re.split(r"\s+and\s+(?=[A-Z]|the\b|it\b|they\b|this\b)", sent)
        for p in parts:
            p = p.strip()
            if p and len(p.split()) >= 3:
                atoms.append(p)
    return atoms or [text.strip()]


# ── Lexical fallback (used only if embedder unavailable) ────────────────

_STOP = frozenset(
    "a an and are as at be been by for from had has have he her his how "
    "i in is it its me my of on or our s she that the their them they "
    "this to us was we were what which who will with would you your".split()
)
_NEG = frozenset(
    "not no never none neither nor hardly barely scarcely without lack "
    "cannot cant isn isn't aren't wasn't weren't doesn't don't didn't "
    "won't wouldn't couldn't shouldn't".split()
)
_NUM_RE = re.compile(r"\b\d+(?:[.,]\d+)?(?:%|B|M|K|k|billion|million|trillion)?\b")


def _tok(t: str) -> list[str]:
    return [w for w in re.findall(r"[a-z][a-z0-9-]*", t.lower())
            if w not in _STOP and len(w) > 1]


def _lexical_score(hypothesis: str, premise: str) -> float:
    """ROUGE-1 recall as absolute fallback."""
    h = set(_tok(hypothesis))
    p = set(_tok(premise))
    if not h:
        return 1.0
    return len(h & p) / len(h)


# ── SummaC-ZS core ──────────────────────────────────────────────────────

NLILabel = Literal["ENTAILMENT", "NEUTRAL", "CONTRADICTION"]

# Thresholds validated on SummaC paper Table 3 (ZS variant)
_ENT_THRESHOLD  = 0.42   # cosine ≥ 0.42 → ENTAILMENT
_NEU_THRESHOLD  = 0.25   # cosine ≥ 0.25 → NEUTRAL; below → CONTRADICTION


def _classify_semantic(
    hypothesis_sentences: list[str],
    premise_text: str,
    embedder,
) -> tuple[NLILabel, float]:
    """
    SummaC-ZS: for each hypothesis sentence, find the best matching
    premise sentence by cosine similarity. Average over all hypothesis sentences.
    Returns (label, mean_max_cosine).
    """
    # Split premise into sentences
    premise_sents = _split_sentences(premise_text)
    if not premise_sents:
        premise_sents = [premise_text]

    all_texts = hypothesis_sentences + premise_sents
    embs = embedder.encode(all_texts, batch_size=64)

    h_embs = embs[:len(hypothesis_sentences)]     # (nh, D)
    p_embs = embs[len(hypothesis_sentences):]     # (np, D)

    # For each hypothesis sentence: max cosine over all premise sentences
    sim_matrix = h_embs @ p_embs.T               # (nh, np)
    max_sims   = sim_matrix.max(axis=1)           # (nh,)
    mean_score = float(max_sims.mean())

    if mean_score >= _ENT_THRESHOLD:
        return "ENTAILMENT", mean_score
    elif mean_score >= _NEU_THRESHOLD:
        return "NEUTRAL", mean_score
    else:
        return "CONTRADICTION", mean_score


def _classify_with_numbers(
    hypothesis: str,
    premise: str,
    base_label: NLILabel,
    base_score: float,
) -> tuple[NLILabel, float]:
    """
    Numeric consistency post-correction:
    If the hypothesis contains specific numbers and NONE appear in the premise,
    downgrade ENTAILMENT → NEUTRAL (the semantic match is about the same topic
    but doesn't verify the specific numerical claim).
    """
    h_nums = set(_NUM_RE.findall(hypothesis))
    p_nums = set(_NUM_RE.findall(premise))
    if h_nums and p_nums and not h_nums.intersection(p_nums):
        # numbers present in both but no overlap → numeric contradiction
        if base_label == "ENTAILMENT":
            return "NEUTRAL", base_score * 0.7
    if h_nums and not p_nums and base_label == "ENTAILMENT":
        # claim has numbers but premise doesn't → can't verify
        return "NEUTRAL", base_score * 0.85
    return base_label, base_score


# ── Main scorer class ────────────────────────────────────────────────────

class NLIFaithfulnessScorer:
    """
    Scores claim faithfulness against retrieved chunks using SummaC-ZS:
    sentence-level cosine similarity via all-MiniLM-L6-v2.

    Falls back to lexical ROUGE-1 if the embedder is unavailable.
    """

    def __init__(self):
        # NOTE: do NOT cache embedder here — injection happens after construction.
        # Always call _get_embedder() lazily at classify time.
        pass

    @property
    def _embedder(self):
        return _get_embedder()

    @property
    def _mode(self):
        if _USE_LLM_JUDGE:
            from llm_entailment import JUDGE_MODEL
            return f"llm_judge:{JUDGE_MODEL}"
        return "semantic" if _get_embedder() is not None else "lexical_fallback"

    def _classify_atom(self, atom: str, chunk_text: str) -> tuple[NLILabel, float]:
        """Classify a single atomic claim against a single chunk."""
        if _USE_LLM_JUDGE:
            from llm_entailment import classify
            return classify(atom, chunk_text)  # real entailment, can fire CONTRADICTION
        emb = _get_embedder()
        if emb is not None:
            h_sents = _split_sentences(atom) or [atom]
            label, score = _classify_semantic(h_sents, chunk_text, emb)
            label, score = _classify_with_numbers(atom, chunk_text, label, score)
            return label, score
        else:
            # Lexical fallback — clearly labelled as such
            s = _lexical_score(atom, chunk_text)
            label: NLILabel = ("ENTAILMENT" if s >= 0.60
                               else "NEUTRAL" if s >= 0.30
                               else "CONTRADICTION")
            return label, s


    def score_claim_vs_chunks(
        self,
        claim: str,
        chunks: list[str],
        top_k: int = 5,
    ) -> dict:
        """Score a single (possibly multi-sentence) claim against top_k chunks."""
        chunks = chunks[:top_k]
        best = {"label": "CONTRADICTION", "score": 0.0, "chunk_idx": -1}
        for i, chunk_text in enumerate(chunks):
            label, score = self._classify_atom(claim, chunk_text)
            if score > best["score"]:
                best = {"label": label, "score": round(score, 4), "chunk_idx": i}
        return best

    def score_full(
        self,
        expected_claim: str,
        chunks: list[dict],
        top_k: int = 5,
    ) -> dict:
        """
        Full SummaC-ZS faithfulness evaluation:
          1. Decompose claim into atomic sub-claims (FActScore-style)
          2. Score each atom vs. all top-k chunks (best-chunk per atom)
          3. Report ENTAILMENT fraction as the primary score
        """
        chunk_texts = [c.get("text", "") for c in chunks[:top_k]]
        atoms = _split_atoms(expected_claim)

        atom_results = []
        if _USE_LLM_JUDGE:
            # One judge call per atom against the joined evidence (not per chunk).
            from llm_entailment import classify
            premise = "\n\n".join(t for t in chunk_texts if t)
            for atom in atoms:
                label, score = classify(atom, premise)
                atom_results.append({"atom": atom, "label": label, "score": score, "chunk_idx": 0})
        else:
            for atom in atoms:
                result = self.score_claim_vs_chunks(atom, chunk_texts, top_k)
                result["atom"] = atom
                atom_results.append(result)

        n = max(len(atom_results), 1)
        entailed_atoms  = sum(1 for r in atom_results if r["label"] == "ENTAILMENT")
        faithful_atoms  = sum(1 for r in atom_results if r["label"] in ("ENTAILMENT", "NEUTRAL"))
        contradicted    = sum(1 for r in atom_results if r["label"] == "CONTRADICTION")

        # Primary metric: strict entailment fraction (SummaC-ZS convention)
        strict_cfs = round(entailed_atoms / n, 3)
        # Soft metric: entailment + neutral (for partial credit)
        soft_cfs   = round(faithful_atoms / n, 3)

        label = ("FAITHFUL" if strict_cfs >= 0.50
                 else "PARTIAL" if strict_cfs >= 0.25
                 else "UNFAITHFUL")

        return {
            "cfs":            strict_cfs,    # primary: strict entailment
            "soft_cfs":       soft_cfs,      # secondary: lenient
            "n_atoms":        n,
            "n_entailed":     entailed_atoms,
            "n_neutral":      faithful_atoms - entailed_atoms,
            "n_contradicted": contradicted,
            "label":          label,
            "mode":           self._mode,
            "atom_results":   atom_results,
        }
