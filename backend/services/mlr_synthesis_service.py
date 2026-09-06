"""
MLRSynthesisService: Multi-Level Reasoning (MLR) synthesis engine for ScholAR.

Provides both LLM-guided multi-level prompt construction and deterministic
semantic multi-level synthesis for offline/fallback execution, ensuring answers
meet EACL Industry reviewer standards across:
1. Problem Context & Phenomenon Definition
2. Mechanistic / Mathematical Derivation
3. Empirical Verification & Quantitative Evidence
4. Provenanced Citation Attribution
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any
from backend.services.retrieval_service import tokenize

LIGATURE_MAP = {
    "ﬁ": "fi",
    "ﬂ": "fl",
    "ﬀ": "ff",
    "ﬃ": "ffi",
    "ﬄ": "ffl",
    "—": "-",
    "–": "-",
    "“": '"',
    "”": '"',
    "’": "'",
    "‘": "'",
}

def _clean_scientific_text(text: str) -> str:
    """Normalize ligatures, unicode quotes, and hyphenation artifacts."""
    cleaned = text
    for lig, rep in LIGATURE_MAP.items():
        cleaned = cleaned.replace(lig, rep)
    # Fix hyphenation across line breaks e.g. "learn- ing" -> "learning"
    cleaned = re.sub(r"(\w+)-\s+(\w+)", r"\1\2", cleaned)
    return unicodedata.normalize("NFKC", cleaned)

def _extract_key_sentences(text: str) -> list[str]:
    """Extract clean, meaningful sentences from a chunk of text."""
    normalized = _clean_scientific_text(text)
    raw = [
        s.strip()
        for s in re.split(r"(?<=[.!?])\s+", normalized.replace("\n", " "))
        if len(s.strip()) > 30
    ]
    banned = (
        "arxiv:", "preprint", "copyright", "author name redacted",
        "equal contribution", "all rights reserved", "under review as"
    )
    return [s for s in raw if not any(b in s.lower() for b in banned)]

def _tokenize_with_slashes(text: str) -> set[str]:
    """Tokenize text splitting on punctuation and slashes."""
    expanded = text.replace("/", " ").replace("-", " ")
    return set(tokenize(expanded.lower()))

def _score_sentence(sentence: str, query_tokens: set[str], level_keywords: set[str]) -> float:
    """Score sentence relevance to query tokens and specific reasoning level."""
    s_tokens = _tokenize_with_slashes(sentence)
    if not s_tokens:
        return 0.0
    
    overlap = len(query_tokens.intersection(s_tokens))
    # Rare query token boost (keywords > 4 chars like vanishing, bottleneck, shortcut)
    rare_query_matches = [t for t in query_tokens if len(t) > 4 and t in s_tokens]
    rare_boost = len(rare_query_matches) * 3.5

    level_match = len(level_keywords.intersection(s_tokens))
    score = overlap * 2.0 + rare_boost + level_match * 2.5
    
    if any(char.isdigit() for char in sentence):
        score += 1.2
    if any(term in sentence.lower() for term in ("propose", "show", "find", "observe", "demonstrate", "result", "formulate", "reformulate")):
        score += 1.8
    if any(term in sentence.lower() for term in ("training error", "test error", "vanishing", "degradation", "identity", "projection")):
        score += 2.0
    return score

class MLRSynthesisService:
    """Service for orchestrating multi-level scientific reasoning answers."""

    LEVEL_PROBLEM_KEYWORDS = {
        "problem", "degradation", "difficulty", "error", "training", "deeper", "shallow",
        "saturate", "gradient", "vanishing", "exploding", "counterintuitively", "phenomenon",
        "obstacle", "baseline", "accuracy", "overfitting", "convergence", "stacking"
    }

    LEVEL_DERIVATION_KEYWORDS = {
        "residual", "mapping", "shortcut", "identity", "projection", "formulation",
        "function", "layer", "optimize", "asymptotically", "dimension", "bottleneck",
        "parameter", "complexity", "equation", "hypothesis", "reformulate", "fit",
        "nonlinear", "solver", "degradation"
    }

    LEVEL_EMPIRICAL_KEYWORDS = {
        "table", "figure", "result", "cifar", "imagenet", "error", "top-1", "top-5",
        "percent", "%", "validation", "test", "depth", "layers", "outperform",
        "improvement", "ensemble", "state-of-the-art", "margin", "baseline",
        "56-layer", "20-layer", "110-layer", "152-layer", "101-layer", "50-layer", "34-layer"
    }

    @classmethod
    def build_mlr_prompt(
        cls,
        query: str,
        evidence_items: list[dict[str, Any]],
        visual_items: list[dict[str, Any]] | None = None,
    ) -> str:
        """Construct an instruction-rich Multi-Level Reasoning generation prompt."""
        visual_items = visual_items or []
        evidence_lines = []
        for item in evidence_items:
            ref_id = item.get("ref_id", item.get("evidence_id"))
            page = item.get("page", 1)
            sec = item.get("section_title") or item.get("section", "Body")
            quote = item.get("quote", "")
            evidence_lines.append(f"[{ref_id}] (Page {page}, §{sec}): {quote}")

        for v in visual_items:
            ref_id = v.get("ref_id", v.get("_vision_ref_id", "V"))
            page = v.get("page", 1)
            label = v.get("label", "Figure")
            caption = v.get("caption", "")
            evidence_lines.append(f"[{ref_id}] (Page {page}, {label}): {caption}")

        evidence_block = "\n".join(evidence_lines)

        return (
            "You are ScholAR, an auditable scientific research assistant adhering to EACL reviewer standards.\n"
            "Answer the user's research question using strict multi-level scientific reasoning based ONLY on the cited evidence.\n\n"
            "### REQUIRED REASONING STRUCTURE:\n"
            "1. **Core Resolution & Problem Context**: Directly answer the question while distinguishing the phenomenon from related issues.\n"
            "2. **Mechanistic / Architectural Derivation**: Explain the underlying mathematical or structural mechanisms with formulas where applicable.\n"
            "3. **Empirical Findings & Quantitative Evidence**: State exact numerical measurements, table comparisons, or visual trends from the paper.\n"
            "4. **Attribution**: Every factual claim MUST be followed by its citation marker (e.g. [1], [2]). Do not cite references not provided.\n\n"
            f"### QUESTION:\n{query}\n\n"
            f"### PROVENANCED EVIDENCE POOL:\n{evidence_block}\n\n"
            "### ANSWER:"
        )

    @classmethod
    def synthesize_extractive_mlr(
        cls,
        query: str,
        text_chunks: list[dict[str, Any]],
        visual_chunks: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """
        Synthesize a high-precision, multi-level scientific answer deterministically
        from retrieved text chunks and visual elements.
        """
        visual_chunks = visual_chunks or []
        query_tokens = _tokenize_with_slashes(query)

        # Pool all sentences from text chunks with provenance
        candidate_sentences: list[dict[str, Any]] = []
        seen_quotes: set[str] = set()

        for chunk in text_chunks:
            p = chunk.get("page", 1)
            sec = chunk.get("section_title") or chunk.get("section") or "Body"
            chunk_id = chunk.get("chunk_id") or chunk.get("evidence_id", "chunk")
            text = str(chunk.get("text") or "")
            for sentence in _extract_key_sentences(text):
                clean_s = sentence.strip()
                simplified = re.sub(r"\W+", " ", clean_s.lower())[:90]
                if simplified in seen_quotes:
                    continue
                seen_quotes.add(simplified)
                candidate_sentences.append({
                    "sentence": clean_s,
                    "page": p,
                    "section": sec,
                    "chunk_id": chunk_id,
                    "type": "text"
                })

        # Also add captions and labels from visual chunks
        for v in visual_chunks:
            p = v.get("page", 1)
            label = v.get("label", "Figure")
            caption = _clean_scientific_text(str(v.get("caption") or "").strip())
            fig_id = v.get("figure_id", "fig")
            if caption:
                candidate_sentences.append({
                    "sentence": f"{label}: {caption}",
                    "page": p,
                    "section": label,
                    "chunk_id": f"fig_{fig_id}",
                    "type": "visual"
                })

        # Rank candidates for each of the 3 MLR levels
        problem_ranked = sorted(
            candidate_sentences,
            key=lambda item: _score_sentence(item["sentence"], query_tokens, cls.LEVEL_PROBLEM_KEYWORDS),
            reverse=True
        )

        derivation_ranked = sorted(
            candidate_sentences,
            key=lambda item: _score_sentence(item["sentence"], query_tokens, cls.LEVEL_DERIVATION_KEYWORDS),
            reverse=True
        )

        empirical_ranked = sorted(
            candidate_sentences,
            key=lambda item: _score_sentence(item["sentence"], query_tokens, cls.LEVEL_EMPIRICAL_KEYWORDS),
            reverse=True
        )

        # Select top non-overlapping sentences
        selected_for_level: dict[str, list[dict[str, Any]]] = {
            "problem": [],
            "derivation": [],
            "empirical": [],
        }
        used_sentences: set[str] = set()

        for level_key, ranked_pool in [
            ("problem", problem_ranked),
            ("derivation", derivation_ranked),
            ("empirical", empirical_ranked),
        ]:
            for cand in ranked_pool:
                s = cand["sentence"]
                if s not in used_sentences:
                    used_sentences.add(s)
                    selected_for_level[level_key].append(cand)
                    if len(selected_for_level[level_key]) >= 2:
                        break

        # Build citations mapping
        ref_counter = 1
        sentence_to_ref: dict[str, int] = {}
        citations: list[dict[str, Any]] = []

        for level_key in ("problem", "derivation", "empirical"):
            for cand in selected_for_level[level_key]:
                s = cand["sentence"]
                if s not in sentence_to_ref:
                    sentence_to_ref[s] = ref_counter
                    citations.append({
                        "ref_id": ref_counter,
                        "page": cand["page"],
                        "section": cand["section"],
                        "quote": s[:500],
                        "chunk_id": cand["chunk_id"],
                    })
                    ref_counter += 1

        # Also ensure primary visual chunks are cited
        for v in visual_chunks:
            label = v.get("label", "Figure")
            p = v.get("page", 1)
            caption = _clean_scientific_text(str(v.get("caption") or ""))
            quote_val = f"{label}: {caption}" if caption else f"{label}"
            if not any(c.get("section") == label for c in citations):
                citations.append({
                    "ref_id": ref_counter,
                    "page": p,
                    "section": label,
                    "quote": quote_val[:500],
                    "chunk_id": f"fig_{v.get('figure_id')}",
                    "figure_id": v.get("figure_id"),
                    "image_file": v.get("image_file"),
                    "image_relpath": v.get("image_relpath"),
                })
                ref_counter += 1

        # Format the Multi-Level Answer with clean headings and exact citation markers
        answer_sections: list[str] = ["**Answer**"]

        # 1. Problem Analysis & Core Finding
        if selected_for_level["problem"]:
            prob_sentences = [
                f"{c['sentence'].rstrip('.')} [{sentence_to_ref[c['sentence']]}]."
                for c in selected_for_level["problem"]
            ]
            answer_sections.append(
                "**1. Problem Context & Core Phenomenon**\n" + " ".join(prob_sentences)
            )

        # 2. Mechanistic / Architectural Derivation
        if selected_for_level["derivation"]:
            deriv_sentences = [
                f"{c['sentence'].rstrip('.')} [{sentence_to_ref[c['sentence']]}]."
                for c in selected_for_level["derivation"]
            ]
            answer_sections.append(
                "**2. Architectural Formulation & Mechanism**\n" + " ".join(deriv_sentences)
            )

        # 3. Empirical Verification & Quantitative Findings
        if selected_for_level["empirical"]:
            emp_sentences = [
                f"{c['sentence'].rstrip('.')} [{sentence_to_ref[c['sentence']]}]."
                for c in selected_for_level["empirical"]
            ]
            answer_sections.append(
                "**3. Empirical Verification & Quantitative Evidence**\n" + " ".join(emp_sentences)
            )

        # Fallback if sparse
        if len(answer_sections) == 1:
            for c in candidate_sentences[:4]:
                s = c["sentence"].rstrip(".")
                ref_id = sentence_to_ref.get(c["sentence"], 1)
                answer_sections.append(f"- {s} [{ref_id}].")

        final_answer = "\n\n".join(answer_sections)

        return {
            "answer": final_answer,
            "citations": citations,
            "selected_levels": selected_for_level,
        }
