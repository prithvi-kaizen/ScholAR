from __future__ import annotations

import math
import re
import hashlib
from collections import Counter
from typing import Any

# Visual-cue terms: when any of these appear in the query the retriever
# boosts figure/table chunks so they compete effectively against text chunks.
_VISUAL_CUE_TERMS = frozenset({
    "figure", "fig", "table", "plot", "chart", "graph", "diagram",
    "shown", "depicted", "illustrat", "visuali", "image", "schematic",
    "architecture diagram", "attention map", "heatmap",
})
_VISUAL_BOOST = 2.5   # multiplier applied to figure/table chunk score
_CAPTION_MATCH_BONUS = 1.8  # extra score when query tokens overlap with caption


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "how",
    "in",
    "is",
    "it",
    "main",
    "of",
    "on",
    "or",
    "paper",
    "propose",
    "proposed",
    "that",
    "the",
    "this",
    "to",
    "what",
    "with",
}

QUERY_EXPANSIONS = {
    "result": ["result", "results", "achieve", "outperform", "significant", "comparison", "table", "accuracy", "bleu", "score"],
    "results": ["result", "results", "achieve", "outperform", "significant", "comparison", "table", "accuracy", "bleu", "score"],
    "finding": ["finding", "findings", "result", "evidence", "show", "support"],
    "findings": ["finding", "findings", "result", "evidence", "show", "support"],
    "method": ["method", "approach", "framework", "procedure", "algorithm", "architecture"],
    "contribution": ["contribution", "introduce", "propose", "present", "novel", "new", "main"],
    "contributions": ["contribution", "introduce", "propose", "present", "novel", "new", "main"],
    "architecture": ["architecture", "model", "component", "layer", "module", "algorithm"],
    "experiment": ["experiment", "evaluation", "dataset", "benchmark", "baseline", "metric"],
    "experiments": ["experiment", "evaluation", "dataset", "benchmark", "baseline", "metric"],
    "limitation": ["limitation", "limits", "failure", "assumption", "future", "caveat"],
    "limitations": ["limitation", "limits", "failure", "assumption", "future", "caveat"],
}


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
        if token not in STOP_WORDS and len(token) > 2
    ]


def expand_query_terms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_EXPANSIONS.get(token, []))
    return expanded


def extract_page_hints(message: str) -> list[int]:
    pages: list[int] = []
    for match in re.finditer(r"\bpages?\s+([0-9,\sand]+)", message.lower()):
        for value in re.findall(r"\d+", match.group(1)):
            page = int(value)
            if page not in pages:
                pages.append(page)
    return pages[:6]


def _section_hints(message: str) -> set[str]:
    lowered = message.lower()
    hints: set[str] = set()
    if any(term in lowered for term in ("abstract", "summary", "motivation", "problem", "contribution")):
        hints.update({"abstract", "introduction"})
    if any(term in lowered for term in ("method", "methodology", "architecture", "algorithm", "component", "workflow")):
        hints.update({"method", "body"})
    if any(term in lowered for term in ("experiment", "setup", "dataset", "benchmark", "baseline", "metric")):
        hints.add("experiment")
    if any(term in lowered for term in ("result", "finding", "outperform", "score", "accuracy", "bleu")):
        hints.add("result")
    if any(term in lowered for term in ("limitation", "weakness", "future", "assumption", "caveat")):
        hints.add("limitation")
    return hints


def _hashed_embedding(tokens: list[str], dimensions: int = 128) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokens:
        digest = hashlib.md5(token.encode("utf-8")).digest()
        index = int.from_bytes(digest[:4], "little") % dimensions
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[index] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def _cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def _bm25_scores(query_terms: list[str], chunks: list[dict[str, Any]]) -> dict[str, float]:
    query_counts = Counter(query_terms)
    document_frequency: Counter[str] = Counter()
    chunk_tokens: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        chunk_tokens.append(tokens)
        document_frequency.update(set(tokens))

    total_docs = max(len(chunks), 1)
    average_length = sum(len(tokens) for tokens in chunk_tokens) / max(len(chunk_tokens), 1)
    scores: dict[str, float] = {}
    k1 = 1.4
    b = 0.72

    for chunk, tokens in zip(chunks, chunk_tokens):
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        length_norm = k1 * (1 - b + b * (len(tokens) / max(average_length, 1)))
        for term, query_weight in query_counts.items():
            frequency = counts.get(term, 0)
            if not frequency:
                continue
            idf = math.log((total_docs - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5) + 1)
            score += query_weight * idf * ((frequency * (k1 + 1)) / (frequency + length_norm))
        scores[str(chunk.get("chunk_id"))] = score
    return scores


def _is_visual_query(message: str, base_query_terms: list[str]) -> bool:
    """Return True when the query contains visual-cue language."""
    lowered = message.lower()
    # Direct term match
    if any(cue in lowered for cue in _VISUAL_CUE_TERMS):
        return True
    # Token-level match (handles stemming/fragments)
    query_set = set(base_query_terms)
    if query_set.intersection({"figure", "fig", "table", "plot", "chart", "diagram"}):
        return True
    return False


def retrieve_chunks(
    message: str,
    chunks: list[dict[str, Any]],
    limit: int = 4,
    preferred_pages: list[int] | None = None,
) -> list[dict[str, Any]]:
    base_query_terms = tokenize(message)
    query_terms = expand_query_terms(base_query_terms)
    if not base_query_terms:
        return []

    visual_query = _is_visual_query(message, base_query_terms)

    # BM25 is the primary signal because the project evaluation showed it was
    # the most reliable way to keep exact evidence chunks in the top results.
    # Semantic, section, phrase, and page signals are used as light rerankers.
    bm25 = _bm25_scores(base_query_terms, chunks)
    query_vector = _hashed_embedding(query_terms)
    preferred = set(preferred_pages or [])
    section_hints = _section_hints(message)
    bm25_ranked: list[tuple[float, dict[str, Any]]] = []

    for chunk in chunks:
        chunk_id = str(chunk.get("chunk_id"))
        text = chunk.get("text", "")
        tokens = tokenize(text)
        is_fig = chunk.get("is_figure_chunk", False)
        if not tokens:
            # Figure chunks may have very short captions — give them a floor
            # score when a visual query is active so they enter the candidate set.
            if is_fig and visual_query:
                bm25_ranked.append((0.05, chunk))
            continue
        score = bm25.get(chunk_id, 0.0)
        if score > 0 or is_fig:
            bm25_ranked.append((max(score, 0.0), chunk))

    bm25_ranked.sort(key=lambda item: item[0], reverse=True)
    # Expand candidate window to include figure chunks that may have low BM25
    candidate_limit = max(limit * 8, 20)
    if visual_query:
        candidate_limit = max(candidate_limit, limit * 12)
    candidates = bm25_ranked[:candidate_limit]

    reranked: list[tuple[float, dict[str, Any]]] = []
    for rank, (bm25_score, chunk) in enumerate(candidates):
        text = chunk.get("text", "")
        lowered = text.lower()
        tokens = tokenize(text)
        semantic = _cosine(query_vector, _hashed_embedding(tokens)) if tokens else 0.0
        expansion_overlap = len(set(query_terms).intersection(tokens)) / max(len(set(query_terms)), 1)
        exact_overlap = len(set(base_query_terms).intersection(tokens)) / max(len(set(base_query_terms)), 1)

        rerank_score = bm25_score
        rerank_score += semantic * 0.03
        rerank_score += expansion_overlap * 0.05
        rerank_score += exact_overlap * 0.08

        if chunk.get("page") in preferred:
            rerank_score += 1.25
        if chunk.get("chunk_type") in section_hints:
            rerank_score += 0.08
        section_title = str(chunk.get("section_title") or "").lower()
        if any(hint in section_title for hint in section_hints):
            rerank_score += 0.06
        if any(phrase in lowered for phrase in ("we propose", "we introduce", "we present")) and section_hints.intersection({"abstract", "introduction"}):
            rerank_score += 0.08
        if rank < limit:
            rerank_score += 0.05
        if any(term in lowered for term in ("we propose", "we introduce", "we show", "we find")):
            rerank_score += 0.05
        if any(term in lowered for term in ("table", "result", "experiment", "evaluation")) and section_hints.intersection({"result", "experiment"}):
            rerank_score += 0.1
        if any(term in lowered for term in ("limitation", "future work", "however", "although")) and "limitation" in section_hints:
            rerank_score += 0.1

        # ── Visual-cue boosting ──────────────────────────────────────────────
        # When the query contains figure/table language, boost figure chunks
        # so they compete against much-longer text chunks in BM25 ranking.
        if chunk.get("is_figure_chunk") and visual_query:
            rerank_score *= _VISUAL_BOOST
            # Extra bonus if query label tokens match this chunk's label
            # (e.g. query mentions "Figure 1" and chunk label is "Figure 1")
            label_tokens = tokenize(chunk.get("label", ""))
            label_overlap = len(set(base_query_terms).intersection(label_tokens))
            if label_overlap >= 1:
                rerank_score += _CAPTION_MATCH_BONUS * label_overlap

        reranked.append((rerank_score, chunk))

    reranked.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in reranked[:limit]]


def short_quote(chunk: dict[str, Any], query: str, max_length: int = 220) -> str:
    terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
    for sentence in sentences:
        if terms.intersection(tokenize(sentence)):
            return sentence[:max_length].strip()
    return chunk.get("text", "")[:max_length].strip()
