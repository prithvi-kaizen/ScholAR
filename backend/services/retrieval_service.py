from __future__ import annotations

import math
import re
from collections import Counter
from typing import Any


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


def tokenize(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
        if token not in STOP_WORDS and len(token) > 2
    ]


def retrieve_chunks(message: str, chunks: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    query_terms = tokenize(message)
    if not query_terms:
        return []

    query_counts = Counter(query_terms)
    document_frequency: Counter[str] = Counter()
    chunk_tokens: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        chunk_tokens.append(tokens)
        document_frequency.update(set(tokens))

    scored: list[tuple[float, dict[str, Any]]] = []
    total_docs = max(len(chunks), 1)
    average_length = sum(len(tokens) for tokens in chunk_tokens) / max(len(chunk_tokens), 1)

    for chunk, tokens in zip(chunks, chunk_tokens):
        if not tokens:
            continue
        counts = Counter(tokens)
        score = 0.0
        for term, query_weight in query_counts.items():
            if term not in counts:
                continue
            idf = math.log((total_docs + 1) / (document_frequency[term] + 0.5)) + 1
            length_norm = 0.75 + 0.25 * (len(tokens) / max(average_length, 1))
            score += query_weight * idf * (counts[term] / length_norm)
        if score > 0:
            scored.append((score, chunk))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:limit]]


def short_quote(chunk: dict[str, Any], query: str, max_length: int = 220) -> str:
    terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
    for sentence in sentences:
        if terms.intersection(tokenize(sentence)):
            return sentence[:max_length].strip()
    return chunk.get("text", "")[:max_length].strip()
