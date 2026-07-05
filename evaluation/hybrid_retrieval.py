"""
hybrid_retrieval.py — Hybrid BM25 + Dense + RRF Retrieval
==========================================================
Implements the industry-standard hybrid retrieval pipeline for AAAI-27:

  Score_hybrid(q, d) = RRF(rank_BM25(q,d), rank_dense(q,d))
  RRF score = 1/(k + rank_BM25) + 1/(k + rank_dense), k=60

Dense retrieval uses all-MiniLM-L6-v2 embeddings loaded from HF cache.
Falls back to BM25-only if the embedder is unavailable.

References:
- RRF: Cormack et al., SIGIR 2009
- Hybrid RAG: Chen et al., 2024; BEIR benchmarks (Thakur et al., 2021)
"""
from __future__ import annotations

import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import numpy as np

# ── try to load the dense embedder ────────────────────────────────────
_EMBEDDER = None
_EMBEDDER_LOADED = False

def _get_embedder():
    global _EMBEDDER, _EMBEDDER_LOADED
    if _EMBEDDER_LOADED:
        return _EMBEDDER
    try:
        eval_dir = Path(__file__).resolve().parent
        if str(eval_dir) not in sys.path:
            sys.path.insert(0, str(eval_dir))
        from embedder import LocalEmbedder
        _EMBEDDER = LocalEmbedder()
        print(f"  [Embedder] Loaded MiniLM ({_EMBEDDER._loaded} weight tensors)")
    except Exception as exc:
        print(f"  [Embedder] Not available ({exc}); using BM25-only")
        _EMBEDDER = None
    _EMBEDDER_LOADED = True
    return _EMBEDDER


# ── BM25 (unchanged from production service) ──────────────────────────
import re

_STOP_WORDS = {
    "a","an","and","are","as","at","be","by","for","from","how",
    "in","is","it","main","of","on","or","paper","propose","proposed",
    "that","the","this","to","what","with",
}

def _tokenize(text: str) -> list[str]:
    return [
        t for t in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text.lower())
        if t not in _STOP_WORDS and len(t) > 2
    ]


def _bm25_ranked(query: str, chunks: list[dict[str, Any]],
                 k1: float = 1.4, b: float = 0.72) -> list[tuple[float, int]]:
    """Returns [(score, chunk_index)] sorted descending."""
    query_terms = Counter(_tokenize(query))
    chunk_tokens = [_tokenize(c.get("text", "")) for c in chunks]
    df: Counter = Counter()
    for toks in chunk_tokens:
        df.update(set(toks))
    N   = max(len(chunks), 1)
    avg = sum(len(t) for t in chunk_tokens) / max(len(chunk_tokens), 1)

    scored: list[tuple[float, int]] = []
    for i, (chunk, toks) in enumerate(zip(chunks, chunk_tokens)):
        if not toks:
            continue
        tf  = Counter(toks)
        dl  = len(toks)
        s   = 0.0
        for term, qw in query_terms.items():
            f = tf.get(term, 0)
            if not f:
                continue
            idf = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            s  += qw * idf * (f * (k1 + 1)) / (f + k1 * (1 - b + b * dl / avg))
        scored.append((s, i))
    scored.sort(key=lambda x: -x[0])
    return scored


# ── Reciprocal Rank Fusion ─────────────────────────────────────────────

def _rrf(rank_lists: list[list[int]], k: int = 60) -> list[tuple[int, float]]:
    """
    Combine multiple rank lists via RRF.
    rank_lists: each is a list of chunk indices in rank order (best first).
    Returns [(chunk_idx, rrf_score)] sorted by score descending.
    """
    scores: dict[int, float] = {}
    for ranks in rank_lists:
        for rank_pos, chunk_idx in enumerate(ranks, start=1):
            idx = int(chunk_idx)
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank_pos)
    # items() returns (key=idx, val=score) — sort descending by score
    return sorted(scores.items(), key=lambda x: -x[1])


# ── Dense ranking ──────────────────────────────────────────────────────

def _dense_ranked(query: str, chunks: list[dict[str, Any]],
                  embedder) -> list[tuple[float, int]]:
    """Returns [(cosine_score, chunk_index)] sorted descending."""
    texts = [c.get("text", "")[:512] for c in chunks]
    all_texts = [query] + texts
    embs = embedder.encode(all_texts, batch_size=64)

    q_emb  = embs[0:1]       # (1, D)
    c_embs = embs[1:]         # (N, D)
    sims   = (c_embs @ q_emb.T).reshape(-1)  # (N,) — explicit reshape

    scored = [(float(sims[i]), int(i)) for i in range(len(chunks))]
    scored.sort(key=lambda x: -x[0])
    return scored


# ── Public API ─────────────────────────────────────────────────────────

def retrieve_hybrid(query: str,
                    chunks: list[dict[str, Any]],
                    limit: int = 5,
                    preferred_pages: Optional[list[int]] = None) -> list[dict[str, Any]]:
    """
    Hybrid BM25 + Dense retrieval via Reciprocal Rank Fusion.
    Applies page-hint boost post-fusion.
    Falls back to BM25-only if embedder unavailable.
    """
    preferred = set(preferred_pages or [])
    embedder  = _get_embedder()

    # Stage 1: BM25 → [(score, chunk_idx)]
    bm25_results = _bm25_ranked(query, chunks)
    bm25_order   = [int(i) for _, i in bm25_results]

    if embedder is not None:
        # Stage 2: Dense → [(score, chunk_idx)]
        dense_results = _dense_ranked(query, chunks, embedder)
        dense_order   = [int(i) for _, i in dense_results]

        # Stage 3: RRF fusion → [(chunk_idx, rrf_score)]
        rrf_items = _rrf([bm25_order, dense_order])
    else:
        # BM25-only fallback — normalise to same (chunk_idx, score) format
        rrf_items = [(int(i), float(s)) for s, i in bm25_results]

    # Stage 4: page-hint boost; rrf_items is [(chunk_idx:int, score:float)]
    boosted: list[tuple[float, int]] = []
    for chunk_idx, score in rrf_items:
        idx = int(chunk_idx)
        if chunks[idx].get("page") in preferred:
            score += 0.05
        boosted.append((score, idx))
    boosted.sort(key=lambda x: -x[0])

    return [chunks[i] for _, i in boosted[:limit]]


def retrieve_dense_only(query: str,
                        chunks: list[dict[str, Any]],
                        limit: int = 5,
                        preferred_pages: Optional[list[int]] = None) -> list[dict[str, Any]]:
    """
    Naive single-pass dense retrieval: MiniLM cosine similarity only.
    No BM25, no RRF fusion, no page-hint boost. Returns [] if the
    embedder is unavailable.
    """
    embedder = _get_embedder()
    if embedder is None:
        return []
    dense_results = _dense_ranked(query, chunks, embedder)
    return [chunks[i] for _, i in dense_results[:limit]]
