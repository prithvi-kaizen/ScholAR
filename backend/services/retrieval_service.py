"""Tri-Channel Hybrid Retrieval Service for ScholAR.

Features:
- Channel 1 (Lexical): BM25 scoring with camelCase tokenization and scientific term expansion
- Channel 2 (Dense): Semantic embedding similarity via DenseEmbeddingService
- Channel 3 (Visual & Tabular): Modality-directed boosting for figures, charts, and tables
- Reciprocal Rank Fusion (RRF with k=60):
    RRF(d) = sum_m 1 / (60 + rank_m(d))
- Cross-Encoder Reranking: Qwen3-Reranker / Cross-Encoder sequence scoring
- Software-owned provenance preservation
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from collections import Counter
from typing import Any

from backend.services.dense_embedding_service import DenseEmbeddingService
from backend.services.reranker_service import RerankerService

logger = logging.getLogger("scholar.retrieval")

# Visual-cue terms: when any of these appear in the query the retriever
# boosts figure/table chunks so they compete effectively against text chunks.
_VISUAL_CUE_TERMS = frozenset({
    "figure", "fig", "table", "plot", "chart", "graph", "diagram",
    "shown", "depicted", "illustrat", "visuali", "image", "schematic",
    "architecture diagram", "attention map", "heatmap",
})
_VISUAL_BOOST = 2.5
_CAPTION_MATCH_BONUS = 1.8

STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "main", "of", "on", "or", "paper",
    "propose", "proposed", "that", "the", "this", "to", "what", "with",
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

_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_EXPLICIT_FIGURE_RE = re.compile(r"\b(figure|fig\.?|table)\s*\.?\s*(\d+(?:\.\d+)?)\b", re.IGNORECASE)


def tokenize(text: str) -> list[str]:
    """Tokenize text preserving camelCase boundaries and technical tokens."""
    tokens: list[str] = []
    for raw in re.findall(r"[A-Za-z][A-Za-z0-9_-]+", text):
        lowered = raw.lower()
        if lowered not in STOP_WORDS and len(lowered) > 2:
            tokens.append(lowered)
        for part in _CAMEL_BOUNDARY_RE.split(raw):
            part_lower = part.lower()
            if part_lower != lowered and part_lower not in STOP_WORDS and len(part_lower) > 2:
                tokens.append(part_lower)
    return tokens


def expand_query_terms(tokens: list[str]) -> list[str]:
    """Expand query terms with scientific synonyms."""
    expanded = list(tokens)
    for token in tokens:
        expanded.extend(QUERY_EXPANSIONS.get(token, []))
    return expanded


def _normalize_figure_label(label: str) -> str:
    return re.sub(r"\s+", " ", label.strip().lower())


def extract_figure_refs(message: str) -> set[str]:
    """Extract explicit figure/table references (e.g. 'Figure 2', 'Table 3') from query."""
    refs: set[str] = set()
    for kind, number in _EXPLICIT_FIGURE_RE.findall(message):
        kind_norm = "table" if kind.lower().startswith("table") else "figure"
        refs.add(f"{kind_norm} {number}")
    return refs


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
    if bool(_COMPARATIVE_SCALING_PATTERNS.search(lowered)) or any(term in lowered for term in ("result", "finding", "outperform", "score", "accuracy", "bleu", "observation", "evaluation")):
        hints.update({"result", "experiment"})
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


def _bm25_scores(query_terms: list[str], chunks: list[dict[str, Any]]) -> dict[int, float]:
    """Calculate Okapi BM25 scores for chunks."""
    query_counts = Counter(query_terms)
    document_frequency: Counter[str] = Counter()
    chunk_tokens: list[list[str]] = []
    for chunk in chunks:
        tokens = tokenize(chunk.get("text", ""))
        chunk_tokens.append(tokens)
        document_frequency.update(set(tokens))

    total_docs = max(len(chunks), 1)
    average_length = sum(len(tokens) for tokens in chunk_tokens) / max(len(chunk_tokens), 1)
    scores: dict[int, float] = {}
    k1 = 1.4
    b = 0.72

    for index, (chunk, tokens) in enumerate(zip(chunks, chunk_tokens)):
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
        scores[index] = score
    return scores


_IMPLICIT_METRIC_PATTERNS = re.compile(
    r"\b("
    r"bleu|rouge|meteor|cider|ter|wer|cer|perplexity|f1|f1-score|exact match|"
    r"accuracy|top-1|top-5|auc|roc|map|iou|psnr|ssim|fid|is|"
    r"flops|gflops|tflops|latency|throughput|runtime|inference time|training time|"
    r"parameters?|param count|params|memory footprint|vram|gpu hours|speedup|"
    r"learning rate|warmup|warmup steps|weight decay|dropout|attention dropout|"
    r"d_model|d_ff|d_k|d_v|hidden size|embedding dimension|dimension of|"
    r"number of heads|head count|layer count|number of layers|batch size|sequence length|"
    r"outperform|highest score|lowest loss|best performing|ablation|scaling|breakdown|"
    r"performance on|results on|benchmark results|leaderboard|comparison against|versus|"
    r"wmt|glue|superglue|squad|imagenet|cifar|coco|mmlu|gsm8k|human-eval|ptb|penn treebank"
    r")\b",
    re.IGNORECASE,
)

_COMPARATIVE_SCALING_PATTERNS = re.compile(
    r"\b("
    r"outperform|outperforms|crossover|tradeoff|trade-off|threshold|"
    r"input context|context length|context window|scaling|how much better|"
    r"degrade|degradation|longer context|short context|small context|"
    r"surpass|exceeds|compared to|comparison between|versus baseline"
    r")\b",
    re.IGNORECASE,
)

_IMPLICIT_VISUAL_PATTERNS = re.compile(
    r"\b("
    r"architecture|pipeline|workflow|attention distribution|attention weight|attention map|"
    r"loss curve|convergence|training curve|learning curve|tradeoff|scatter|distribution|"
    r"encoder-decoder structure|module breakdown|overview of the model"
    r")\b",
    re.IGNORECASE,
)


def _is_implicit_visual_or_tabular_query(message: str, base_query_terms: list[str]) -> tuple[bool, bool]:
    """Detect explicit and implicit intent for figures or tables."""
    lowered = message.lower()
    has_explicit_vis = any(cue in lowered for cue in _VISUAL_CUE_TERMS) or bool(
        set(base_query_terms).intersection({"figure", "fig", "table", "plot", "chart", "diagram"})
    )
    has_implicit_tab = bool(_IMPLICIT_METRIC_PATTERNS.search(lowered)) or "table" in lowered or "tabular" in lowered
    has_implicit_vis = has_explicit_vis or bool(_IMPLICIT_VISUAL_PATTERNS.search(lowered))
    return has_implicit_vis, has_implicit_tab


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
) -> list[dict[str, Any]]:
    """Fuse multiple ranked lists using Reciprocal Rank Fusion (RRF).

    Formula: RRF(d) = sum_{m} 1 / (k + rank_m(d))
    """
    rrf_scores: dict[str, float] = {}
    chunk_map: dict[str, dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            cid = str(chunk.get("chunk_id") or chunk.get("evidence_id") or id(chunk))
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 / (k + rank))

    # Sort items by accumulated RRF score
    sorted_items = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
    fused: list[dict[str, Any]] = []
    for cid, score in sorted_items:
        c = dict(chunk_map[cid])
        c["rrf_score"] = round(score, 6)
        fused.append(c)

    return fused


def retrieve_chunks(
    message: str,
    chunks: list[dict[str, Any]],
    limit: int = 4,
    preferred_pages: list[int] | None = None,
    paper_id: str = "",
) -> list[dict[str, Any]]:
    """Execute Tri-Channel Hybrid Retrieval (BM25 + Dense + Visual) with RRF & Reranking."""
    base_query_terms = tokenize(message)
    query_terms = expand_query_terms(base_query_terms)
    if not base_query_terms or not chunks:
        return []

    has_visual_intent, has_tabular_intent = _is_implicit_visual_or_tabular_query(message, base_query_terms)

    # 1. Pinned Figure Reference Override
    pinned_chunk: dict[str, Any] | None = None
    figure_refs = extract_figure_refs(message)
    if figure_refs:
        for chunk in chunks:
            if not chunk.get("is_figure_chunk"):
                continue
            if _normalize_figure_label(chunk.get("label", "")) in figure_refs:
                pinned_chunk = chunk
                break

    # 2. Channel 1: BM25 Lexical Ranking
    bm25 = _bm25_scores(base_query_terms, chunks)
    bm25_ranked = []
    for index, chunk in enumerate(chunks):
        score = bm25.get(index, 0.0)
        is_fig = chunk.get("is_figure_chunk", False)
        is_tbl = chunk.get("is_table_chunk", False)
        if score > 0 or (is_fig and (has_visual_intent or (is_tbl and has_tabular_intent))):
            bm25_ranked.append((max(score, 0.08 if (is_tbl and has_tabular_intent) else (0.05 if is_fig else 0.0)), chunk))
    bm25_ranked.sort(key=lambda x: x[0], reverse=True)
    lexical_list = [c for _, c in bm25_ranked]

    # 3. Channel 2: Dense Semantic Embedding Ranking
    p_id = paper_id or chunks[0].get("document_id", "default")
    dense_results = DenseEmbeddingService.search_dense(p_id, message, chunks, top_k=max(limit * 8, 30))
    dense_list = [c for c, _ in dense_results]

    # 4. Cross-Modal Reference Bridging: Detect table/figure citations in top text hits
    bridged_figure_refs: set[str] = set()
    for c in lexical_list[:5] + dense_list[:5]:
        if not c.get("is_figure_chunk"):
            bridged_figure_refs.update(extract_figure_refs(c.get("text", "")))

    # 5. Channel 3: Modality, Implicit Intent & Section Directed Visual Ranking
    modality_ranked = []
    sec_hints = _section_hints(message)
    has_comparative_scaling = bool(_COMPARATIVE_SCALING_PATTERNS.search(message.lower()))
    for chunk in chunks:
        mod_score = 0.0
        is_fig = chunk.get("is_figure_chunk", False)
        is_tbl = chunk.get("is_table_chunk", False)
        norm_label = _normalize_figure_label(chunk.get("label", ""))
        caption_lower = (chunk.get("caption", "") + " " + chunk.get("label", "")).lower()
        sec_str = (str(chunk.get("section", "")) + " " + str(chunk.get("section_title", ""))).lower()

        # High priority boost if explicitly cited by top text chunks
        if norm_label and norm_label in bridged_figure_refs:
            mod_score += 3.0
        elif is_tbl and has_tabular_intent:
            mod_score += 2.5
        elif is_fig and has_comparative_scaling and any(term in caption_lower for term in ("comparison", "scaling", "degrade", "performance", "score", "versus", "vs", "benchmark")):
            mod_score += 3.0
        elif is_fig and has_visual_intent:
            mod_score += 2.0
        elif is_tbl and ("table" in message.lower() or "result" in sec_hints):
            mod_score += 2.0

        if has_comparative_scaling:
            if chunk.get("chunk_type") in ("result", "experiment") or any(term in sec_str for term in ("result", "observation", "evaluation", "discussion", "scaling")):
                mod_score += 2.5

        if chunk.get("page") in set(preferred_pages or []):
            mod_score += 1.0
        if chunk.get("chunk_type") in sec_hints or any(h in sec_str for h in sec_hints):
            mod_score += 0.8

        if mod_score > 0:
            modality_ranked.append((mod_score, chunk))
    modality_ranked.sort(key=lambda x: x[0], reverse=True)
    modality_list = [c for _, c in modality_ranked]

    # 6. Tri-Channel Reciprocal Rank Fusion (k=60)
    channel_lists = [lexical_list, dense_list]
    if modality_list:
        channel_lists.append(modality_list)

    fused_candidates = reciprocal_rank_fusion(channel_lists, k=60)
    top_candidates = fused_candidates[: max(limit * 6, 25)]

    # Tag bridged visual chunks so reranker knows they are linked to top text
    for cand in top_candidates:
        norm_lbl = _normalize_figure_label(cand.get("label", ""))
        if norm_lbl and norm_lbl in bridged_figure_refs:
            cand["is_bridged_visual"] = True

    # 7. Cross-Encoder Reranking
    reranked = RerankerService.rerank(message, top_candidates, top_k=limit)

    # Attach pinned chunk if specified
    if pinned_chunk is not None:
        rest = [c for c in reranked if c.get("chunk_id") != pinned_chunk.get("chunk_id")]
        return [pinned_chunk] + rest[: max(limit - 1, 0)]

    return reranked[:limit]


def short_quote(chunk: dict[str, Any], query: str, max_length: int = 220) -> str:
    """Extract a concise supporting sentence from a chunk."""
    terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
    for sentence in sentences:
        if terms.intersection(tokenize(sentence)):
            return sentence[:max_length].strip()
    return chunk.get("text", "")[:max_length].strip()
