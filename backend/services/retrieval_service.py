"""Hybrid text, crop-image, and full-page visual retrieval for ScholAR.

Features:
- Channel 1 (Lexical): BM25 scoring with camelCase tokenization and scientific term expansion
- Channel 2 (Dense): text semantic similarity via DenseEmbeddingService
- Channel 3 (Modality): query/section-directed boosts for figures, charts, and tables
- Channel 4 (Crop image): always-on paired text/image similarity via VisualEmbeddingService
- Channel 5 (Page image): full-page token-to-patch MaxSim via VisualPageRetrievalService
- Reciprocal Rank Fusion (RRF with k=60), with channel score/rank provenance
- Cross-Encoder Reranking with a bounded image-rank prior
- Software-owned, source-scoped provenance preservation
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from typing import Any

from backend.services.dense_embedding_service import DenseEmbeddingService
from backend.services.document_visual_retrieval_service import (
    DocumentVisualRetrievalService,
)
from backend.services.reranker_service import RerankerService
from backend.services.visual_embedding_service import VisualEmbeddingService
from backend.services.visual_page_retrieval_service import VisualPageRetrievalService

logger = logging.getLogger("scholar.retrieval")

# Visual-cue terms: when any of these appear in the query the retriever
# boosts figure/table chunks so they compete effectively against text chunks.
_VISUAL_CUE_TERMS = frozenset({
    "figure", "fig", "table", "plot", "chart", "graph", "diagram",
    "shown", "depicted", "illustrat", "visuali", "image", "schematic",
    "architecture diagram", "attention map", "heatmap",
})
STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "main", "of", "on", "or", "paper",
    "propose", "proposed", "that", "the", "this", "to", "what", "with",
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


def _identity_part(value: Any) -> str:
    """Normalize an identity component without treating falsey numeric IDs as absent."""
    if value is None:
        return ""
    return str(value).strip()


def evidence_identity(
    chunk: dict[str, Any],
    paper_id: str = "",
) -> tuple[str, str, str]:
    """Return a deterministic, source-scoped identity for an evidence chunk.

    Per-chunk provenance always takes precedence over the caller paper, which is
    only a compatibility fallback for legacy single-paper chunks. Existing
    ``chunk_id`` and ``evidence_id`` fields are preserved and never rewritten.
    """
    source_id = (
        _identity_part(chunk.get("source_paper_id"))
        or _identity_part(chunk.get("document_id"))
        or _identity_part(paper_id)
        or "__unknown_source__"
    )

    chunk_id = _identity_part(chunk.get("chunk_id"))
    if chunk_id:
        return source_id, "chunk_id", chunk_id

    evidence_id = _identity_part(chunk.get("evidence_id"))
    if evidence_id:
        return source_id, "evidence_id", evidence_id

    # Stable, content-addressed fallback for legacy chunks with no identifiers.
    # Restrict the digest to evidence content/provenance fields so transient
    # ranking annotations do not change identity between retrieval channels.
    content_fields = (
        "page",
        "section",
        "section_title",
        "chunk_type",
        "label",
        "caption",
        "is_figure_chunk",
        "is_table_chunk",
        "text",
    )
    payload = {field: chunk.get(field) for field in content_fields}
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
    return source_id, "content_sha256", digest


def evidence_key(
    chunk: dict[str, Any],
    paper_id: str = "",
) -> tuple[str, str, str]:
    """Alias for the shared canonical evidence identity helper."""
    return evidence_identity(chunk, paper_id=paper_id)


_CHANNEL_METADATA_FIELDS = {
    "bm25_score", "bm25_rank", "bm25_has_lexical_overlap",
    "dense_score", "dense_rank",
    "modality_score", "modality_rank",
    "image_embedding_score", "image_embedding_rank",
    "image_embedding_eligible", "image_embedding_threshold",
    "image_embedding_corroborated", "visual_inspection_candidate",
    "page_image_score", "page_image_rank", "page_image_eligible",
    "page_image_threshold", "page_image_corroborated",
}


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    k: int = 60,
    paper_id: str = "",
) -> list[dict[str, Any]]:
    """Fuse ranked channels while preserving their source-specific metadata."""
    rrf_scores: dict[tuple[str, str, str], float] = {}
    chunk_map: dict[tuple[str, str, str], dict[str, Any]] = {}

    for ranked_list in ranked_lists:
        for rank, chunk in enumerate(ranked_list, start=1):
            key = evidence_key(chunk, paper_id=paper_id)
            if key not in chunk_map:
                chunk_map[key] = dict(chunk)
            else:
                current = chunk_map[key]
                merged = dict(current)
                merged.update(chunk)
                for field in _CHANNEL_METADATA_FIELDS:
                    if field in current and field not in chunk:
                        merged[field] = current[field]
                chunk_map[key] = merged
            rrf_scores[key] = rrf_scores.get(key, 0.0) + (1.0 / (k + rank))

    sorted_items = sorted(
        rrf_scores.items(),
        key=lambda item: (-item[1], item[0]),
    )
    fused: list[dict[str, Any]] = []
    for key, score in sorted_items:
        candidate = dict(chunk_map[key])
        candidate["rrf_score"] = round(score, 6)
        fused.append(candidate)
    return fused


def retrieve_chunks(
    message: str,
    chunks: list[dict[str, Any]],
    limit: int = 4,
    preferred_pages: list[int] | None = None,
    paper_id: str = "",
    *,
    include_image_channel: bool = True,
    include_crop_image_channel: bool | None = None,
    include_page_image_channel: bool | None = None,
    visual_page_backend: str | None = None,
    retrieval_metadata: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Run lexical, text-dense, modality, crop-image, and page-image retrieval."""
    base_query_terms = tokenize(message)
    if not base_query_terms or not chunks:
        return []

    has_visual_intent, has_tabular_intent = _is_implicit_visual_or_tabular_query(
        message, base_query_terms
    )

    pinned_chunks: list[dict[str, Any]] = []
    figure_refs = extract_figure_refs(message)
    if figure_refs:
        matched_refs: set[str] = set()
        for chunk in chunks:
            if not chunk.get("is_figure_chunk"):
                continue
            norm_label = _normalize_figure_label(chunk.get("label", ""))
            if norm_label in figure_refs and norm_label not in matched_refs:
                pinned_chunks.append(chunk)
                matched_refs.add(norm_label)

    bm25 = _bm25_scores(base_query_terms, chunks)
    bm25_ranked: list[tuple[float, bool, dict[str, Any]]] = []
    for index, chunk in enumerate(chunks):
        score = bm25.get(index, 0.0)
        is_figure = bool(chunk.get("is_figure_chunk"))
        is_table = bool(chunk.get("is_table_chunk"))
        if score > 0 or (is_figure and (has_visual_intent or (is_table and has_tabular_intent))):
            floor = 0.08 if is_table and has_tabular_intent else (0.05 if is_figure else 0.0)
            bm25_ranked.append((max(score, floor), score > 0, chunk))
    bm25_ranked.sort(key=lambda item: item[0], reverse=True)
    lexical_list = [
        {
            **chunk,
            "bm25_score": round(score, 6),
            "bm25_rank": rank,
            "bm25_has_lexical_overlap": has_lexical_overlap,
        }
        for rank, (score, has_lexical_overlap, chunk) in enumerate(
            bm25_ranked, start=1
        )
    ]

    default_source = paper_id or str(chunks[0].get("document_id") or "default")
    dense_results = DenseEmbeddingService.search_dense(
        default_source,
        message,
        chunks,
        top_k=max(limit * 8, 30),
    )
    dense_list = [
        {**chunk, "dense_score": round(score, 6), "dense_rank": rank}
        for rank, (chunk, score) in enumerate(dense_results, start=1)
    ]

    # Unlike modality heuristics, paired image retrieval is attempted for every
    # query containing image-bearing evidence and never downloads at runtime.
    crop_image_enabled = (
        include_image_channel
        if include_crop_image_channel is None
        else include_crop_image_channel
    )
    page_image_enabled = (
        include_image_channel
        if include_page_image_channel is None
        else include_page_image_channel
    )
    minimum_image_similarity = VisualEmbeddingService.minimum_similarity()
    if crop_image_enabled:
        image_results = VisualEmbeddingService.search_visual(
            default_source,
            message,
            chunks,
            top_k=max(limit * 6, 20),
        )
    else:
        image_results = []
    qualified_image_results = [
        (chunk, score)
        for chunk, score in image_results
        if score >= minimum_image_similarity
    ]
    image_list = [
        {
            **chunk,
            "image_embedding_score": round(score, 6),
            "image_embedding_rank": rank,
            "image_embedding_eligible": True,
            "image_embedding_threshold": minimum_image_similarity,
        }
        for rank, (chunk, score) in enumerate(qualified_image_results, start=1)
    ]

    source_ids = sorted({
        str(chunk.get("source_paper_id") or chunk.get("document_id") or paper_id)
        for chunk in chunks
        if chunk.get("source_paper_id") or chunk.get("document_id") or paper_id
    })
    if page_image_enabled:
        page_search = DocumentVisualRetrievalService.search(
            message,
            source_ids,
            top_k=max(limit * 4, 12),
            backend=visual_page_backend,
        )
    else:
        page_search = DocumentVisualRetrievalService.search(
            "", [], top_k=0, backend=visual_page_backend
        )
    if retrieval_metadata is not None:
        retrieval_metadata["visual_page_retrieval"] = page_search.status.as_dict()
    page_list = [
        {
            **chunk,
            "page_image_score": round(score, 6),
            "page_image_rank": rank,
            "page_image_eligible": True,
            "page_image_threshold": page_search.status.minimum_score,
        }
        for rank, (chunk, score) in enumerate(page_search.hits, start=1)
    ]

    bridged_figure_refs: set[str] = set()
    for candidate in lexical_list[:5] + dense_list[:5]:
        if not candidate.get("is_figure_chunk"):
            bridged_figure_refs.update(extract_figure_refs(candidate.get("text", "")))

    modality_ranked: list[tuple[float, dict[str, Any]]] = []
    section_hints = _section_hints(message)
    comparative = bool(_COMPARATIVE_SCALING_PATTERNS.search(message.lower()))
    preferred_page_set = set(preferred_pages or [])
    for chunk in chunks:
        score = 0.0
        is_figure = bool(chunk.get("is_figure_chunk"))
        is_table = bool(chunk.get("is_table_chunk"))
        normalized_label = _normalize_figure_label(chunk.get("label", ""))
        caption = (str(chunk.get("caption", "")) + " " + str(chunk.get("label", ""))).lower()
        section = (str(chunk.get("section", "")) + " " + str(chunk.get("section_title", ""))).lower()

        if normalized_label and normalized_label in bridged_figure_refs:
            score += 3.0
        elif is_table and has_tabular_intent:
            score += 2.5
        elif is_figure and comparative and any(
            term in caption
            for term in ("comparison", "scaling", "degrade", "performance", "score", "versus", "vs", "benchmark")
        ):
            score += 3.0
        elif is_figure and has_visual_intent:
            score += 2.0
        elif is_table and ("table" in message.lower() or "result" in section_hints):
            score += 2.0

        if comparative and (
            chunk.get("chunk_type") in ("result", "experiment")
            or any(term in section for term in ("result", "observation", "evaluation", "discussion", "scaling"))
        ):
            score += 2.5
        if chunk.get("page") in preferred_page_set:
            score += 1.0
        if chunk.get("chunk_type") in section_hints or any(hint in section for hint in section_hints):
            score += 0.8
        if score > 0:
            modality_ranked.append((score, chunk))
    modality_ranked.sort(key=lambda item: item[0], reverse=True)
    modality_list = [
        {**chunk, "modality_score": round(score, 6), "modality_rank": rank}
        for rank, (score, chunk) in enumerate(modality_ranked, start=1)
    ]

    channel_lists = [lexical_list, dense_list]
    if modality_list:
        channel_lists.append(modality_list)
    if image_list:
        channel_lists.append(image_list)
    if page_list:
        channel_lists.append(page_list)

    fused_candidates = reciprocal_rank_fusion(channel_lists, k=60, paper_id=paper_id)
    top_candidates = fused_candidates[: max(limit * 6, 25)]
    for candidate in top_candidates:
        normalized_label = _normalize_figure_label(candidate.get("label", ""))
        if normalized_label and normalized_label in bridged_figure_refs:
            candidate["is_bridged_visual"] = True
        candidate["image_embedding_corroborated"] = bool(
            candidate.get("image_embedding_eligible") is True
            and (
                candidate.get("bm25_has_lexical_overlap") is True
                or candidate.get("is_bridged_visual") is True
                or (
                    normalized_label
                    and normalized_label in figure_refs
                )
            )
        )
        if candidate.get("is_page_visual_chunk"):
            source_id = str(candidate.get("source_paper_id") or candidate.get("document_id") or paper_id)
            page = candidate.get("page")
            candidate["page_image_corroborated"] = any(
                str(item.get("source_paper_id") or item.get("document_id") or paper_id) == source_id
                and item.get("page") == page
                for item in lexical_list[:5] + dense_list[:5]
            )

    reranked = RerankerService.rerank(message, top_candidates, top_k=limit)

    # A score-floor-qualified, uncorroborated image is retained only as a
    # bounded inspection candidate. It receives no reranker prior and does not
    # establish evidence sufficiency.
    inspection_candidate = next(
        (
            candidate for candidate in top_candidates
            if candidate.get("image_embedding_eligible") is True
        ),
        None,
    )
    if inspection_candidate is not None and limit > 0:
        inspection_key = evidence_key(inspection_candidate, paper_id=paper_id)
        if all(
            evidence_key(candidate, paper_id=paper_id) != inspection_key
            for candidate in reranked
        ):
            retained = dict(inspection_candidate)
            retained["visual_inspection_candidate"] = True
            reranked = reranked[: max(limit - 1, 0)] + [retained]

    page_inspection_candidate = next(
        (
            candidate for candidate in top_candidates
            if candidate.get("page_image_eligible") is True
        ),
        None,
    )
    if page_inspection_candidate is not None and limit > 0:
        inspection_key = evidence_key(page_inspection_candidate, paper_id=paper_id)
        if all(
            evidence_key(candidate, paper_id=paper_id) != inspection_key
            for candidate in reranked
        ):
            retained = dict(page_inspection_candidate)
            retained["visual_inspection_candidate"] = True
            reranked = reranked[: max(limit - 1, 0)] + [retained]

    if pinned_chunks:
        pinned_keys = {evidence_key(chunk, paper_id=paper_id) for chunk in pinned_chunks}
        annotated_pinned_list: list[dict[str, Any]] = []
        for p_chunk in pinned_chunks:
            p_key = evidence_key(p_chunk, paper_id=paper_id)
            annotated = next(
                (
                    candidate
                    for candidate in reranked + top_candidates
                    if evidence_key(candidate, paper_id=paper_id) == p_key
                ),
                dict(p_chunk),
            )
            annotated_pinned_list.append(annotated)
        rest = [
            chunk
            for chunk in reranked
            if evidence_key(chunk, paper_id=paper_id) not in pinned_keys
        ]
        remaining_slots = max(limit - len(annotated_pinned_list), 0)
        return annotated_pinned_list + rest[:remaining_slots]

    return reranked[:limit]


def short_quote(chunk: dict[str, Any], query: str, max_length: int = 220) -> str:
    """Extract a concise supporting sentence from a chunk."""
    terms = set(tokenize(query))
    sentences = re.split(r"(?<=[.!?])\s+", chunk.get("text", ""))
    for sentence in sentences:
        if terms.intersection(tokenize(sentence)):
            return sentence[:max_length].strip()
    return chunk.get("text", "")[:max_length].strip()
