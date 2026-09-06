"""Single production/evaluation orchestration path for grounded ScholAR answers."""

from __future__ import annotations

import asyncio
import hashlib
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx

from backend.schemas.answer_trace import (
    ANSWER_PROMPT_VERSION,
    AbstentionTrace,
    AnswerPipelineRequest,
    AnswerTrace,
    CitationOrigin,
    CitationTrace,
    EvidenceIdentity,
    ExecutionPolicy,
    GenerationMode,
    InterventionExecutionTrace,
    LocalGenerationMetadata,
    PipelineStatus,
    PromptEvidenceTrace,
    RetrievalHitTrace,
    RetrievalQueryChannelTrace,
    RunIdentity,
    StageTiming,
    VerificationTrace,
)
from backend.schemas.capabilities import ModelRegistry
from backend.schemas.numeric_plan import NumericOp
from backend.schemas.reasoning import TargetModality
from backend.services.budgeting_service import BudgetingService
from backend.services.chunking_service import chunk_figures
from backend.services.evidence_graph_service import EvidenceGraphService
from backend.services.ollama_service import OLLAMA_MODEL, generate_result, ollama_available
from backend.services.pdf_service import paper_dir, read_json, safe_paper_id
from backend.services.question_analyzer import QuestionAnalyzer
from backend.services.retrieval_service import (
    _normalize_figure_label,
    evidence_identity,
    evidence_key,
    extract_figure_refs,
    extract_page_hints,
    retrieve_chunks,
    tokenize,
)
from backend.services.routing_service import QuestionRouter, QuestionRouteType
from backend.services.table_arithmetic_service import TableArithmeticService
from backend.services.telemetry_service import TelemetryService
from backend.services.verifier_service import ClaimVerifierService
from backend.services.visual_embedding_service import VisualEmbeddingService
from backend.services.vision_service import (
    answer_with_custom_snippet,
    answer_with_multimodal_evidence,
    is_uninformative_visual_answer,
)


def _git_identity() -> tuple[str | None, bool | None]:
    root = Path(__file__).resolve().parents[2]
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        ).stdout.strip())
        return revision or None, dirty
    except Exception:
        return None, None


def _paper_paths(paper_id: str) -> tuple[Path, Path, Path, Path]:
    directory = paper_dir(safe_paper_id(paper_id))
    paths = (
        directory / "metadata.json",
        directory / "pages.json",
        directory / "chunks.json",
        directory / "paper.pdf",
    )
    if not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"Paper {paper_id!r} has not been completely prepared")
    return paths


def _merge_source_figure_chunks(
    source_paper_id: str,
    source_chunks: list[dict[str, Any]],
    figures: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Enrich canonical visual chunks and append only missing legacy figures."""
    merged_chunks = [
        {
            **chunk,
            "source_paper_id": chunk.get("source_paper_id") or source_paper_id,
        }
        for chunk in source_chunks
    ]
    indexes: dict[tuple[str, str], int] = {}
    for index, chunk in enumerate(merged_chunks):
        image_file = str(chunk.get("image_file") or "")
        figure_id = str(chunk.get("figure_id") or "")
        if image_file:
            indexes[("image_file", image_file)] = index
        if figure_id:
            indexes[("figure_id", figure_id)] = index

    for figure_chunk in chunk_figures(figures, source_paper_id=source_paper_id):
        image_file = str(figure_chunk.get("image_file") or "")
        figure_id = str(figure_chunk.get("figure_id") or "")
        index = indexes.get(("image_file", image_file)) if image_file else None
        if index is None and figure_id:
            index = indexes.get(("figure_id", figure_id))
        if index is None:
            figure_chunk["document_id"] = source_paper_id
            merged_chunks.append(figure_chunk)
            index = len(merged_chunks) - 1
        else:
            canonical = merged_chunks[index]
            identity = {
                key: canonical.get(key)
                for key in ("chunk_id", "evidence_id", "document_id", "source_paper_id")
                if canonical.get(key) is not None
            }
            enriched = dict(canonical)
            enriched.update({
                key: value
                for key, value in figure_chunk.items()
                if value not in (None, "", [])
            })
            enriched.update(identity)
            merged_chunks[index] = enriched
        if image_file:
            indexes[("image_file", image_file)] = index
        if figure_id:
            indexes[("figure_id", figure_id)] = index
    return merged_chunks


def _important_sentences(text: str) -> list[str]:
    banned = ("table 1:", "figure", "copyright", "provided proper attribution")
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if len(item.strip()) > 45
    ]
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in banned):
            continue
        score = sum(
            2
            for term in (
                "we introduce", "we propose", "we show", "we find", "results",
                "achieves", "outperforms", "significant", "accuracy", "benchmark",
                "evaluation", "dataset", "method", "framework", "limitation",
            )
            if term in lowered
        )
        if any(char.isdigit() for char in sentence):
            score += 1
        ranked.append((score, sentence[:420]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [sentence for score, sentence in ranked if score > 0][:8] or [sentence[:420] for sentence in sentences[:6]]


def _extractive_tutor_answer(question: str, selected: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    from backend.services.mlr_synthesis_service import MLRSynthesisService
    text_chunks = [c for c in selected if not c.get("is_figure_chunk")]
    fig_chunks = [c for c in selected if c.get("is_figure_chunk")]
    mlr_res = MLRSynthesisService.synthesize_extractive_mlr(question, text_chunks or selected, fig_chunks)
    return mlr_res.get("answer", ""), mlr_res.get("citations", [])


def _citation_candidates(text: str) -> list[str]:
    sentences = [
        item.strip()
        for item in re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
        if len(item.strip()) > 55
    ]
    return [sentence[:500] for sentence in sentences]


def _build_evidence_items(
    question: str,
    chunks: list[dict[str, Any]],
    limit: int = 8,
) -> list[dict[str, Any]]:
    question_terms = set(tokenize(question))
    banned = (
        "author name redacted", "copyright", "provided proper attribution", "equal contribution",
        "arxiv:", "preprint.", "facebook ai research", "university college london",
        "new york university",
    )
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for chunk_index, chunk in enumerate(chunks):
        for sentence_index, sentence in enumerate(_citation_candidates(str(chunk.get("text", "")))):
            lowered = sentence.lower()
            if any(term in lowered for term in banned):
                continue
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            score = len(question_terms.intersection(sentence_terms)) * 2.0
            if any(term in lowered for term in ("we propose", "we introduce", "we present", "we show", "we find")):
                score += 4.0
            if any(term in lowered for term in ("result", "achieve", "outperform", "benchmark", "dataset", "experiment", "limitation")):
                score += 2.0
            if any(char.isdigit() for char in sentence):
                score += 0.4
            score += max(0, 2.5 - chunk_index * 0.35)
            score += max(0, 0.8 - sentence_index * 0.05)
            if score <= 1.2:
                continue
            scored.append((score, chunk_index, {
                "page": chunk.get("page"),
                "chunk_id": chunk.get("chunk_id"),
                "section_title": chunk.get("section_title") or "Paper",
                "chunk_type": chunk.get("chunk_type") or "body",
                "quote": sentence[:520],
                "source_paper_id": chunk.get("source_paper_id") or chunk.get("document_id"),
                "document_id": chunk.get("document_id"),
                "source_evidence_id": chunk.get("evidence_id"),
            }))
    scored.sort(key=lambda item: item[0], reverse=True)
    evidence: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    for _, _, item in scored:
        quote_key = re.sub(r"\W+", " ", str(item["quote"]).lower())[:120]
        if quote_key in seen_quotes:
            continue
        seen_quotes.add(quote_key)
        item["evidence_id"] = f"E{len(evidence) + 1}"
        evidence.append(item)
        if len(evidence) >= limit:
            break
    return evidence


def _format_evidence_context(
    evidence_items: list[dict[str, Any]],
    secondary_meta: dict[str, dict[str, Any]] | None = None,
) -> str:
    if not evidence_items:
        return "No relevant paper evidence was retrieved."
    lines: list[str] = []
    for item in evidence_items:
        source_id = item.get("source_paper_id")
        if source_id and secondary_meta and source_id in secondary_meta:
            short_title = (secondary_meta[source_id].get("title") or source_id)[:40]
            source_label = f"ref:{short_title}"
        else:
            source_label = "anchor"
        lines.append(
            f"[{item['evidence_id']} | {source_label} | p. {item.get('page')} | {item.get('chunk_id')}]\n{item.get('quote')}"
        )
    return "\n\n".join(lines)


def _normalize_evidence_citations(
    answer: str,
    evidence_items: list[dict[str, Any]],
) -> tuple[str, list[dict[str, Any]]]:
    evidence_by_id = {str(item["evidence_id"]).upper(): item for item in evidence_items}
    used_ids: list[str] = []
    answer_without_direct_pages = re.sub(r"\[p\.\s*\d+\]", "", answer, flags=re.IGNORECASE)

    def replace_evidence_group(match: re.Match[str]) -> str:
        refs: list[str] = []
        for number in re.findall(r"\d+", match.group(0)):
            evidence_id = f"E{number}".upper()
            if evidence_id not in evidence_by_id:
                continue
            if evidence_id not in used_ids:
                used_ids.append(evidence_id)
            refs.append(f"[{used_ids.index(evidence_id) + 1}]")
        return "".join(refs)

    normalized = re.sub(
        r"\[\s*E\s*\d+(?:\s*(?:,|;|/|&|and)\s*E?\s*\d+)*\s*\]",
        replace_evidence_group,
        answer_without_direct_pages,
        flags=re.IGNORECASE,
    )
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    citations = [
        {
            "ref_id": index,
            "page": evidence_by_id[evidence_id].get("page"),
            "chunk_id": evidence_by_id[evidence_id].get("chunk_id"),
            "section_title": evidence_by_id[evidence_id].get("section_title"),
            "chunk_type": evidence_by_id[evidence_id].get("chunk_type"),
            "quote": evidence_by_id[evidence_id].get("quote"),
            "source_paper_id": evidence_by_id[evidence_id].get("source_paper_id"),
            "document_id": evidence_by_id[evidence_id].get("document_id"),
            "source_evidence_id": evidence_by_id[evidence_id].get("source_evidence_id"),
        }
        for index, evidence_id in enumerate(used_ids[:5], start=1)
    ]
    return normalized.strip(), citations


def _build_answer_citations(
    answer: str,
    question: str,
    chunks: list[dict[str, Any]],
    limit: int = 4,
) -> list[dict[str, Any]]:
    answer_terms = set(tokenize(answer))
    question_terms = set(tokenize(question))
    page_hints = set(extract_page_hints(question))
    scored: list[tuple[float, dict[str, Any]]] = []
    banned = ("author name redacted", "abstract", "when models know better", "april 3, 2026")
    for chunk in chunks:
        page = chunk.get("page")
        for sentence in _citation_candidates(str(chunk.get("text", ""))):
            lowered = sentence.lower()
            if any(term in lowered for term in banned):
                continue
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            score = len(answer_terms.intersection(sentence_terms)) * 2.0 + len(question_terms.intersection(sentence_terms))
            if page in page_hints:
                score += 6.0
            if any(char.isdigit() for char in sentence):
                score += 0.5
            if score > 1:
                scored.append((score, {
                    "page": page,
                    "chunk_id": chunk.get("chunk_id"),
                    "quote": sentence,
                    "source_paper_id": chunk.get("source_paper_id") or chunk.get("document_id"),
                    "document_id": chunk.get("document_id"),
                    "source_evidence_id": chunk.get("evidence_id"),
                }))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {"ref_id": index, **citation}
        for index, (_, citation) in enumerate(scored[:limit], start=1)
    ]


def _identity_model(chunk: dict[str, Any], paper_id: str) -> EvidenceIdentity:
    source_id, kind, local_id = evidence_identity(chunk, paper_id=paper_id)
    return EvidenceIdentity(source_id=source_id, local_id_kind=kind, local_id=local_id)


def _identity_string(chunk: dict[str, Any], paper_id: str) -> str:
    return _identity_model(chunk, paper_id).global_id


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    return None


def _safe_rank(value: Any) -> int | None:
    return value if type(value) is int and value > 0 else None


def _query_channel_trace(
    chunk: dict[str, Any],
    retrieval_query: str,
    subquery_id: str,
) -> RetrievalQueryChannelTrace:
    return RetrievalQueryChannelTrace(
        retrieval_query=retrieval_query,
        subquery_id=subquery_id,
        bm25_score=_safe_float(chunk.get("bm25_score")),
        bm25_rank=_safe_rank(chunk.get("bm25_rank")),
        dense_score=_safe_float(chunk.get("dense_score")),
        dense_rank=_safe_rank(chunk.get("dense_rank")),
        modality_score=_safe_float(chunk.get("modality_score")),
        modality_rank=_safe_rank(chunk.get("modality_rank")),
        image_embedding_score=_safe_float(chunk.get("image_embedding_score")),
        image_embedding_rank=_safe_rank(chunk.get("image_embedding_rank")),
        image_embedding_eligible=chunk.get("image_embedding_eligible") is True,
        image_embedding_threshold=_safe_float(chunk.get("image_embedding_threshold")),
        image_embedding_corroborated=chunk.get("image_embedding_corroborated") is True,
        page_image_score=_safe_float(chunk.get("page_image_score")),
        page_image_rank=_safe_rank(chunk.get("page_image_rank")),
        page_image_eligible=chunk.get("page_image_eligible") is True,
        page_image_threshold=_safe_float(chunk.get("page_image_threshold")),
        page_image_corroborated=chunk.get("page_image_corroborated") is True,
        visual_retrieval_backend=(
            str(chunk.get("visual_retrieval_backend"))
            if chunk.get("visual_retrieval_backend") else None
        ),
        visual_retrieval_model=(
            str(chunk.get("visual_retrieval_model"))
            if chunk.get("visual_retrieval_model") else None
        ),
        visual_inspection_candidate=chunk.get("visual_inspection_candidate") is True,
        rrf_score=_safe_float(chunk.get("rrf_score")),
        rerank_score=_safe_float(chunk.get("rerank_score")),
    )


def _hit_trace(
    chunk: dict[str, Any],
    paper_id: str,
    rank: int | None,
    queries: list[str],
    subquery_ids: list[str],
    query_channel_results: list[RetrievalQueryChannelTrace] | None = None,
) -> RetrievalHitTrace:
    modality = (
        "page_visual"
        if chunk.get("is_page_visual_chunk")
        else "table"
        if chunk.get("is_table_chunk")
        else "figure"
        if chunk.get("is_figure_chunk")
        else "text"
    )
    return RetrievalHitTrace(
        identity=_identity_model(chunk, paper_id),
        retrieval_queries=queries,
        subquery_ids=subquery_ids,
        query_channel_results=query_channel_results or [],
        final_rank=rank,
        page=chunk.get("page") if isinstance(chunk.get("page"), int) else None,
        section=str(chunk.get("section_title") or chunk.get("section") or ""),
        modality=modality,
        bm25_score=_safe_float(chunk.get("bm25_score")),
        bm25_rank=_safe_rank(chunk.get("bm25_rank")),
        dense_score=_safe_float(chunk.get("dense_score")),
        dense_rank=_safe_rank(chunk.get("dense_rank")),
        modality_score=_safe_float(chunk.get("modality_score")),
        modality_rank=_safe_rank(chunk.get("modality_rank")),
        image_embedding_score=_safe_float(chunk.get("image_embedding_score")),
        image_embedding_rank=_safe_rank(chunk.get("image_embedding_rank")),
        image_embedding_eligible=chunk.get("image_embedding_eligible") is True,
        image_embedding_threshold=_safe_float(chunk.get("image_embedding_threshold")),
        image_embedding_corroborated=chunk.get("image_embedding_corroborated") is True,
        page_image_score=_safe_float(chunk.get("page_image_score")),
        page_image_rank=_safe_rank(chunk.get("page_image_rank")),
        page_image_eligible=chunk.get("page_image_eligible") is True,
        page_image_threshold=_safe_float(chunk.get("page_image_threshold")),
        page_image_corroborated=chunk.get("page_image_corroborated") is True,
        visual_retrieval_backend=(
            str(chunk.get("visual_retrieval_backend"))
            if chunk.get("visual_retrieval_backend") else None
        ),
        visual_retrieval_model=(
            str(chunk.get("visual_retrieval_model"))
            if chunk.get("visual_retrieval_model") else None
        ),
        candidate_regions=(
            list(chunk.get("candidate_regions") or [])
            if isinstance(chunk.get("candidate_regions"), list) else []
        ),
        visual_inspection_candidate=chunk.get("visual_inspection_candidate") is True,
        rrf_score=_safe_float(chunk.get("rrf_score")),
        rerank_score=_safe_float(chunk.get("rerank_score")),
        text_preview=str(chunk.get("text") or chunk.get("caption") or "")[:240],
    )


def _citation_trace(
    citation: dict[str, Any],
    paper_id: str,
    origin: CitationOrigin,
) -> CitationTrace:
    known = {
        "ref_id", "page", "chunk_id", "section_title", "chunk_type", "quote",
        "source_paper_id", "document_id", "source_evidence_id", "verification", "confidence",
        "repair_origin",
    }
    if citation.get("repair_origin") == CitationOrigin.REMAPPED.value:
        origin = CitationOrigin.REMAPPED
    identity_payload = {
        "source_paper_id": citation.get("source_paper_id"),
        "document_id": citation.get("document_id"),
        "chunk_id": citation.get("chunk_id"),
        "evidence_id": citation.get("source_evidence_id"),
        "page": citation.get("page"),
        "text": citation.get("quote"),
    }
    return CitationTrace(
        ref_id=int(citation.get("ref_id") or 0),
        page=citation.get("page") if isinstance(citation.get("page"), int) else None,
        chunk_id=str(citation.get("chunk_id") or ""),
        section_title=citation.get("section_title"),
        chunk_type=citation.get("chunk_type"),
        quote=str(citation.get("quote") or ""),
        source_paper_id=citation.get("source_paper_id"),
        document_id=citation.get("document_id"),
        source_evidence_id=citation.get("source_evidence_id"),
        verification=citation.get("verification"),
        confidence=_safe_float(citation.get("confidence")),
        origin=origin,
        identity=_identity_model(identity_payload, paper_id),
        extra={key: value for key, value in citation.items() if key not in known},
    )


def _build_prompt(
    request: AnswerPipelineRequest,
    metadata: dict[str, Any],
    evidence_items: list[dict[str, Any]],
    secondary_meta: dict[str, dict[str, Any]],
    route_type: QuestionRouteType,
    numeric_result: Any | None = None,
    effective_query: str | None = None,
) -> str:
    recent_history = "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:180]}"
        for item in request.history[-6:]
        if isinstance(item, dict)
    )
    route_instructions = ""
    if route_type == QuestionRouteType.CODE_ALGORITHM:
        route_instructions = """
Special Algorithm / Code Instructions:
- Provide a clean, formatted code or pseudocode block.
- State inputs, hyperparameters, tensor dimensions, execution steps, and complexity.
"""
    elif route_type == QuestionRouteType.TABLE_NUMERIC:
        route_instructions = """
Special Tabular / Numeric Instructions:
- Reconstruct available tabular data in a clean Markdown table.
- Report metric deltas only when the supplied evidence supports them.
"""

    numeric_context = ""
    if numeric_result and getattr(numeric_result, "is_exact", False):
        op_val = getattr(numeric_result.operation, "value", str(numeric_result.operation))
        numeric_context = f"""
Deterministic Calculation Result (Precomputed exact arithmetic):
- Operation: {op_val}
- Computed Value: {numeric_result.formatted_value}
- Exact Statement: {numeric_result.formatted_statement}
- Instruction: Incorporate this exact numerical result into your quantitative explanation. Do not alter this calculated value.
"""

    query_line = request.query[:900]
    if effective_query and effective_query.strip() != request.query.strip():
        query_line = f"{request.query[:900]} [Context: {effective_query[:900]}]"

    return f"""
You are ScholAR, a rigorous research paper tutor. Answer strictly from the supplied local paper evidence. If it is insufficient, say what is missing instead of guessing.

Paper metadata:
Title: {metadata.get('title')}
Authors: {', '.join(metadata.get('authors', []))}
Abstract: {str(metadata.get('summary') or '')[:350]}

Recent conversation:
{recent_history or 'No previous conversation.'}

Response requirements:
- Give a precise study answer grounded only in the evidence adhering to EACL reviewer standards.
- Multi-Level Reasoning: If the question requires conceptual, comparative, or mechanistic analysis, structure the response across:
  1. Problem Context & Direct Resolution
  2. Mathematical or Architectural Derivation
  3. Empirical Verification & Quantitative Evidence
- Use bold section labels rather than Markdown headings.
- Cite claims only with supplied evidence IDs such as [E1].
- Never invent page citations or evidence IDs.
- Keep citations close to the claims they support.
- If evidence is insufficient, abstain plainly without attaching a citation to the disclaimer.
- Answer directly in 180 to 450 words.
{route_instructions}
{numeric_context}

Paper evidence:
{_format_evidence_context(evidence_items, secondary_meta)}

Question: {query_line}
""".strip()


def resolve_conversational_query(query: str, history: list[dict[str, Any]] | None) -> str:
    """Resolve follow-up referents and pronouns in multi-turn conversation.

    Ensures that queries like 'Does that hold for the larger model?' or 'How does it compare?'
    retain the referent entity from previous conversation turns for retrieval and routing.
    """
    if not history:
        return query.strip()

    clean_q = query.strip()
    referent_pattern = re.compile(
        r"\b(it|its|they|them|their|this|that|these|those)\b|\bthe\s+(model|method|approach|architecture|authors?)\b(?!\s+(of|in|for|from|described|proposed|named))\b",
        re.IGNORECASE,
    )
    is_elliptical = (
        clean_q.lower().startswith(
            ("what about", "how about", "does that", "is that", "why?", "and with", "what of")
        )
        or (len(clean_q.split()) <= 6 and bool(referent_pattern.search(clean_q)))
    )
    has_referent = bool(referent_pattern.search(clean_q))

    if not (is_elliptical or has_referent):
        return clean_q

    prior_turn_texts: list[str] = []
    last_user_query = ""
    for item in reversed(history[-6:]):
        content = (item.get("content") or item.get("text") or item.get("message") or "").strip()
        role = (item.get("role") or item.get("sender") or "").lower()
        if content:
            prior_turn_texts.append(content)
            if role == "user" and not last_user_query:
                last_user_query = content

    if not prior_turn_texts:
        return clean_q

    context_candidate = last_user_query or prior_turn_texts[0]
    clean_candidate = re.sub(
        r"^(what|how|why|when|where|is|are|does|do|can|could|would)\s+(is|are|does|do)?\s*",
        "",
        context_candidate,
        flags=re.IGNORECASE,
    )
    clean_candidate = clean_candidate.strip("? .!").strip()
    if not clean_candidate:
        clean_candidate = context_candidate.strip("? .!").strip()

    words = clean_candidate.split()
    if len(words) > 8:
        clean_candidate = " ".join(words[:8])

    if not clean_candidate:
        return clean_q

    return f"{clean_q} (context: {clean_candidate})"


class AnswerPipelineService:
    """Executes the exact answer path shared by the API and claim-bearing evaluations."""

    @classmethod
    async def answer(
        cls,
        request: AnswerPipelineRequest,
        stage_callback: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> AnswerTrace:
        if not request.query.strip():
            raise ValueError("Message cannot be empty")

        started_at = time.perf_counter()
        paper_id = safe_paper_id(request.paper_id)
        requested_model = request.requested_model or OLLAMA_MODEL
        effective_query = resolve_conversational_query(request.query, request.history)
        analysis = QuestionAnalyzer.analyze_query(effective_query)
        analysis.original_query = request.query
        capabilities = ModelRegistry.resolve_capabilities(
            requested_model, mode=request.capability_mode
        )
        route_budget = QuestionRouter.route(effective_query, capabilities)
        evidence_budget = BudgetingService.get_evidence_budget(capabilities)
        git_revision, git_dirty = _git_identity()
        trace = AnswerTrace(
            trace_id=f"trace_{uuid.uuid4().hex[:12]}",
            timestamp=time.time(),
            paper_id=paper_id,
            query=request.query,
            request=request.model_copy(update={"paper_id": paper_id}),
            run_identity=RunIdentity(
                experiment_id=request.experiment_id,
                git_revision=git_revision,
                git_dirty=git_dirty,
            ),
            capabilities=capabilities,
            route_budget=route_budget.model_dump(mode="json"),
            evidence_budget=evidence_budget,
            analysis=analysis,
            reasoning_level=analysis.reasoning_level.value,
            target_modalities=[item.value for item in analysis.target_modalities],
            subqueries=analysis.subqueries,
            hardware_tier=evidence_budget.hardware_tier.value,
            generation=LocalGenerationMetadata(
                requested_model=requested_model,
                resolved_model=requested_model,
                prompt_version=ANSWER_PROMPT_VERSION,
            ),
            intervention=InterventionExecutionTrace(requested=request.intervention),
        )

        def stage(name: str, stage_started: float, error: Exception | None = None) -> None:
            timing = StageTiming(
                stage=name,
                duration_ms=round((time.perf_counter() - stage_started) * 1000.0, 3),
                status="error" if error else "ok",
                error=f"{type(error).__name__}: {error}" if error else None,
            )
            trace.timings.append(timing)
            if stage_callback is not None:
                try:
                    stage_callback(name, timing.model_dump(mode="json"))
                except Exception as cb_exc:
                    logger.warning("Stage callback error for %s: %s", name, cb_exc)

        def finish() -> AnswerTrace:
            trace.latency_ms = round((time.perf_counter() - started_at) * 1000.0, 3)
            TelemetryService.persist_trace(trace)
            return trace

        load_started = time.perf_counter()
        try:
            metadata_path, _pages_path, chunks_path, _pdf_path = _paper_paths(paper_id)
            metadata = read_json(metadata_path)
            chunks = read_json(chunks_path)
        except Exception as exc:
            stage("load_paper", load_started, exc)
            raise
        stage("load_paper", load_started)

        source_started = time.perf_counter()
        anchor_chunks = [
            {**chunk, "source_paper_id": chunk.get("source_paper_id") or paper_id}
            for chunk in chunks
        ]
        anchor_figures_path = paper_dir(paper_id) / "figures.json"
        anchor_figures = read_json(anchor_figures_path) if anchor_figures_path.exists() else []
        anchor_chunks = _merge_source_figure_chunks(
            paper_id,
            anchor_chunks,
            anchor_figures if isinstance(anchor_figures, list) else [],
        )
        all_chunks = list(anchor_chunks)
        secondary_meta: dict[str, dict[str, Any]] = {}
        for secondary_id_raw in request.secondary_paper_ids:
            secondary_id = safe_paper_id(secondary_id_raw)
            secondary_directory = paper_dir(secondary_id)
            secondary_chunks_path = secondary_directory / "chunks.json"
            secondary_metadata_path = secondary_directory / "metadata.json"
            secondary_figures_path = secondary_directory / "figures.json"
            if secondary_chunks_path.exists():
                raw_secondary_chunks = read_json(secondary_chunks_path)
                secondary_chunks = [
                    {**chunk, "source_paper_id": chunk.get("source_paper_id") or secondary_id}
                    for chunk in raw_secondary_chunks
                ]
                secondary_figures = (
                    read_json(secondary_figures_path)
                    if secondary_figures_path.exists() else []
                )
                all_chunks.extend(_merge_source_figure_chunks(
                    secondary_id,
                    secondary_chunks,
                    secondary_figures if isinstance(secondary_figures, list) else [],
                ))
            if secondary_metadata_path.exists():
                secondary_meta[secondary_id] = read_json(secondary_metadata_path)
        stage("load_sources", source_started)

        # Snippet requests are still represented by the same trace contract even though
        # they do not execute document retrieval.
        if request.snippet_id and request.snippet_page and request.snippet_bbox:
            generation_started = time.perf_counter()
            try:
                result = await answer_with_custom_snippet(
                    question=request.query,
                    snippet_id=request.snippet_id,
                    page_number=request.snippet_page,
                    bbox_norm=request.snippet_bbox,
                    snippet_text=request.snippet_text or "",
                    paper_id=paper_id,
                    paper_metadata=metadata,
                    model=request.requested_model,
                    seed=request.generation_seed,
                    decoding_options=request.decoding.model_dump(mode="json"),
                )
                raw_answer = str(result.get("answer") or "")
                verification = ClaimVerifierService.verify_and_repair_detailed(
                    answer=raw_answer,
                    citations=result.get("citations", []),
                    candidate_pool=result.get("citations", []),
                    controls=request.intervention,
                )
                answer = verification.final_answer
                citations = verification.citations
                is_fallback = bool(result.get("fallback"))
                if is_fallback and request.execution_policy == ExecutionPolicy.REQUIRE_LOCAL_MODEL:
                    raise RuntimeError("Snippet vision model fell back instead of executing")
                trace.generation.mode = GenerationMode.EXTRACTIVE_FALLBACK if is_fallback else GenerationMode.VISION_MODEL
                trace.generation.resolved_model = str(result.get("model_used") or requested_model)
                trace.generation.model_digest = result.get("model_digest")
                trace.generation.quantization = result.get("quantization")
                trace.generation.options = dict(result.get("generation_options") or {})
                trace.raw_answer = raw_answer
                trace.normalized_answer = raw_answer
                trace.final_answer = answer
                trace.citations = [_citation_trace(citation, paper_id, CitationOrigin.VISION_SERVICE) for citation in citations]
                trace.verification_report = verification.final_report
                trace.verification = VerificationTrace(
                    initial_report=verification.initial_report,
                    report=verification.final_report,
                    repair_requested=request.intervention.repair_mode.value != "NONE",
                    repair_actions_recorded=[edit.action.value for edit in verification.edits],
                    edits=verification.edits,
                    answer_text_changed=answer != raw_answer,
                    reverified=verification.reverified,
                )
                trace.intervention.executed_repair_mode = request.intervention.repair_mode
                trace.intervention.verification_reached = True
                if verification.final_report.has_abstained:
                    trace.status = PipelineStatus.ABSTAINED
                    trace.abstention = AbstentionTrace(
                        abstained=True,
                        stage="verification",
                        reason_code="NO_SUPPORTED_CLAIMS_AFTER_REPAIR",
                        user_message=answer,
                    )
                trace.response_metadata = {
                    "vision": True,
                    "is_snippet": True,
                    "snippet_id": request.snippet_id,
                    "figure_label": f"Snippet (Page {request.snippet_page})",
                    "figure_image_url": f"/api/papers/{paper_id}/snippets/{request.snippet_id}.png",
                }
                stage("generation", generation_started)
                return finish()
            except Exception as exc:
                stage("generation", generation_started, exc)
                trace.status = PipelineStatus.ERROR
                trace.generation.error = f"{type(exc).__name__}: {exc}"
                return finish()

        retrieval_started = time.perf_counter()
        page_hints = extract_page_hints(request.query)
        selected: list[dict[str, Any]] = []
        provenance: dict[tuple[str, str, str], dict[str, Any]] = {}
        seen: set[tuple[str, str, str]] = set()
        visual_page_query_traces: list[dict[str, Any]] = []

        async def retrieve(query: str, limit: int, subquery_id: str) -> list[dict[str, Any]]:
            call_metadata: dict[str, Any] = {}
            hits = await asyncio.to_thread(
                retrieve_chunks,
                query,
                all_chunks,
                limit,
                page_hints,
                paper_id,
                visual_page_backend=(
                    None
                    if request.visual_page_backend == "configured"
                    else request.visual_page_backend
                ),
                retrieval_metadata=call_metadata,
            )
            visual_page_query_traces.append({
                "retrieval_query": query,
                "subquery_id": subquery_id,
                **dict(call_metadata.get("visual_page_retrieval") or {}),
            })
            return hits

        if len(analysis.subqueries) > 1:
            retrieval_specs = [(effective_query, route_budget.text_top_k, "GLOBAL")]
            per_sub_limit = max(2, route_budget.text_top_k // len(analysis.subqueries) + 1)
            retrieval_specs.extend(
                (subquery.query_text, per_sub_limit, subquery.subquery_id)
                for subquery in analysis.subqueries
            )
        else:
            retrieval_specs = [(effective_query, route_budget.text_top_k, analysis.subqueries[0].subquery_id)]

        for retrieval_query, limit, subquery_id in retrieval_specs:
            for hit in await retrieve(retrieval_query, limit, subquery_id):
                key = evidence_key(hit, paper_id=paper_id)
                record = provenance.setdefault(key, {
                    "queries": [],
                    "subquery_ids": [],
                    "query_channel_results": [],
                })
                if retrieval_query not in record["queries"]:
                    record["queries"].append(retrieval_query)
                if subquery_id not in record["subquery_ids"]:
                    record["subquery_ids"].append(subquery_id)
                record["query_channel_results"].append(
                    _query_channel_trace(hit, retrieval_query, subquery_id)
                )
                if key not in seen:
                    seen.add(key)
                    selected.append(hit)

        # Active Cross-Modal Evidence Graph Expansion:
        # If retrieved text chunks cite figures or tables (e.g. 'Table 3', 'Figure 2')
        # that are not yet in selected, actively pull in their canonical visual/table chunks.
        existing_fig_labels = {
            _normalize_figure_label(str(chunk.get("label") or ""))
            for chunk in selected
            if chunk.get("is_figure_chunk") and chunk.get("label")
        }
        referenced_labels: list[str] = []
        for chunk in selected[:8]:
            if not chunk.get("is_figure_chunk"):
                for ref in extract_figure_refs(str(chunk.get("text") or "")):
                    if ref not in referenced_labels and ref not in existing_fig_labels:
                        referenced_labels.append(ref)

        if referenced_labels:
            expanded_count = 0
            for ref in referenced_labels:
                if expanded_count >= evidence_budget.max_visual_crops:
                    break
                for chunk in all_chunks:
                    if chunk.get("is_figure_chunk") and chunk.get("label"):
                        norm_label = _normalize_figure_label(str(chunk["label"]))
                        if norm_label == ref:
                            key = evidence_key(chunk, paper_id=paper_id)
                            if key not in seen:
                                seen.add(key)
                                expanded_chunk = dict(chunk)
                                expanded_chunk["is_bridged_visual"] = True
                                expanded_chunk["graph_expanded"] = True
                                expanded_chunk["reasoning_role"] = "cross_modal_grounding"
                                selected.append(expanded_chunk)
                                provenance[key] = {
                                    "queries": ["GRAPH_COREF_EXPANSION"],
                                    "subquery_ids": ["GRAPH_COREF"],
                                    "query_channel_results": [],
                                }
                                expanded_count += 1
                                break

        stage("retrieval", retrieval_started)
        trace.retrieval_metadata["image_embedding"] = VisualEmbeddingService.status()
        trace.retrieval_metadata["visual_page_retrieval"] = visual_page_query_traces
        trace.retrieval_hits = []
        for rank, hit in enumerate(selected, start=1):
            record = provenance[evidence_key(hit, paper_id=paper_id)]
            trace.retrieval_hits.append(_hit_trace(
                hit,
                paper_id,
                rank,
                record["queries"],
                record["subquery_ids"],
                record["query_channel_results"],
            ))

        if not selected:
            message = "The paper context does not contain enough information to answer that."
            trace.status = PipelineStatus.ABSTAINED
            trace.abstention = AbstentionTrace(
                abstained=True, stage="retrieval", reason_code="NO_EVIDENCE_RETRIEVED", user_message=message,
            )
            trace.final_answer = message
            return finish()

        context_started = time.perf_counter()
        candidate_chunks: list[dict[str, Any]] = []
        candidate_seen: set[tuple[str, str, str]] = set()
        for chunk in chunks[:2] + selected:
            key = evidence_key(chunk, paper_id=paper_id)
            if key not in candidate_seen:
                candidate_seen.add(key)
                candidate_chunks.append(chunk)

        reasoning_started = time.perf_counter()
        graph, path = EvidenceGraphService.build_evidence_graph(request.query, candidate_chunks, analysis)
        pruned_graph, pruned_path = BudgetingService.prune_to_budget(graph, path, evidence_budget)
        trace.evidence_graph = pruned_graph
        trace.reasoning_path = pruned_path.steps

        retained_node_ids = {n.node_id for n in pruned_graph.nodes}
        budgeted_chunks = [
            chunk for chunk in candidate_chunks
            if (chunk.get("evidence_id") or chunk.get("chunk_id")) in retained_node_ids
        ]
        context_chunks = budgeted_chunks if budgeted_chunks else candidate_chunks[:evidence_budget.max_evidence_blocks]
        prompt_chunks = context_chunks

        if analysis.requires_arithmetic:
            table_chunks = [
                chunk for chunk in context_chunks
                if chunk.get("is_table_chunk") or "|" in str(chunk.get("text", ""))
            ]
            if table_chunks:
                words = [word for word in request.query.split() if len(word) > 3]
                entity_a = words[0] if words else "Model"
                entity_b = words[1] if len(words) > 1 else "Baseline"
                trace.numeric_plan = TableArithmeticService.extract_and_calculate_from_table_text(
                    table_text=str(table_chunks[0].get("text", "")),
                    entity_a=entity_a,
                    entity_b=entity_b,
                    op=NumericOp.DIFFERENCE,
                )
        stage("reasoning_artifacts", reasoning_started)

        sufficiency_started = time.perf_counter()
        sufficiency = ClaimVerifierService.compute_sufficiency(
            query=request.query,
            retrieved_chunks=prompt_chunks,
            requires_vision=route_budget.requires_native_vision,
            can_vision=capabilities.can_process_images(),
        )
        stage("sufficiency", sufficiency_started)
        visual_inspection_required = (
            sufficiency.reason_code == "VISUAL_INSPECTION_REQUIRED"
        )
        if not sufficiency.is_sufficient and not visual_inspection_required:
            message = (
                "**Abstained**\n\nThe paper context does not provide sufficient evidence to answer "
                f"this question reliably ({sufficiency.reason_code.lower().replace('_', ' ')})."
            )
            trace.status = PipelineStatus.ABSTAINED
            trace.abstention = AbstentionTrace(
                abstained=True,
                stage="sufficiency",
                reason_code=sufficiency.reason_code,
                user_message=message,
            )
            trace.final_answer = message
            return finish()

        context_ids = {_identity_string(chunk, paper_id) for chunk in context_chunks}
        prompt_ids = {_identity_string(chunk, paper_id) for chunk in prompt_chunks}
        existing_trace_ids = {item.identity.global_id for item in trace.retrieval_hits}
        for chunk in context_chunks:
            global_id = _identity_string(chunk, paper_id)
            if global_id not in existing_trace_ids:
                trace.retrieval_hits.append(_hit_trace(chunk, paper_id, None, ["CONTEXT_PRELUDE"], []))
                existing_trace_ids.add(global_id)
        for item in trace.retrieval_hits:
            item.selected_for_context = item.identity.global_id in context_ids
            item.shown_to_generator = item.identity.global_id in prompt_ids
        evidence_items = await asyncio.to_thread(_build_evidence_items, request.query[:900], prompt_chunks, 7)
        trace.prompt_evidence = [
            PromptEvidenceTrace(
                prompt_evidence_id=str(item["evidence_id"]),
                identity=_identity_model(
                    {
                        "source_paper_id": item.get("source_paper_id"),
                        "document_id": item.get("document_id"),
                        "chunk_id": item.get("chunk_id"),
                        "evidence_id": item.get("source_evidence_id"),
                        "page": item.get("page"),
                        "text": item.get("quote"),
                    },
                    paper_id,
                ),
                page=item.get("page") if isinstance(item.get("page"), int) else None,
                section=str(item.get("section_title") or ""),
                modality=str(item.get("chunk_type") or "text"),
                quote=str(item.get("quote") or ""),
                content_sha256=hashlib.sha256(str(item.get("quote") or "").encode("utf-8")).hexdigest(),
            )
            for item in evidence_items
        ]
        prompt = _build_prompt(request, metadata, evidence_items, secondary_meta, route_budget.route_type, numeric_result=trace.numeric_plan, effective_query=effective_query)
        trace.generation.prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        stage("context_assembly", context_started)

        # Vision is selected by fused evidence relevance, not only visual keywords.
        retrieved_figures = [chunk for chunk in selected if chunk.get("is_figure_chunk")]
        unique_figures: list[dict[str, Any]] = []
        seen_fig_keys: set[str] = set()
        for fig in retrieved_figures:
            key_id = str(fig.get("label") or fig.get("image_file") or fig.get("figure_id") or "").lower().strip()
            if key_id and key_id not in seen_fig_keys:
                seen_fig_keys.add(key_id)
                unique_figures.append(fig)
            elif not key_id:
                unique_figures.append(fig)
        requested_visual_limit = max(1, route_budget.visual_items or 2)
        visual_limit = min(requested_visual_limit, evidence_budget.max_visual_crops)
        if visual_limit >= 2 and (
            TargetModality.TABLE in analysis.target_modalities
            or analysis.requires_arithmetic
            or any(sq.target_modality in (TargetModality.TABLE, TargetModality.MULTIMODAL) for sq in analysis.subqueries)
        ):
            table_figs = [
                f for f in unique_figures
                if f.get("figure_type") == "table" or "table" in str(f.get("label", "")).lower()
            ]
            diagram_figs = [
                f for f in unique_figures
                if f.get("figure_type") != "table" and "table" not in str(f.get("label", "")).lower()
            ]
            if table_figs and diagram_figs:
                selected_figures = [diagram_figs[0], table_figs[0]]
                remaining = [f for f in unique_figures if f not in selected_figures]
                selected_figures.extend(remaining[: visual_limit - 2])
            else:
                selected_figures = unique_figures[:visual_limit] if visual_limit > 0 else []
        else:
            selected_figures = unique_figures[:visual_limit] if visual_limit > 0 else []
        top_has_visual = any(chunk.get("is_figure_chunk") for chunk in selected[:2])
        has_ranked_image_evidence = any(
            (
                chunk.get("image_embedding_eligible") is True
                and type(chunk.get("image_embedding_rank")) is int
                and chunk["image_embedding_rank"] <= max(3, visual_limit)
                and isinstance(chunk.get("image_embedding_score"), (int, float))
                and not isinstance(chunk.get("image_embedding_score"), bool)
                and isinstance(chunk.get("image_embedding_threshold"), (int, float))
                and not isinstance(chunk.get("image_embedding_threshold"), bool)
                and float(chunk["image_embedding_score"])
                >= float(chunk["image_embedding_threshold"])
            )
            or (
                chunk.get("page_image_eligible") is True
                and type(chunk.get("page_image_rank")) is int
                and chunk["page_image_rank"] <= max(3, visual_limit)
                and isinstance(chunk.get("page_image_score"), (int, float))
                and not isinstance(chunk.get("page_image_score"), bool)
                and isinstance(chunk.get("page_image_threshold"), (int, float))
                and not isinstance(chunk.get("page_image_threshold"), bool)
                and float(chunk["page_image_score"])
                >= float(chunk["page_image_threshold"])
            )
            for chunk in selected_figures
        )
        should_use_vision = bool(
            selected_figures
            and capabilities.can_process_images()
            and (
                has_ranked_image_evidence
                or route_budget.requires_native_vision
                or top_has_visual
                or route_budget.route_type in {
                    QuestionRouteType.FIGURE_VISUAL,
                    QuestionRouteType.CHART_NUMERIC,
                    QuestionRouteType.MIXED_TEXT_VISUAL,
                    QuestionRouteType.COMPARISON,
                    QuestionRouteType.TABLE_NUMERIC,
                }
            )
        )
        if should_use_vision:
            shown_visual_ids = {
                _identity_string(chunk, paper_id) for chunk in selected_figures
            }
            for item in trace.retrieval_hits:
                if item.identity.global_id in shown_visual_ids:
                    item.selected_for_context = True
                    item.shown_to_generator = True
            generation_started = time.perf_counter()
            try:
                vision_result = await answer_with_multimodal_evidence(
                    question=request.query,
                    figure_chunks=selected_figures,
                    context_chunks=[chunk for chunk in selected if not chunk.get("is_figure_chunk")],
                    paper_id=paper_id,
                    paper_metadata=metadata,
                    source_metadata={paper_id: metadata, **secondary_meta},
                    model=request.requested_model,
                    seed=request.generation_seed,
                    decoding_options=request.decoding.model_dump(mode="json"),
                )
                raw_vision_answer = str(vision_result.get("answer") or "")
                is_fallback = bool(vision_result.get("fallback"))
                if (
                    not is_uninformative_visual_answer(raw_vision_answer)
                    and not (visual_inspection_required and is_fallback)
                ):
                    if is_fallback and request.execution_policy == ExecutionPolicy.REQUIRE_LOCAL_MODEL:
                        raise RuntimeError("Vision model fell back instead of executing")
                    verification = ClaimVerifierService.verify_and_repair_detailed(
                        answer=raw_vision_answer,
                        citations=vision_result.get("citations", []),
                        candidate_pool=selected,
                        controls=request.intervention,
                    )
                    final_answer = verification.final_answer
                    verified_citations = verification.citations
                    trace.generation.mode = GenerationMode.EXTRACTIVE_FALLBACK if is_fallback else GenerationMode.VISION_MODEL
                    trace.generation.resolved_model = str(vision_result.get("model_used") or requested_model)
                    trace.generation.model_digest = vision_result.get("model_digest")
                    trace.generation.quantization = vision_result.get("quantization")
                    trace.generation.options = dict(vision_result.get("generation_options") or {})
                    trace.raw_answer = raw_vision_answer
                    trace.normalized_answer = raw_vision_answer
                    trace.final_answer = final_answer
                    trace.citations = [
                        _citation_trace(citation, paper_id, CitationOrigin.VISION_SERVICE)
                        for citation in verified_citations
                    ]
                    trace.verification_report = verification.final_report
                    trace.verification = VerificationTrace(
                        initial_report=verification.initial_report,
                        report=verification.final_report,
                        repair_requested=request.intervention.repair_mode.value != "NONE",
                        repair_actions_recorded=[edit.action.value for edit in verification.edits],
                        edits=verification.edits,
                        answer_text_changed=final_answer != raw_vision_answer,
                        reverified=verification.reverified,
                    )
                    trace.intervention.executed_repair_mode = request.intervention.repair_mode
                    trace.intervention.verification_reached = True
                    if verification.final_report.has_abstained:
                        trace.status = PipelineStatus.ABSTAINED
                        trace.abstention = AbstentionTrace(
                            abstained=True,
                            stage="verification",
                            reason_code="NO_SUPPORTED_CLAIMS_AFTER_REPAIR",
                            user_message=final_answer,
                        )
                    vision_source_id = str(
                        vision_result.get("source_paper_id") or paper_id
                    )
                    trace.response_metadata = {
                        "vision": True,
                        "vision_fallback": is_fallback,
                        "visual_retrieval_triggered": has_ranked_image_evidence,
                        "visual_observation_model_generated": bool(
                            vision_result.get("visual_observation_model_generated")
                        ),
                        "figure_id": vision_result.get("figure_id"),
                        "figure_source_paper_id": vision_source_id,
                        "figure_label": vision_result.get("label", "Figure"),
                        "figure_image_url": (
                            vision_result.get("image_url")
                            or (
                                f"/api/papers/{vision_source_id}/figures/{vision_result.get('figure_id')}.png"
                                if vision_result.get("figure_id") else None
                            )
                        ),
                    }
                    stage("generation", generation_started)
                    return finish()
            except Exception as exc:
                if request.execution_policy == ExecutionPolicy.REQUIRE_LOCAL_MODEL:
                    stage("generation", generation_started, exc)
                    trace.status = PipelineStatus.ERROR
                    trace.generation.error = f"{type(exc).__name__}: {exc}"
                    return finish()
            stage("generation", generation_started)

        if visual_inspection_required:
            message = (
                "**Abstained**\n\nThe retrieved text is insufficient, and local pixel "
                "inspection did not produce usable source-scoped visual evidence."
            )
            trace.status = PipelineStatus.ABSTAINED
            trace.abstention = AbstentionTrace(
                abstained=True,
                stage="visual_inspection",
                reason_code="VISUAL_INSPECTION_UNSUCCESSFUL",
                user_message=message,
            )
            trace.final_answer = message
            return finish()

        generation_started = time.perf_counter()
        raw_answer = ""
        available = await ollama_available()
        if available:
            try:
                result = await generate_result(
                    prompt,
                    temperature=request.decoding.temperature,
                    model=request.requested_model,
                    seed=request.generation_seed,
                    top_p=request.decoding.top_p,
                    num_ctx=request.decoding.num_ctx,
                    num_predict=request.decoding.num_predict,
                )
                raw_answer = result.response
                trace.generation = LocalGenerationMetadata(
                    requested_model=result.requested_model,
                    resolved_model=result.resolved_model,
                    model_digest=result.model_digest,
                    quantization=result.quantization,
                    options=result.options,
                    prompt_version=ANSWER_PROMPT_VERSION,
                    prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                    mode=GenerationMode.LOCAL_MODEL,
                    prompt_eval_count=result.prompt_eval_count,
                    eval_count=result.eval_count,
                    total_duration_ns=result.total_duration,
                    load_duration_ns=result.load_duration,
                    prompt_eval_duration_ns=result.prompt_eval_duration,
                    eval_duration_ns=result.eval_duration,
                )
                if not raw_answer:
                    raise RuntimeError("The local model returned an empty answer")
            except Exception as exc:
                if request.execution_policy == ExecutionPolicy.REQUIRE_LOCAL_MODEL:
                    stage("generation", generation_started, exc)
                    trace.status = PipelineStatus.ERROR
                    trace.generation.error = f"{type(exc).__name__}: {exc}"
                    return finish()
                trace.generation.error = f"{type(exc).__name__}: {exc}"
        elif request.execution_policy == ExecutionPolicy.REQUIRE_LOCAL_MODEL:
            exc = RuntimeError("Required local model is unavailable")
            stage("generation", generation_started, exc)
            trace.status = PipelineStatus.ERROR
            trace.generation.error = str(exc)
            return finish()

        extractive_citations: list[dict[str, Any]] = []
        if not raw_answer:
            raw_answer, extractive_citations = _extractive_tutor_answer(request.query, selected)
            trace.generation.mode = GenerationMode.EXTRACTIVE_FALLBACK
            trace.generation.resolved_model = None
        trace.raw_answer = raw_answer
        stage("generation", generation_started)

        verification_started = time.perf_counter()
        normalized = re.sub(r"\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$", r"\1-\2", raw_answer)
        normalized = re.sub(r"\$\\text\{([^}]+)\}_\{([^}]+)\}\$", r"\1-\2", normalized)
        normalized = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", normalized)
        normalized = re.sub(r"\\text\{([^}]+)\}", r"\1", normalized)
        normalized, citations = _normalize_evidence_citations(normalized, evidence_items)
        citation_origin = CitationOrigin.MODEL_EMITTED
        if not citations and extractive_citations:
            citation_origin = CitationOrigin.APPLICATION_IMPUTED
            citations = extractive_citations
        elif not citations and evidence_items:
            citation_origin = CitationOrigin.APPLICATION_IMPUTED
            citations = [
                {
                    "ref_id": index,
                    "page": item.get("page"),
                    "chunk_id": item.get("chunk_id"),
                    "section_title": item.get("section_title"),
                    "chunk_type": item.get("chunk_type"),
                    "quote": item.get("quote"),
                    "source_paper_id": item.get("source_paper_id"),
                    "document_id": item.get("document_id"),
                    "source_evidence_id": item.get("source_evidence_id"),
                }
                for index, item in enumerate(evidence_items[:2], start=1)
            ]
        verification = ClaimVerifierService.verify_and_repair_detailed(
            answer=normalized,
            citations=citations,
            candidate_pool=evidence_items,
            controls=request.intervention,
        )
        final_answer = verification.final_answer
        verified_citations = verification.citations
        trace.normalized_answer = normalized
        trace.final_answer = final_answer
        trace.citations = [
            _citation_trace(citation, paper_id, citation_origin)
            for citation in verified_citations
        ]
        trace.verification_report = verification.final_report
        trace.verification = VerificationTrace(
            initial_report=verification.initial_report,
            report=verification.final_report,
            repair_requested=request.intervention.repair_mode.value != "NONE",
            repair_actions_recorded=[edit.action.value for edit in verification.edits],
            edits=verification.edits,
            answer_text_changed=final_answer != normalized,
            reverified=verification.reverified,
        )
        trace.intervention.executed_repair_mode = request.intervention.repair_mode
        trace.intervention.verification_reached = True
        if verification.final_report.has_abstained:
            trace.status = PipelineStatus.ABSTAINED
            trace.abstention = AbstentionTrace(
                abstained=True,
                stage="verification",
                reason_code="NO_SUPPORTED_CLAIMS_AFTER_REPAIR",
                user_message=final_answer,
            )
        stage("verification", verification_started)
        return finish()
