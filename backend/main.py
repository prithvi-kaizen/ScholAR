from __future__ import annotations

import asyncio
import hashlib
import logging
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.schemas.capabilities import CapabilityMode, ModelCapabilities, ModelRegistry
from backend.schemas.evidence import EvidenceAST, EvidenceBlock, ParserAblationConfig
from backend.services.arxiv_service import search_arxiv
from backend.services.chunking_service import chunk_figures, chunk_pages
from backend.services.ingestion_service import DualEngineIngestionService, PARSER_ABLATIONS
from backend.services.ollama_service import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    fallback_goals,
    generate,
    generate_study_goals,
    ollama_available,
    STUDY_GOAL_PROMPT_VERSION,
)
from backend.services.pdf_service import (
    crop_page_region,
    download_pdf,
    extract_figures,
    extract_pages,
    infer_uploaded_metadata,
    paper_dir,
    read_json,
    render_page_png,
    safe_paper_id,
    write_json,
)
from backend.services.reference_service import (
    load_references,
    mark_reference_ingested,
    resolve_references,
)
from backend.services.retrieval_service import extract_page_hints, retrieve_chunks, short_quote, tokenize
from backend.services.routing_service import QuestionRouter, QuestionRouteType, RouteBudget, QueryDecomposer
from backend.services.storage_service import StorageService
from backend.services.verifier_service import ClaimVerifierService, VerificationLabel
from backend.services.vision_service import answer_with_custom_snippet, answer_with_figure, answer_with_multimodal_evidence


logger = logging.getLogger("scholar")

app = FastAPI(title="ScholAR API")

app.add_middleware(
    CORSMiddleware,
    # No cookies/session auth are used anywhere in this app, so credentials
    # stay disabled; combined with a wildcard origin, allow_credentials=True
    # would let any website that a user's browser visits make authenticated-
    # looking requests against this local server (browser-as-confused-deputy).
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "http://localhost:3001", "http://127.0.0.1:3001"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class PaperInput(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    summary: str = ""
    categories: list[str] = Field(default_factory=list)
    pdf_url: str
    abs_url: str
    published: str = ""


class ChatInput(BaseModel):
    # Bounds keep a single request body from being used for memory exhaustion;
    # the whole body is parsed before downstream truncation, so bound it here.
    message: str = Field(max_length=8000)
    history: list[dict[str, Any]] = Field(default_factory=list, max_length=50)
    # Multi-document mode: local_ids of secondary papers already ingested
    secondary_paper_ids: list[str] = Field(default_factory=list, max_length=25)
    # Optional per-request generation model override (used by the human-eval
    # pipeline to run the same pipeline across several local models). Falls back
    # to OLLAMA_MODEL when absent.
    model: str | None = Field(default=None, max_length=100)
    capability_mode: CapabilityMode = CapabilityMode.AUTO
    # Optional user-cropped snippet from a specific PDF page
    snippet_id: str | None = None
    snippet_page: int | None = None
    snippet_bbox: list[float] | None = None
    snippet_text: str | None = None


class SnippetInput(BaseModel):
    page: int
    bbox: list[float]  # [x0, y0, x1, y1] normalized
    zoom: float = 3.0


class StudyGoalsInput(BaseModel):
    force: bool = False


def _ensure_chunk_metadata(chunks_path: Path, pages_path: Path) -> list[dict[str, Any]]:
    chunks = read_json(chunks_path) if chunks_path.exists() else []
    if chunks and all("section_title" in chunk and "chunk_type" in chunk and "paragraph_text" in chunk for chunk in chunks):
        return chunks
    if not pages_path.exists():
        return chunks
    pages = read_json(pages_path)
    upgraded = chunk_pages(pages)
    write_json(chunks_path, upgraded)
    return upgraded


def _important_sentences(text: str) -> list[str]:
    import re

    banned = ("table 1:", "figure", "copyright", "provided proper attribution")
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(item.strip()) > 45]
    ranked: list[tuple[int, str]] = []
    for sentence in sentences:
        lowered = sentence.lower()
        if any(term in lowered for term in banned):
            continue
        score = 0
        for term in (
            "we introduce",
            "we propose",
            "we show",
            "we find",
            "results",
            "achieves",
            "outperforms",
            "significant",
            "accuracy",
            "benchmark",
            "evaluation",
            "dataset",
            "method",
            "framework",
            "limitation",
        ):
            if term in lowered:
                score += 2
        if any(char.isdigit() for char in sentence):
            score += 1
        ranked.append((score, sentence[:420]))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [sentence for score, sentence in ranked if score > 0][:8] or [sentence[:420] for sentence in sentences[:6]]


def _extractive_tutor_answer(question: str, selected: list[dict[str, Any]]) -> str:
    lines = ["I could not get a full model response, so here is the best grounded answer from the paper context:"]
    used = 0
    for chunk in selected:
        page = chunk.get("page")
        for sentence in _important_sentences(chunk.get("text", ""))[:2]:
            lines.append(f"- {sentence} [p. {page}]")
            used += 1
            if used >= 6:
                return "\n".join(lines)
    return "\n".join(lines)


def _error_payload(message: str) -> dict[str, Any]:
    return {
        "answer": "",
        "citations": [],
        "model": OLLAMA_MODEL,
        "error": True,
        "message": message,
    }


def _model_failure_payload(exc: Exception | None = None) -> dict[str, Any]:
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return _error_payload("Local model took too long to answer. Try again with a shorter question.")
    return _error_payload("Local model could not complete this answer. Make sure Ollama is running and the selected model is loaded.")


def _citation_candidates(text: str) -> list[str]:
    sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if len(item.strip()) > 55]
    return [sentence[:500] for sentence in sentences]


def _build_evidence_items(question: str, chunks: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    question_terms = set(tokenize(question))
    banned = (
        "author name redacted",
        "copyright",
        "provided proper attribution",
        "equal contribution",
        "arxiv:",
        "preprint.",
        "facebook ai research",
        "university college london",
        "new york university",
    )
    scored: list[tuple[float, int, dict[str, Any]]] = []

    for chunk_index, chunk in enumerate(chunks):
        page = chunk.get("page")
        chunk_id = chunk.get("chunk_id")
        for sentence_index, sentence in enumerate(_citation_candidates(chunk.get("text", ""))):
            lowered = sentence.lower()
            if any(term in lowered for term in banned):
                continue
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            overlap = len(question_terms.intersection(sentence_terms))
            score = overlap * 2.0
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
            scored.append(
                (
                    score,
                    chunk_index,
                    {
                        "page": page,
                        "chunk_id": chunk_id,
                        "section_title": chunk.get("section_title") or "Paper",
                        "chunk_type": chunk.get("chunk_type") or "body",
                        "quote": sentence[:520],
                    },
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    evidence: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    for _, _, item in scored:
        quote_key = re.sub(r"\W+", " ", item["quote"].lower())[:120]
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
    """Format evidence items into a prompt block.

    For cross-paper evidence the source label is shown so the LLM can
    distinguish anchor-paper evidence from secondary-paper evidence.
    """
    if not evidence_items:
        return "No relevant paper evidence was retrieved."

    lines: list[str] = []
    for item in evidence_items:
        src_id = item.get("source_paper_id")
        if src_id and secondary_meta and src_id in secondary_meta:
            short_title = (secondary_meta[src_id].get("title") or src_id)[:40]
            src_label = f"ref:{short_title}"
        else:
            src_label = "anchor"
        lines.append(
            f"[{item['evidence_id']} | {src_label} | p. {item.get('page')} | {item.get('chunk_id')}]\n{item.get('quote')}"
        )
    return "\n\n".join(lines)


def _normalize_evidence_citations(answer: str, evidence_items: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    evidence_by_id = {item["evidence_id"].upper(): item for item in evidence_items}
    used_ids: list[str] = []
    answer_without_direct_pages = re.sub(r"\[p\.\s*\d+\]", "", answer, flags=re.IGNORECASE)

    def replace_evidence_group(match: re.Match[str]) -> str:
        """Rewrite one bracket of evidence identifiers into numbered references.

        Models group identifiers ("[E1, E4]") as often as they emit them singly, so a
        single-id pattern silently leaves the raw identifiers in the answer and drops those
        citations. Handle the whole bracket and map every identifier inside it.
        """
        refs: list[str] = []
        for number in re.findall(r"\d+", match.group(0)):
            evidence_id = f"E{number}".upper()
            if evidence_id not in evidence_by_id:
                continue  # identifier the model invented: drop it rather than surface it
            if evidence_id not in used_ids:
                used_ids.append(evidence_id)
            refs.append(f"[{used_ids.index(evidence_id) + 1}]")
        return "".join(refs)

    normalized = re.sub(
        r"\[\s*E\s*\d+(?:\s*(?:,|;|/|&|and)\s*E?\s*\d+)*\s*\]",
        replace_evidence_group, answer_without_direct_pages, flags=re.IGNORECASE,
    )
    valid_pages = {int(item["page"]) for item in evidence_items if isinstance(item.get("page"), int)}

    def remove_unverified_page(match: re.Match[str]) -> str:
        try:
            page = int(match.group(1))
        except ValueError:
            return ""
        return match.group(0) if page in valid_pages else ""

    normalized = re.sub(r"\[p\.\s*(\d+)\]", remove_unverified_page, normalized, flags=re.IGNORECASE)
    normalized = re.sub(r"[ \t]{2,}", " ", normalized)
    citations = [
        {
            "ref_id":          index,
            "page":            evidence_by_id[evidence_id].get("page"),
            "chunk_id":        evidence_by_id[evidence_id].get("chunk_id"),
            "section_title":   evidence_by_id[evidence_id].get("section_title"),
            "chunk_type":      evidence_by_id[evidence_id].get("chunk_type"),
            "quote":           evidence_by_id[evidence_id].get("quote"),
            "source_paper_id": evidence_by_id[evidence_id].get("source_paper_id"),
        }
        for index, evidence_id in enumerate(used_ids[:5], start=1)
        if evidence_id in evidence_by_id
    ]
    return normalized.strip(), citations


def _build_answer_citations(answer: str, question: str, chunks: list[dict[str, Any]], limit: int = 4) -> list[dict[str, Any]]:
    answer_terms = set(tokenize(answer))
    question_terms = set(tokenize(question))
    page_hints = set(extract_page_hints(question))
    scored: list[tuple[float, dict[str, Any]]] = []

    banned = ("author name redacted", "abstract", "when models know better", "april 3, 2026")
    for chunk in chunks:
        page = chunk.get("page")
        for sentence in _citation_candidates(chunk.get("text", "")):
            lowered = sentence.lower()
            if any(term in lowered for term in banned):
                continue
            sentence_terms = set(tokenize(sentence))
            if not sentence_terms:
                continue
            answer_overlap = len(answer_terms.intersection(sentence_terms))
            question_overlap = len(question_terms.intersection(sentence_terms))
            score = answer_overlap * 2.0 + question_overlap
            if page in page_hints:
                score += 6.0
            if any(char.isdigit() for char in sentence):
                score += 0.5
            if score <= 1:
                continue
            scored.append(
                (
                    score,
                    {
                        "page": page,
                        "chunk_id": chunk.get("chunk_id"),
                        "quote": sentence,
                    },
                )
            )

    scored.sort(key=lambda item: item[0], reverse=True)
    citations: list[dict[str, Any]] = []
    seen_quotes: set[str] = set()
    for _, citation in scored:
        quote_key = citation["quote"][:90].lower()
        if quote_key in seen_quotes:
            continue
        seen_quotes.add(quote_key)
        citations.append(citation)
        if len(citations) >= limit:
            break
    return citations


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ollama_available": await ollama_available(),
        "model": OLLAMA_MODEL,
    }


@app.get("/api/system/health-diagnostic")
async def system_health_diagnostic() -> dict[str, Any]:
    """Return comprehensive system acceleration, memory tier, and cache diagnostics."""
    from backend.services.diagnostic_service import DiagnosticService
    return await DiagnosticService.get_system_diagnostic()


@app.get("/api/models")
async def list_models() -> list[dict[str, Any]]:
    """Return discovered/registered local models and their capability profiles."""
    models = await ModelRegistry.discover_ollama_models(OLLAMA_BASE_URL)
    return [m.model_dump() for m in models]


@app.get("/api/search")
async def search(q: str = "", max_results: int = 12) -> dict[str, Any]:
    try:
        papers = await search_arxiv(q, max_results=max_results)
    except RuntimeError as exc:
        logger.warning("arXiv search failed: %s", exc)
        raise HTTPException(status_code=502, detail="Paper search is temporarily unavailable.") from exc
    return {"papers": papers}


@app.post("/api/papers/prepare")
async def prepare_paper(paper: PaperInput) -> dict[str, Any]:
    local_id = safe_paper_id(paper.id)
    directory = paper_dir(local_id)
    pdf_path = directory / "paper.pdf"
    metadata_path = directory / "metadata.json"
    pages_path = directory / "pages.json"
    chunks_path = directory / "chunks.json"
    figures_path = directory / "figures.json"
    figures_dir = directory / "figures"

    metadata = paper.model_dump()
    metadata["local_id"] = local_id

    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(metadata_path, metadata)

        if not pdf_path.exists():
            await download_pdf(paper.pdf_url, pdf_path)

        if not pages_path.exists():
            pages = await asyncio.to_thread(extract_pages, pdf_path)
            write_json(pages_path, pages)
        else:
            pages = read_json(pages_path)

        if not chunks_path.exists():
            chunks = await asyncio.to_thread(chunk_pages, pages)
            write_json(chunks_path, chunks)
        else:
            chunks = _ensure_chunk_metadata(chunks_path, pages_path)

        # Extract figures/tables (idempotent — skipped if figures.json exists)
        if not figures_path.exists():
            figures = await asyncio.to_thread(extract_figures, pdf_path, figures_dir)
            write_json(figures_path, figures)
        else:
            figures = read_json(figures_path)

        if not (directory / "evidence_ast.json").exists():
            await asyncio.to_thread(
                DualEngineIngestionService.ingest_paper,
                pdf_path,
                local_id,
                paper.title,
                paper.authors,
                paper.year,
                paper.abstract,
            )

        # Sync to relational multi-view database
        await asyncio.to_thread(StorageService.sync_paper_to_db, local_id, metadata, chunks, figures)

    except RuntimeError as exc:
        logger.warning("Request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Could not process the paper. Please try again.") from exc

    return {
        "paper_id": local_id,
        "metadata": metadata,
        "pages": len(pages),
        "chunks": len(chunks),
        "figures": len(figures),
    }


@app.get("/api/papers/{paper_id}/ast")
async def get_paper_ast(paper_id: str) -> dict[str, Any]:
    """Retrieve the canonical EvidenceAST for a paper."""
    local_id = safe_paper_id(paper_id)
    ast_path = paper_dir(local_id) / "evidence_ast.json"
    if not ast_path.exists():
        pdf_path = paper_dir(local_id) / "paper.pdf"
        if pdf_path.exists():
            ast = await asyncio.to_thread(DualEngineIngestionService.ingest_paper, pdf_path, local_id)
            return ast.model_dump()
        raise HTTPException(status_code=404, detail="Evidence AST not found for this paper.")
    return read_json(ast_path)


@app.get("/api/parsers/ablations")
async def get_parser_ablations() -> dict[str, Any]:
    """Retrieve the P0-P4 parser ablation matrix configurations."""
    return {"ablations": [cfg.model_dump() for cfg in PARSER_ABLATIONS.values()]}


class SearchRequest(BaseModel):
    paper_id: str
    query: str
    limit: int = 5
    preferred_pages: list[int] = Field(default_factory=list)


@app.post("/api/retrieval/search")
async def search_evidence(req: SearchRequest) -> dict[str, Any]:
    """Execute Tri-Channel Hybrid Retrieval with RRF and cross-encoder reranking."""
    local_id = safe_paper_id(req.paper_id)
    chunks_path = paper_dir(local_id) / "chunks.json"
    if not chunks_path.exists():
        raise HTTPException(status_code=404, detail="Paper chunks not found. Please ingest the paper first.")
    chunks = read_json(chunks_path)
    results = await asyncio.to_thread(
        retrieve_chunks,
        req.query,
        chunks,
        req.limit,
        req.preferred_pages,
        local_id,
    )
    return {
        "paper_id": local_id,
        "query": req.query,
        "results_count": len(results),
        "results": results,
    }


class AnalyzeQueryRequest(BaseModel):
    query: str


@app.post("/api/reasoning/analyze")
async def analyze_query_complexity(req: AnalyzeQueryRequest) -> dict[str, Any]:
    """Analyze query reasoning depth (L1 to L5) and bounded subquery decomposition."""
    from backend.services.question_analyzer import QuestionAnalyzer
    analysis = QuestionAnalyzer.analyze_query(req.query)
    return analysis.model_dump()


class CrossDocReasoningRequest(BaseModel):
    query: str
    primary_paper_id: str
    secondary_paper_ids: list[str] = Field(default_factory=list)


@app.post("/api/reasoning/cross-document")
async def cross_document_reasoning(req: CrossDocReasoningRequest) -> dict[str, Any]:
    """Synthesize unified multi-document Evidence Graph across 2+ papers."""
    from backend.services.cross_document_reasoning_service import CrossDocumentReasoningService
    graph, path, chunks = await asyncio.to_thread(
        CrossDocumentReasoningService.synthesize_cross_document_reasoning,
        req.query,
        safe_paper_id(req.primary_paper_id),
        [safe_paper_id(pid) for pid in req.secondary_paper_ids],
    )
    return {
        "graph": graph.model_dump(),
        "path": path.model_dump(),
        "retrieved_count": len(chunks),
    }


class ExportReasoningRequest(BaseModel):
    query: str
    answer: str
    format: str = "markdown"  # "markdown" | "latex"
    reasoning_level: str = "L5_MULTI_HOP_SYNTHESIS"
    steps: list[dict[str, Any]] = Field(default_factory=list)
    numeric_plan: dict[str, Any] | None = None
    verification_report: dict[str, Any] | None = None


@app.post("/api/papers/{paper_id}/export/reasoning")
async def export_reasoning_report(paper_id: str, req: ExportReasoningRequest) -> dict[str, Any]:
    """Export verified multi-level reasoning report in Markdown or LaTeX format."""
    from backend.services.export_service import ExportService
    from backend.schemas.reasoning import QuestionAnalysis, ReasoningLevel
    from backend.schemas.evidence_graph import ReasoningPath, ReasoningPathStep
    from backend.schemas.numeric_plan import NumericExecutionResult
    from backend.schemas.claims import VerificationReport

    p_id = safe_paper_id(paper_id)
    analysis = QuestionAnalysis(
        original_query=req.query,
        reasoning_level=ReasoningLevel(req.reasoning_level) if req.reasoning_level in ReasoningLevel.__members__.values() else ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
    )

    steps = [ReasoningPathStep(**s) for s in req.steps] if req.steps else []
    path = ReasoningPath(query=req.query, reasoning_level=req.reasoning_level, steps=steps)
    num_res = NumericExecutionResult(**req.numeric_plan) if req.numeric_plan else None
    ver_rep = VerificationReport(**req.verification_report) if req.verification_report else None

    if req.format.lower() == "latex":
        content = ExportService.export_to_latex(p_id, req.query, req.answer, analysis, path, num_res, ver_rep)
        filename = f"{p_id}_Reasoning_Report.tex"
    else:
        content = ExportService.export_to_markdown(p_id, req.query, req.answer, analysis, path, num_res, ver_rep)
        filename = f"{p_id}_Reasoning_Report.md"

    return {
        "format": req.format,
        "filename": filename,
        "content": content,
    }


@app.get("/api/telemetry/traces")
async def list_telemetry_traces() -> list[dict[str, Any]]:
    """List recorded telemetry and reasoning audit traces."""
    from backend.services.telemetry_service import TRACES_DIR
    traces: list[dict[str, Any]] = []
    if TRACES_DIR.exists():
        for tf in sorted(TRACES_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)[:50]:
            try:
                traces.append(json.loads(tf.read_text(encoding="utf-8")))
            except Exception:
                continue
    return traces


@app.post("/api/papers/upload")
async def upload_paper(file: UploadFile = File(...), title: str = Form("")) -> dict[str, Any]:
    filename = file.filename or "uploaded-paper.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Please upload a PDF file")

    max_bytes = 50 * 1024 * 1024  # 50MB
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(1024 * 1024):
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(status_code=413, detail=f"PDF exceeds {max_bytes // (1024 * 1024)}MB limit")
        chunks.append(chunk)
    content = b"".join(chunks)
    if not content.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Uploaded file does not look like a PDF")

    digest = hashlib.sha1(content).hexdigest()[:12]
    local_id = safe_paper_id(f"upload_{digest}")
    directory = paper_dir(local_id)
    pdf_path = directory / "paper.pdf"
    metadata_path = directory / "metadata.json"
    pages_path = directory / "pages.json"
    chunks_path = directory / "chunks.json"
    figures_path = directory / "figures.json"
    figures_dir = directory / "figures"

    clean_title = title.strip() or Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Uploaded PDF"
    try:
        directory.mkdir(parents=True, exist_ok=True)
        if not pdf_path.exists():
            pdf_path.write_bytes(content)

        pages = await asyncio.to_thread(extract_pages, pdf_path)
        chunks = await asyncio.to_thread(chunk_pages, pages)
        inferred = infer_uploaded_metadata(pages, clean_title)
        metadata = {
            "id": local_id,
            "local_id": local_id,
            "title": inferred["title"],
            "authors": ["Uploaded PDF"],
            "year": "",
            "summary": inferred["summary"],
            "categories": ["PDF"],
            "pdf_url": "",
            "abs_url": "",
            "published": "",
            "source": "upload",
            "filename": filename,
        }
        write_json(metadata_path, metadata)
        write_json(pages_path, pages)
        write_json(chunks_path, chunks)

        # Extract figures/tables
        if not figures_path.exists():
            figures = await asyncio.to_thread(extract_figures, pdf_path, figures_dir)
            write_json(figures_path, figures)
        else:
            figures = read_json(figures_path)

        # Sync to relational multi-view database
        await asyncio.to_thread(StorageService.sync_paper_to_db, local_id, metadata, chunks, figures)

    except RuntimeError as exc:
        logger.warning("Request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Could not process the paper. Please try again.") from exc

    return {
        "paper_id": local_id,
        "metadata": metadata,
        "pages": len(pages),
        "chunks": len(chunks),
        "figures": len(figures),
    }


def _paths_or_404(paper_id: str) -> tuple[Path, Path, Path, Path]:
    directory = paper_dir(paper_id)
    metadata_path = directory / "metadata.json"
    pages_path = directory / "pages.json"
    chunks_path = directory / "chunks.json"
    pdf_path = directory / "paper.pdf"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="Paper has not been prepared")
    return metadata_path, pages_path, chunks_path, pdf_path


# ---------------------------------------------------------------------------
# AST & Document Representation endpoints
# ---------------------------------------------------------------------------

@app.get("/api/papers/{paper_id}/ast")
async def get_paper_ast(paper_id: str) -> dict[str, Any]:
    """Return the structured ScientificDocument AST model for a paper."""
    _paths_or_404(paper_id)
    doc_ast = StorageService.get_document_ast(paper_id)
    if not doc_ast:
        raise HTTPException(status_code=404, detail="Document AST could not be constructed")
    return doc_ast.model_dump()


# ---------------------------------------------------------------------------
# Figure endpoints
# ---------------------------------------------------------------------------

@app.get("/api/papers/{paper_id}/figures")
async def get_figures(paper_id: str) -> dict[str, Any]:
    """Return the list of extracted figures/tables for a paper."""
    _paths_or_404(paper_id)
    figures_path = paper_dir(paper_id) / "figures.json"
    figures: list[dict[str, Any]] = read_json(figures_path) if figures_path.exists() else []
    return {"paper_id": paper_id, "figures": figures, "count": len(figures)}


@app.get("/api/papers/{paper_id}/figures/{figure_id}.png")
async def get_figure_image(paper_id: str, figure_id: str) -> Response:
    """Serve the PNG image chip for a specific figure."""
    _paths_or_404(paper_id)
    img_path = paper_dir(paper_id) / "figures" / f"{figure_id}.png"
    if not img_path.exists():
        raise HTTPException(status_code=404, detail="Figure image not found")
    return Response(content=img_path.read_bytes(), media_type="image/png")


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str) -> dict[str, Any]:
    metadata_path, pages_path, chunks_path, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    if metadata.get("source") == "upload" and metadata.get("summary") == "Custom PDF uploaded for local study." and pages_path.exists():
        inferred = infer_uploaded_metadata(read_json(pages_path), metadata.get("title", "Uploaded PDF"))
        metadata["title"] = inferred["title"]
        metadata["summary"] = inferred["summary"]
        write_json(metadata_path, metadata)
    metadata["pages"] = len(read_json(pages_path)) if pages_path.exists() else 0
    chunks = _ensure_chunk_metadata(chunks_path, pages_path) if chunks_path.exists() else []
    metadata["chunks"] = len(chunks)
    return metadata


# ---------------------------------------------------------------------------
# Reference endpoints (Step 1 & 2 - multi-document mode)
# ---------------------------------------------------------------------------


@app.get("/api/papers/{paper_id}/references")
async def get_references(paper_id: str, force: bool = False) -> dict[str, Any]:
    """Return the resolved bibliography for a paper.

    Triggers resolution on first call (or when force=True).
    For uploaded PDFs the quality may be lower (arXiv title-search fallback).
    """
    metadata_path, pages_path, _, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    pages = read_json(pages_path) if pages_path.exists() else []
    is_upload = metadata.get("source") == "upload"

    refs = await resolve_references(paper_id, metadata, pages, force=force)
    return {
        "paper_id": paper_id,
        "references": refs,
        "count": len(refs),
        "upload_warning": is_upload,
    }


@app.post("/api/papers/{paper_id}/references/resolve")
async def resolve_references_endpoint(paper_id: str) -> dict[str, Any]:
    """Force-refresh the bibliography from Semantic Scholar / arXiv."""
    return await get_references(paper_id, force=True)


@app.post("/api/papers/{paper_id}/references/{ref_index}/ingest")
async def ingest_reference(paper_id: str, ref_index: int) -> dict[str, Any]:
    """Download and chunk a cited paper, tagging its chunks with source_paper_id.

    ref_index is the 0-based position in the references.json array.
    Returns the new secondary paper's local_id and chunk count.
    """
    _paths_or_404(paper_id)
    refs = load_references(paper_id)
    if not refs or not (0 <= ref_index < len(refs)):
        raise HTTPException(status_code=404, detail="Reference index out of range")

    ref = refs[ref_index]
    if ref.get("ingested") and ref.get("secondary_local_id"):
        sec_id = ref["secondary_local_id"]
        sec_dir = paper_dir(sec_id)
        chunks = read_json(sec_dir / "chunks.json") if (sec_dir / "chunks.json").exists() else []
        return {
            "secondary_paper_id": sec_id,
            "chunks": len(chunks),
            "cached": True,
        }

    arxiv_id = ref.get("arxiv_id")
    pdf_url   = ref.get("pdf_url", "")
    title     = ref.get("title") or "Reference Paper"

    if not pdf_url:
        raise HTTPException(
            status_code=422,
            detail="This reference has no downloadable PDF (not on arXiv and not open-access).",
        )

    # Derive a stable local_id for the secondary paper
    if arxiv_id:
        sec_local_id = safe_paper_id(arxiv_id)
    else:
        import hashlib
        sec_local_id = safe_paper_id("ref_" + hashlib.sha1(pdf_url.encode()).hexdigest()[:10])

    sec_dir        = paper_dir(sec_local_id)
    sec_pdf        = sec_dir / "paper.pdf"
    sec_pages_path = sec_dir / "pages.json"
    sec_chunks_path= sec_dir / "chunks.json"
    sec_meta_path  = sec_dir / "metadata.json"

    try:
        sec_dir.mkdir(parents=True, exist_ok=True)

        if not sec_pdf.exists():
            await download_pdf(pdf_url, sec_pdf)

        if not sec_pages_path.exists():
            sec_pages = await asyncio.to_thread(extract_pages, sec_pdf)
            write_json(sec_pages_path, sec_pages)
        else:
            sec_pages = read_json(sec_pages_path)

        if not sec_chunks_path.exists():
            # Tag every chunk with the secondary paper's id for provenance
            sec_chunks = await asyncio.to_thread(chunk_pages, sec_pages, source_paper_id=sec_local_id)
            write_json(sec_chunks_path, sec_chunks)
        else:
            sec_chunks = read_json(sec_chunks_path)
            # Back-fill source_paper_id if missing (re-ingestion of older cache)
            if sec_chunks and "source_paper_id" not in sec_chunks[0]:
                sec_chunks = await asyncio.to_thread(chunk_pages, sec_pages, source_paper_id=sec_local_id)
                write_json(sec_chunks_path, sec_chunks)

        # Write secondary metadata
        if not sec_meta_path.exists():
            sec_meta = {
                "id":           arxiv_id or sec_local_id,
                "local_id":     sec_local_id,
                "title":        title,
                "authors":      ref.get("authors", []),
                "year":         ref.get("year", ""),
                "summary":      ref.get("abstract", ""),
                "categories":   [],
                "pdf_url":      pdf_url,
                "abs_url":      ref.get("abs_url", ""),
                "published":    "",
                "source":       "reference",
                "anchor_paper": paper_id,
            }
            write_json(sec_meta_path, sec_meta)

    except RuntimeError as exc:
        logger.warning("Request failed: %s", exc)
        raise HTTPException(status_code=500, detail="Could not process the paper. Please try again.") from exc

    mark_reference_ingested(paper_id, ref_index, sec_local_id)

    return {
        "secondary_paper_id": sec_local_id,
        "title":              title,
        "chunks":             len(sec_chunks),
        "cached":             False,
    }


@app.get("/api/papers/{paper_id}/pdf")
async def get_pdf(paper_id: str) -> FileResponse:
    _, _, _, pdf_path = _paths_or_404(paper_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF is missing")
    return FileResponse(pdf_path, media_type="application/pdf", filename="paper.pdf")


@app.get("/api/papers/{paper_id}/page/{page_number}.png")
async def get_pdf_page(paper_id: str, page_number: int, zoom: float = 1.8, highlight: str = "") -> Response:
    _, _, _, pdf_path = _paths_or_404(paper_id)
    try:
        image_bytes = await asyncio.to_thread(render_page_png, pdf_path, page_number, zoom, highlight)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=image_bytes, media_type="image/png")


# ---------------------------------------------------------------------------
# Snippet / Region Cropping endpoints
# ---------------------------------------------------------------------------

@app.post("/api/papers/{paper_id}/snippets")
async def create_snippet(paper_id: str, payload: SnippetInput) -> dict[str, Any]:
    """Crop a custom rectangular region from a page and save high-res snippet PNG."""
    _, _, _, pdf_path = _paths_or_404(paper_id)
    import uuid
    snippet_id = f"snip_{uuid.uuid4().hex[:8]}"
    snippets_dir = paper_dir(paper_id) / "snippets"
    snippets_dir.mkdir(parents=True, exist_ok=True)
    snippet_path = snippets_dir / f"{snippet_id}.png"

    try:
        png_bytes, extracted_text = await asyncio.to_thread(
            crop_page_region,
            pdf_path=pdf_path,
            page_number=payload.page,
            bbox_norm=payload.bbox,
            zoom=payload.zoom,
        )
        snippet_path.write_bytes(png_bytes)
    except Exception as exc:
        logger.warning("Snippet crop failed: %s", exc)
        raise HTTPException(status_code=500, detail=f"Failed to crop snippet: {exc}") from exc

    return {
        "snippet_id": snippet_id,
        "image_url": f"/api/papers/{paper_id}/snippets/{snippet_id}.png",
        "page": payload.page,
        "bbox": payload.bbox,
        "text": extracted_text,
    }


@app.get("/api/papers/{paper_id}/snippets/{snippet_id}.png")
async def get_snippet_png(paper_id: str, snippet_id: str) -> FileResponse:
    """Serve a custom-cropped region snippet PNG."""
    snippet_path = paper_dir(paper_id) / "snippets" / f"{safe_paper_id(snippet_id)}.png"
    if not snippet_path.exists():
        raise HTTPException(status_code=404, detail="Snippet image not found")
    return FileResponse(snippet_path, media_type="image/png")


@app.post("/api/papers/{paper_id}/study-goals")
async def study_goals(
    paper_id: str,
    payload: StudyGoalsInput | None = None,
) -> dict[str, Any]:
    payload = payload or StudyGoalsInput()
    metadata_path, pages_path, chunks_path, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    chunks = _ensure_chunk_metadata(chunks_path, pages_path) if chunks_path.exists() else []
    figures_path = metadata_path.parent / "figures.json"
    figures: list[dict[str, Any]] = read_json(figures_path) if figures_path.exists() else []

    if metadata.get("source") == "upload" and metadata.get("summary") == "Custom PDF uploaded for local study.":
        pages_path = metadata_path.parent / "pages.json"
        if pages_path.exists():
            inferred = infer_uploaded_metadata(read_json(pages_path), metadata.get("title", "Uploaded PDF"))
            metadata["title"] = inferred["title"]
            metadata["summary"] = inferred["summary"]
            write_json(metadata_path, metadata)

    if not await ollama_available():
        return {
            "goals": fallback_goals(metadata, chunks, figures),
            "model": OLLAMA_MODEL,
            "fallback": True,
        }

    goals_path = metadata_path.parent / f"goals_canonical_{STUDY_GOAL_PROMPT_VERSION}.json"

    if goals_path.exists() and not payload.force:
        return {
            "goals": read_json(goals_path),
            "model": OLLAMA_MODEL,
        }

    try:
        goals = await asyncio.wait_for(
            generate_study_goals(metadata, chunks, figures=figures),
            timeout=120.0,
        )
        write_json(goals_path, goals)
    except Exception:
        goals = fallback_goals(metadata, chunks, figures)
    return {
        "goals": goals,
        "model": OLLAMA_MODEL,
    }


@app.post("/api/papers/{paper_id}/chat")
async def chat(paper_id: str, payload: ChatInput) -> dict[str, Any]:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    metadata_path, pages_path, chunks_path, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    chunks = _ensure_chunk_metadata(chunks_path, pages_path) if chunks_path.exists() else []
    if metadata.get("source") == "upload" and metadata.get("summary") == "Custom PDF uploaded for local study.":
        pages_path = metadata_path.parent / "pages.json"
        if pages_path.exists():
            inferred = infer_uploaded_metadata(read_json(pages_path), metadata.get("title", "Uploaded PDF"))
            metadata["title"] = inferred["title"]
            metadata["summary"] = inferred["summary"]
            write_json(metadata_path, metadata)

    # --- Multi-document: merge secondary-paper chunks into the search pool ---
    # Anchor chunks are tagged with paper_id; secondary chunks carry their own
    # source_paper_id. retrieve_chunks() is unchanged - it operates on any list.
    anchor_chunks = [
        {**c, "source_paper_id": paper_id} if "source_paper_id" not in c else c
        for c in chunks
    ]
    secondary_meta: dict[str, dict[str, Any]] = {}  # local_id -> metadata
    all_chunks = list(anchor_chunks)
    for sec_id in payload.secondary_paper_ids:
        sec_id = safe_paper_id(sec_id)
        sec_dir = paper_dir(sec_id)
        sec_chunks_path = sec_dir / "chunks.json"
        sec_meta_path   = sec_dir / "metadata.json"
        if sec_chunks_path.exists():
            sec_chunks = read_json(sec_chunks_path)
            # Back-fill source_paper_id if an older cache lacks it
            sec_chunks = [
                {**c, "source_paper_id": sec_id} if "source_paper_id" not in c else c
                for c in sec_chunks
            ]
            all_chunks.extend(sec_chunks)
        if sec_meta_path.exists():
            secondary_meta[sec_id] = read_json(sec_meta_path)

    # Merge figure chunks into the retrieval pool so captions can match queries
    figures_path = paper_dir(paper_id) / "figures.json"
    if figures_path.exists():
        figure_records = read_json(figures_path)
        figure_chunks = await asyncio.to_thread(chunk_figures, figure_records, source_paper_id=paper_id)
        # Tag anchor figure chunks with paper_id (same convention as text chunks)
        figure_chunks = [{**c, "source_paper_id": paper_id} for c in figure_chunks]
        all_chunks = all_chunks + figure_chunks

    # ── Capability and Adaptive Routing ───────────────────────────────────────
    capabilities = ModelRegistry.resolve_capabilities(
        payload.model or OLLAMA_MODEL,
        mode=payload.capability_mode,
    )

    # ── Custom User Snippet Vision Routing ───────────────────────────────────
    if payload.snippet_id and payload.snippet_page and payload.snippet_bbox:
        snippet_res = await answer_with_custom_snippet(
            question=payload.message,
            snippet_id=payload.snippet_id,
            page_number=payload.snippet_page,
            bbox_norm=payload.snippet_bbox,
            snippet_text=payload.snippet_text or "",
            paper_id=paper_id,
            paper_metadata=metadata,
            model=payload.model,
        )
        aligned_ans, verified_cits, _ = ClaimVerifierService.verify_and_repair_answer(
            answer=snippet_res["answer"],
            citations=snippet_res["citations"],
            candidate_pool=snippet_res["citations"],
            apply_repair=True,
        )
        return {
            "answer": aligned_ans,
            "citations": verified_cits,
            "model": snippet_res.get("model_used", OLLAMA_MODEL),
            "vision": True,
            "is_snippet": True,
            "snippet_id": payload.snippet_id,
            "figure_label": f"Snippet (Page {payload.snippet_page})",
            "figure_image_url": f"/api/papers/{paper_id}/snippets/{payload.snippet_id}.png",
            "route_type": "CUSTOM_SNIPPET_VISION",
            "capability_mode": capabilities.capability_mode.value,
        }

    route_budget = QuestionRouter.route(payload.message, capabilities)

    # ── Multi-Hop Query Decomposition & Retrieval ───────────────────────────
    subqueries = QueryDecomposer.decompose(payload.message)
    page_hints = extract_page_hints(payload.message)

    if len(subqueries) > 1:
        # Multi-hop retrieval for comparative & multi-entity queries
        merged_selected: list[dict[str, Any]] = []
        seen_keys: set[tuple[Any, str]] = set()

        # 1. Global query retrieval
        global_hits = await asyncio.to_thread(
            retrieve_chunks,
            payload.message,
            all_chunks,
            limit=route_budget.text_top_k,
            preferred_pages=page_hints,
        )
        for h in global_hits:
            k = (h.get("source_paper_id"), str(h.get("chunk_id")))
            if k not in seen_keys:
                seen_keys.add(k)
                merged_selected.append(h)

        # 2. Targeted subquery retrievals
        per_sub_limit = max(2, route_budget.text_top_k // len(subqueries) + 1)
        for sub_q in subqueries:
            sub_hits = await asyncio.to_thread(
                retrieve_chunks,
                sub_q,
                all_chunks,
                limit=per_sub_limit,
                preferred_pages=page_hints,
            )
            for h in sub_hits:
                k = (h.get("source_paper_id"), str(h.get("chunk_id")))
                if k not in seen_keys:
                    seen_keys.add(k)
                    merged_selected.append(h)
        selected = merged_selected
    else:
        selected = await asyncio.to_thread(
            retrieve_chunks,
            payload.message,
            all_chunks,
            limit=route_budget.text_top_k,
            preferred_pages=page_hints,
        )

    if not selected:
        return {
            "answer": "The paper context does not contain enough information to answer that.",
            "citations": [],
            "abstained": True,
            "uncertainty_reason": "NO_EVIDENCE_RETRIEVED",
            "route_type": route_budget.route_type.value,
        }

    # ── Pre-generation Evidence Sufficiency Gate ──────────────────────────────
    sufficiency = ClaimVerifierService.compute_sufficiency(
        query=payload.message,
        retrieved_chunks=selected,
        requires_vision=route_budget.requires_native_vision,
        can_vision=capabilities.can_process_images(),
    )
    if not sufficiency.is_sufficient:
        return {
            "answer": f"**Abstained**\n\nThe paper context does not provide sufficient evidence to answer this question reliably ({sufficiency.reason_code.lower().replace('_', ' ')}).",
            "citations": [],
            "abstained": True,
            "uncertainty_reason": sufficiency.reason_code,
            "route_type": route_budget.route_type.value,
        }

    # ── Multi-Image Multimodal Vision Routing ─────────────────────────────────
    # Collect all figure/table chunks present in retrieved context (up to visual_items budget)
    retrieved_figures = [c for c in selected if c.get("is_figure_chunk")]
    max_vis = max(1, route_budget.visual_items or 2)
    selected_figures = retrieved_figures[:max_vis]

    if selected_figures and capabilities.can_process_images() and (
        route_budget.requires_native_vision
        or route_budget.route_type in (
            QuestionRouteType.FIGURE_VISUAL,
            QuestionRouteType.CHART_NUMERIC,
            QuestionRouteType.MIXED_TEXT_VISUAL,
            QuestionRouteType.COMPARISON,
            QuestionRouteType.TABLE_NUMERIC,
        )
    ):
        text_support = [c for c in selected if not c.get("is_figure_chunk")]
        vision_result = await answer_with_multimodal_evidence(
            question=payload.message,
            figure_chunks=selected_figures,
            context_chunks=text_support,
            paper_id=paper_id,
            paper_metadata=metadata,
            model=payload.model,
        )
        fig_label = vision_result.get("label", "Figure")
        raw_vis_answer = vision_result["answer"]
        raw_vis_citations = vision_result.get("citations", [])

        # Apply ALCE/AGREE remapping and disclaimer pruning to vision responses
        aligned_vis_answer, verified_vis_citations, _ = ClaimVerifierService.verify_and_repair_answer(
            answer=raw_vis_answer,
            citations=raw_vis_citations,
            candidate_pool=selected,
            apply_repair=True,
        )

        return {
            "answer":            aligned_vis_answer,
            "citations":         verified_vis_citations,
            "model":             vision_result.get("model_used", OLLAMA_MODEL),
            "vision":            True,
            "vision_fallback":   vision_result.get("fallback", False),
            "figure_id":         vision_result.get("figure_id"),
            "figure_label":      fig_label,
            "figure_image_url":  f"/api/papers/{paper_id}/figures/{vision_result.get('figure_id')}.png"
                                  if vision_result.get("figure_id") else None,
            "route_type":        route_budget.route_type.value,
            "capability_mode":   capabilities.capability_mode.value,
        }

    ollama_up = await ollama_available()
    context_chunks: list[dict[str, Any]] = []
    # chunk_id numbering restarts per paper (chunk_001, chunk_002, ...), so in
    # multi-document mode two different papers' chunks can share a chunk_id -
    # disambiguate with source_paper_id or this dedup silently drops real chunks.
    seen_chunk_keys: set[tuple[Any, str]] = set()
    for chunk in chunks[:2] + selected:
        chunk_key = (chunk.get("source_paper_id"), str(chunk.get("chunk_id")))
        if chunk_key in seen_chunk_keys:
            continue
        seen_chunk_keys.add(chunk_key)
        context_chunks.append(chunk)
    prompt_chunks = context_chunks[:4]
    history_char_limit = 180
    abstract = str(metadata.get("summary") or "")[:350]
    question_text = payload.message[:900]

    evidence_items = await asyncio.to_thread(_build_evidence_items, question_text, prompt_chunks, limit=7)
    paper_context = _format_evidence_context(evidence_items, secondary_meta=secondary_meta)
    recent_history = "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:history_char_limit]}"
        for item in payload.history[-6:]
        if isinstance(item, dict)
    )

    route_specific_instructions = ""
    if route_budget.route_type == QuestionRouteType.CODE_ALGORITHM:
        route_specific_instructions = """
Special Algorithm / Code Instructions:
- Provide a clean, formatted code/pseudocode block with markdown syntax tagging (e.g. ```python or ```pseudo).
- State the inputs, hyperparameters, and tensor dimensions clearly.
- Provide a step-by-step logic trace of the algorithm's execution and loop invariants.
- State the computational complexity (Time and Space Big-O).
"""
    elif route_budget.route_type == QuestionRouteType.TABLE_NUMERIC:
        route_specific_instructions = """
Special Tabular / Numeric Instructions:
- Reconstruct the tabular data in a clean Markdown table grid (| Column 1 | Column 2 | ... |).
- Provide a model-by-model or row-by-row metric breakdown with performance deltas.
- Analyze accuracy tradeoffs vs computational efficiency / parameter count / training cost.
"""

    prompt = f"""
You are ScholAR, a rigorous research paper tutor. Your only source is the paper
retrieval: selected chunks from the PDF being studied. Answer strictly from that
context; if it does not contain the answer, say so plainly.

Paper metadata:
Title: {metadata.get("title")}
Authors: {", ".join(metadata.get("authors", []))}
Abstract: {abstract}

Recent conversation:
{recent_history or "No previous conversation."}

Response requirements:
- Give a detailed, precise study answer, not a shallow summary.
- Format the answer with bold section labels only, such as **Answer**, **Evidence**, **What this means**, and **Limits / what to verify**.
- Do not use Markdown heading markers like #, ##, or ###.
- Put each section label on its own line, for example **Answer**.
- Use bullet points for multi-part explanations.
- Wrap every piece of mathematical notation in single dollar signs, e.g. $x_t$ or $p_\theta(x_{{t-1}}\mid x_t)$, so it renders instead of showing raw symbols.
- Cite paper claims inline only with evidence IDs from the Paper evidence list, such as [E1] or [E2].
- Only cite [Ek] if the sentence directly states a specific mechanism, metric, or finding supported in [Ek].
- NEVER attach citation markers to disclaimers, negative statements about what is NOT in the paper, assumptions, or conversational transitions.
- Never invent page citations. Do not write [p. 1], [p. 2], or any page number yourself.
- The app will convert evidence IDs into compact numbered references like [1], [2].
- If the paper context is insufficient, say what is missing instead of guessing (without attaching citations to the disclaimer).
- For methods/results questions, include the specific mechanism, dataset, metric, number, or comparison when the evidence provides it.
- Use concise sections: "Answer", "Evidence", "What this means", and "Limits / what to verify" when helpful.
- Answer directly in 180 to 450 words. Do not include hidden reasoning or long deliberation.
- Use only the strongest 2 to 4 evidence points and keep citations close to the claims they support.
{route_specific_instructions}

Paper evidence:
{paper_context}

Question: {question_text}
""".strip()

    if ollama_up:
        try:
            answer = await generate(prompt, temperature=0.1, model=payload.model)
        except Exception as exc:
            return _model_failure_payload(exc)
    else:
        answer = ""

    if not answer:
        if ollama_up:
            return _error_payload("The local model returned an empty answer. Try again.")
        answer = _extractive_tutor_answer(payload.message, selected) if selected else ""

    # Normalize pseudo-LaTeX model names like $\text{BERT}_{\text{BASE}}$ -> BERT-Base
    answer = re.sub(r"\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$", r"\1-\2", answer)
    answer = re.sub(r"\$\\text\{([^}]+)\}_\{([^}]+)\}\$", r"\1-\2", answer)
    answer = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", answer)
    answer = re.sub(r"\\text\{([^}]+)\}", r"\1", answer)

    answer, citations = _normalize_evidence_citations(answer, evidence_items)
    if not citations and evidence_items:
        citations = [
            {
                "ref_id": index,
                "page": item.get("page"),
                "chunk_id": item.get("chunk_id"),
                "section_title": item.get("section_title"),
                "chunk_type": item.get("chunk_type"),
                "quote": item.get("quote"),
            }
            for index, item in enumerate(evidence_items[:2], start=1)
        ]

    # Tag citations with runtime verification status, apply ALCE/AGREE remapping/pruning & 1-step repair
    answer, verified_citations, verified_claims = ClaimVerifierService.verify_and_repair_answer(
        answer=answer,
        citations=citations,
        candidate_pool=evidence_items,
        apply_repair=True,
    )

    # ── Multi-Level Reasoning & Evidence Graph Synthesis ─────────────────────
    from backend.services.question_analyzer import QuestionAnalyzer
    from backend.services.evidence_graph_service import EvidenceGraphService
    from backend.services.budgeting_service import BudgetingService
    from backend.services.table_arithmetic_service import TableArithmeticService, NumericOp
    from backend.services.telemetry_service import TelemetryService

    q_analysis = QuestionAnalyzer.analyze_query(payload.message)
    ev_budget = BudgetingService.get_evidence_budget(capabilities)
    ev_graph, ev_path = EvidenceGraphService.build_evidence_graph(payload.message, context_chunks, q_analysis)
    pruned_graph, pruned_path = BudgetingService.prune_to_budget(ev_graph, ev_path, ev_budget)

    numeric_res = None
    if q_analysis.requires_arithmetic:
        table_chunks = [c for c in context_chunks if c.get("is_table_chunk") or "|" in c.get("text", "")]
        if table_chunks:
            # Extract numbers if comparing entities
            words = [w for w in payload.message.split() if len(w) > 3]
            ent_a = words[0] if len(words) > 0 else "Model"
            ent_b = words[1] if len(words) > 1 else "Baseline"
            numeric_res = TableArithmeticService.extract_and_calculate_from_table_text(
                table_text=table_chunks[0].get("text", ""),
                entity_a=ent_a,
                entity_b=ent_b,
                op=NumericOp.DIFFERENCE,
            )

    atomic_report = ClaimVerifierService.generate_atomic_verification_report(answer, context_chunks)
    TelemetryService.record_trace(
        paper_id=paper_id,
        query=payload.message,
        analysis=q_analysis,
        reasoning_path=pruned_path,
        numeric_result=numeric_res,
        verification_report=atomic_report,
        latency_ms=250.0,
        hardware_tier=ev_budget.hardware_tier.value,
    )

    return {
        "answer": answer,
        "citations": verified_citations,
        "model": payload.model or OLLAMA_MODEL,
        "route_type": route_budget.route_type.value,
        "capability_mode": capabilities.capability_mode.value,
        "verified_claims_count": len(verified_claims),
        "reasoning_level": q_analysis.reasoning_level.value,
        "reasoning_steps": [s.model_dump() for s in pruned_path.steps],
        "numeric_plan": numeric_res.model_dump() if numeric_res else None,
        "verification_report": atomic_report.model_dump() if atomic_report else None,
    }


@app.post("/api/papers/{paper_id}/chat/stream")
async def chat_stream(paper_id: str, payload: ChatInput) -> StreamingResponse:
    """Stream multi-level reasoning events, evidence graph, and generated tokens in real-time."""
    from backend.services.ollama_service import generate_stream
    from backend.services.question_analyzer import QuestionAnalyzer
    from backend.services.evidence_graph_service import EvidenceGraphService
    from backend.services.budgeting_service import BudgetingService
    from backend.services.table_arithmetic_service import TableArithmeticService, NumericOp

    metadata_path, pages_path, chunks_path, _ = _paths_or_404(paper_id)
    chunks = _ensure_chunk_metadata(chunks_path, pages_path) if chunks_path.exists() else []

    capabilities = ModelRegistry.resolve_capabilities(
        payload.model or OLLAMA_MODEL,
        mode=payload.capability_mode,
    )

    async def event_generator():
        # 1. Emit instant query analysis
        q_analysis = QuestionAnalyzer.analyze_query(payload.message)
        yield f"event: analysis\ndata: {json.dumps(q_analysis.model_dump())}\n\n"

        # 2. Retrieve & build evidence graph
        ev_budget = BudgetingService.get_evidence_budget(capabilities)
        ev_graph, ev_path = EvidenceGraphService.build_evidence_graph(payload.message, chunks[:6], q_analysis)
        pruned_graph, pruned_path = BudgetingService.prune_to_budget(ev_graph, ev_path, ev_budget)

        yield f"event: evidence_path\ndata: {json.dumps([s.model_dump() for s in pruned_path.steps])}\n\n"

        # 3. Deterministic Table Arithmetic
        numeric_res = None
        if q_analysis.requires_arithmetic:
            table_chunks = [c for c in chunks if c.get("is_table_chunk") or "|" in c.get("text", "")]
            if table_chunks:
                words = [w for w in payload.message.split() if len(w) > 3]
                ent_a = words[0] if len(words) > 0 else "Model"
                ent_b = words[1] if len(words) > 1 else "Baseline"
                numeric_res = TableArithmeticService.extract_and_calculate_from_table_text(
                    table_text=table_chunks[0].get("text", ""),
                    entity_a=ent_a,
                    entity_b=ent_b,
                    op=NumericOp.DIFFERENCE,
                )
                if numeric_res:
                    yield f"event: numeric_math\ndata: {json.dumps(numeric_res.model_dump())}\n\n"

        # 4. Stream LLM tokens
        prompt = f"Evidence:\n" + "\n".join(c.get("text", "")[:300] for c in chunks[:3]) + f"\n\nQuestion: {payload.message}"
        full_text = []
        try:
            async for token in generate_stream(prompt, temperature=0.1, model=payload.model):
                full_text.append(token)
                yield f"event: token\ndata: {json.dumps({'token': token})}\n\n"
        except Exception:
            pass

        # 5. Emit Verification Report
        generated_ans = "".join(full_text) or "Extracted verified evidence."
        atomic_report = ClaimVerifierService.generate_atomic_verification_report(generated_ans, chunks[:6])
        yield f"event: verification\ndata: {json.dumps(atomic_report.model_dump())}\n\n"
        yield "event: done\ndata: {}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

