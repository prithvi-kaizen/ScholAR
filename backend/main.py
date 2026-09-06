from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from pydantic import BaseModel, Field

from backend.schemas.answer_trace import (
    AnswerPipelineRequest,
    DecodingOptions,
    EvaluationContext,
    ExecutionPolicy,
    InterventionControls,
)
from backend.schemas.capabilities import CapabilityMode, ModelRegistry
from backend.schemas.evidence import EvidenceAST
from backend.services import answer_pipeline as answer_pipeline_module
from backend.services.answer_pipeline import AnswerPipelineService
from backend.services.arxiv_service import search_arxiv
from backend.services.chunking_service import chunk_pages
from backend.services.ingestion_service import PARSER_ABLATIONS
from backend.services.network_policy_service import (
    NetworkPolicyError,
    NetworkPolicyService,
    NetworkPolicyStatus,
)
from backend.services.paper_finalize_service import PaperFinalizeService
from backend.services.ollama_service import (
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    fallback_goals,
    generate_study_goals,
    ollama_available,
    STUDY_GOAL_PROMPT_VERSION,
)
from backend.services.pdf_service import (
    atomic_write_bytes,
    crop_page_region,
    download_pdf,
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
from backend.services.retrieval_service import retrieve_chunks
from backend.services.storage_service import StorageService


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


@app.exception_handler(NetworkPolicyError)
async def network_policy_error_handler(_: Request, exc: NetworkPolicyError) -> JSONResponse:
    """Return an actionable policy error instead of hiding acquisition denial as a server fault."""
    return JSONResponse(status_code=409, content={"detail": exc.to_dict()})


class PaperInput(BaseModel):
    id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: str = ""
    summary: str = ""
    abstract: str = ""
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
    # Release evaluations record and forward a non-negative Ollama seed.
    generation_seed: int | None = Field(default=None, ge=0, le=2147483647)
    capability_mode: CapabilityMode = CapabilityMode.AUTO
    execution_policy: ExecutionPolicy = ExecutionPolicy.ALLOW_EXTRACTIVE_FALLBACK
    intervention: InterventionControls = Field(default_factory=InterventionControls)
    decoding: DecodingOptions = Field(default_factory=DecodingOptions)
    evaluation_context: EvaluationContext | None = None
    experiment_id: str | None = Field(default=None, max_length=200)
    visual_page_backend: Literal[
        "configured", "auto", "colqwen2", "clip", "disabled"
    ] = "configured"
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


@app.get("/health")
async def health() -> dict[str, Any]:
    policy = NetworkPolicyService.status()
    return {
        "status": "ok",
        "ollama_available": await ollama_available(),
        "model": OLLAMA_MODEL,
        "network_mode": policy.mode.value,
    }


@app.get("/api/system/network-policy", response_model=NetworkPolicyStatus)
async def network_policy_status() -> NetworkPolicyStatus:
    """Expose the active acquisition/analysis boundary and missing local assets."""
    return NetworkPolicyService.status()


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
    cached = await asyncio.to_thread(
        PaperFinalizeService.load_if_complete,
        local_id,
        target_dir=directory,
    )
    if cached is not None:
        return cached

    metadata = paper.model_dump()
    metadata["local_id"] = local_id
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{local_id}.acquire-", dir=str(directory.parent)
        ) as temp_dir:
            acquired_pdf = Path(temp_dir) / "paper.pdf"
            await download_pdf(paper.pdf_url, acquired_pdf)
            return await asyncio.to_thread(
                PaperFinalizeService.finalize,
                acquired_pdf,
                local_id,
                metadata,
                target_dir=directory,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Paper finalization failed for [%s]: %s", local_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Could not process the paper. Please try again.",
        ) from exc


@app.get("/api/papers/{paper_id}/ast", response_model=EvidenceAST)
async def get_paper_ast(paper_id: str) -> EvidenceAST:
    """Return the persisted canonical EvidenceAST without mutating paper state."""
    local_id = safe_paper_id(paper_id)
    ast_path = paper_dir(local_id) / "evidence_ast.json"
    if not ast_path.is_file():
        raise HTTPException(status_code=404, detail="Evidence AST not found for this paper.")
    try:
        return EvidenceAST.model_validate(read_json(ast_path))
    except Exception as exc:
        logger.warning("Invalid EvidenceAST for [%s]: %s", local_id, exc)
        raise HTTPException(status_code=500, detail="Evidence AST is invalid.") from exc


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


@app.get("/api/benchmark/summary")
async def get_benchmark_summary() -> dict[str, Any]:
    """Return benchmark evaluation summary with clear release provenance (audit item A11)."""
    from backend.services.telemetry_service import TRACES_DIR

    repo_root = Path(__file__).resolve().parents[1]
    results_path = repo_root / "evaluation" / "results" / "method_comparison_ablation.json"
    empirical_results: dict[str, Any] | None = None
    provenance: dict[str, Any] = {
        "status": "unmeasured",
        "provenance_type": "none",
        "release_provenance": "evaluation/results/method_comparison_ablation.json",
        "is_empirical": False,
        "timestamp": None,
    }

    if results_path.exists():
        try:
            loaded = json.loads(results_path.read_text(encoding="utf-8"))
            if isinstance(loaded, dict) and "summary" in loaded:
                empirical_results = loaded
                provenance["status"] = "measured"
                provenance["provenance_type"] = "empirical_release_artifact"
                provenance["is_empirical"] = True
                provenance["timestamp"] = loaded.get("timestamp")
        except Exception:
            pass

    # Collect live telemetry stats
    latencies: list[float] = []
    traces_count = 0
    supported_claims = 0
    total_claims = 0
    if TRACES_DIR.exists():
        trace_files = sorted(TRACES_DIR.glob("*.json"), key=lambda f: f.stat().st_mtime, reverse=True)
        traces_count = len(trace_files)
        for tf in trace_files[:100]:
            try:
                data = json.loads(tf.read_text(encoding="utf-8"))
                lat = data.get("latency_ms")
                if isinstance(lat, (int, float)) and lat > 0:
                    latencies.append(float(lat))
                v_rep = data.get("verification_report")
                if isinstance(v_rep, dict) and isinstance(v_rep.get("claims"), list):
                    for cl in v_rep["claims"]:
                        total_claims += 1
                        if cl.get("status") == "SUPPORTED":
                            supported_claims += 1
            except Exception:
                continue

    p50_latency = round(sorted(latencies)[len(latencies) // 2], 2) if latencies else None
    claim_support_rate = round((supported_claims / total_claims) * 100.0, 1) if total_claims > 0 else None
    methods_summary = empirical_results.get("summary", {}) if empirical_results else {}

    return {
        "provenance": provenance,
        "empirical_summary": methods_summary,
        "live_telemetry": {
            "total_traces_recorded": traces_count,
            "sample_size": len(latencies),
            "p50_latency_ms": p50_latency,
            "total_verified_claims": total_claims,
            "supported_claims_count": supported_claims,
            "claim_support_rate_pct": claim_support_rate,
        },
        "reasoning_levels": [
            {
                "level": "L1",
                "name": "Direct Lookup",
                "desc": "Single-hop hyperparameter or factual definition",
                "dense": "88.2%",
                "hybrid": "94.5%",
                "scholar": "98.8%",
                "cer": "100%",
                "status": "target_projection",
            },
            {
                "level": "L2",
                "name": "Same-Section Reasoning",
                "desc": "Intra-section explanation or architectural rationale",
                "dense": "72.4%",
                "hybrid": "81.0%",
                "scholar": "95.2%",
                "cer": "96.4%",
                "status": "target_projection",
            },
            {
                "level": "L3",
                "name": "Cross-Section Reasoning",
                "desc": "Connecting methodology prose to experimental validation",
                "dense": "51.6%",
                "hybrid": "66.3%",
                "scholar": "91.7%",
                "cer": "94.0%",
                "status": "target_projection",
            },
            {
                "level": "L4",
                "name": "Cross-Modal Reasoning",
                "desc": "Verifying claims against 2D tables and vector figure panels",
                "dense": "38.0%",
                "hybrid": "58.2%",
                "scholar": "94.1%",
                "cer": "95.5%",
                "status": "target_projection",
            },
            {
                "level": "L5",
                "name": "Multi-Hop Synthesis",
                "desc": "End-to-end synthesis: Architecture -> Ablation -> Results",
                "dense": "31.5%",
                "hybrid": "49.0%",
                "scholar": "89.6%",
                "cer": "100.0%",
                "status": "target_projection",
            },
        ],
        "hardware_tiers": [
            {
                "tier": "8GB RAM / VRAM",
                "target": "Gemma 4 2B / Llama 3.2 3B",
                "context": "2,048 tokens",
                "budget": "4 Text blocks, 1 Table",
                "latency_target": "1.8 ms",
            },
            {
                "tier": "16GB RAM / VRAM",
                "target": "Gemma 4 12B / Qwen 2.5 7B",
                "context": "4,096 tokens",
                "budget": "6 Text blocks, 2 Tables, 1 High-res crop",
                "latency_target": "2.4 ms",
            },
            {
                "tier": "32GB+ Unified Memory",
                "target": "Gemma 4 27B / Vision LLMs",
                "context": "8,192 tokens",
                "budget": "10 Blocks, full multimodal DAG",
                "latency_target": "3.1 ms",
            },
        ],
    }


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

    digest = hashlib.sha256(content).hexdigest()[:16]
    local_id = safe_paper_id(f"upload_{digest}")
    directory = paper_dir(local_id)
    cached = await asyncio.to_thread(
        PaperFinalizeService.load_if_complete,
        local_id,
        target_dir=directory,
    )
    if cached is not None:
        return cached

    clean_title = title.strip() or Path(filename).stem.replace("_", " ").replace("-", " ").strip() or "Uploaded PDF"
    directory.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{local_id}.upload-", dir=str(directory.parent)
        ) as temp_dir:
            acquired_pdf = Path(temp_dir) / "paper.pdf"
            atomic_write_bytes(acquired_pdf, content)
            pages = await asyncio.to_thread(extract_pages, acquired_pdf)
            inferred = infer_uploaded_metadata(pages, clean_title)
            metadata = {
                "id": local_id,
                "local_id": local_id,
                "title": inferred["title"],
                "authors": ["Uploaded PDF"],
                "year": "",
                "summary": inferred["summary"],
                "abstract": inferred["summary"],
                "categories": ["PDF"],
                "pdf_url": "",
                "abs_url": "",
                "published": "",
                "source": "upload",
                "filename": filename,
            }
            return await asyncio.to_thread(
                PaperFinalizeService.finalize,
                acquired_pdf,
                local_id,
                metadata,
                target_dir=directory,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Uploaded-paper finalization failed for [%s]: %s", local_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Could not process the paper. Please try again.",
        ) from exc


def _paths_or_404(paper_id: str) -> tuple[Path, Path, Path, Path]:
    directory = paper_dir(safe_paper_id(paper_id))
    metadata_path = directory / "metadata.json"
    pages_path = directory / "pages.json"
    chunks_path = directory / "chunks.json"
    pdf_path = directory / "paper.pdf"
    required = (metadata_path, pages_path, chunks_path, pdf_path)
    if not all(path.is_file() for path in required):
        raise HTTPException(status_code=404, detail="Paper has not been completely prepared")
    return metadata_path, pages_path, chunks_path, pdf_path


# ---------------------------------------------------------------------------
# AST & Document Representation endpoints
# ---------------------------------------------------------------------------

@app.get("/api/papers/{paper_id}/document-ast")
async def get_paper_document_ast(paper_id: str) -> dict[str, Any]:
    """Return the derived ScientificDocument compatibility projection."""
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
        sec_id = safe_paper_id(str(ref["secondary_local_id"]))
        cached = await asyncio.to_thread(
            PaperFinalizeService.load_if_complete,
            sec_id,
            target_dir=paper_dir(sec_id),
        )
        if cached is not None:
            return {
                "secondary_paper_id": sec_id,
                "chunks": cached["chunks"],
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

    sec_dir = paper_dir(sec_local_id)
    cached = await asyncio.to_thread(
        PaperFinalizeService.load_if_complete,
        sec_local_id,
        target_dir=sec_dir,
    )
    if cached is not None:
        mark_reference_ingested(paper_id, ref_index, sec_local_id)
        return {
            "secondary_paper_id": sec_local_id,
            "title": cached["metadata"].get("title", title),
            "chunks": cached["chunks"],
            "cached": True,
        }

    sec_metadata = {
        "id": arxiv_id or sec_local_id,
        "local_id": sec_local_id,
        "title": title,
        "authors": ref.get("authors", []),
        "year": ref.get("year", ""),
        "summary": ref.get("abstract", ""),
        "abstract": ref.get("abstract", ""),
        "categories": [],
        "pdf_url": pdf_url,
        "abs_url": ref.get("abs_url", ""),
        "published": "",
        "source": "reference",
        "anchor_paper": safe_paper_id(paper_id),
    }
    sec_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{sec_local_id}.acquire-", dir=str(sec_dir.parent)
        ) as temp_dir:
            acquired_pdf = Path(temp_dir) / "paper.pdf"
            await download_pdf(pdf_url, acquired_pdf)
            finalized = await asyncio.to_thread(
                PaperFinalizeService.finalize,
                acquired_pdf,
                sec_local_id,
                sec_metadata,
                target_dir=sec_dir,
            )
    except HTTPException:
        raise
    except Exception as exc:
        logger.warning("Reference finalization failed for [%s]: %s", sec_local_id, exc)
        raise HTTPException(
            status_code=500,
            detail="Could not process the paper. Please try again.",
        ) from exc

    mark_reference_ingested(paper_id, ref_index, sec_local_id)
    return {
        "secondary_paper_id": sec_local_id,
        "title": finalized["metadata"].get("title", title),
        "chunks": finalized["chunks"],
        "cached": False,
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


def _answer_request_from_chat(paper_id: str, payload: ChatInput) -> AnswerPipelineRequest:
    return AnswerPipelineRequest(
        paper_id=paper_id,
        query=payload.message,
        history=payload.history,
        secondary_paper_ids=payload.secondary_paper_ids,
        requested_model=payload.model,
        generation_seed=payload.generation_seed,
        capability_mode=payload.capability_mode,
        execution_policy=payload.execution_policy,
        intervention=payload.intervention,
        decoding=payload.decoding,
        evaluation_context=payload.evaluation_context,
        experiment_id=payload.experiment_id,
        visual_page_backend=payload.visual_page_backend,
        snippet_id=payload.snippet_id,
        snippet_page=payload.snippet_page,
        snippet_bbox=payload.snippet_bbox,
        snippet_text=payload.snippet_text,
    )


@app.post("/api/papers/{paper_id}/chat")
async def chat(paper_id: str, payload: ChatInput) -> dict[str, Any]:
    """Execute the shared production/evaluation pipeline and return its v1 trace."""
    try:
        trace = await AnswerPipelineService.answer(
            _answer_request_from_chat(paper_id, payload)
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return trace.to_chat_response()


@app.post("/api/papers/{paper_id}/chat/stream")
async def chat_stream(paper_id: str, payload: ChatInput) -> StreamingResponse:
    """Expose the shared verified answer trace as server-sent events.

    Predictable request and paper errors are validated before the response is
    opened, preserving `/chat` HTTP status semantics. Runtime failures after
    streaming starts use a typed terminal SSE error event.
    """
    request = _answer_request_from_chat(paper_id, payload)
    try:
        if not payload.message.strip():
            raise ValueError("Message cannot be empty")
        await asyncio.to_thread(answer_pipeline_module._paper_paths, paper_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    async def event_generator():
        stage_queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()

        def on_stage(stage_name: str, stage_data: dict[str, Any]) -> None:
            stage_queue.put_nowait(stage_data)

        task = asyncio.create_task(
            AnswerPipelineService.answer(request, stage_callback=on_stage)
        )

        try:
            trace = await task
        except asyncio.CancelledError:
            task.cancel()
            raise
        except Exception as exc:
            payload_data = {
                "message": str(exc),
                "error_type": type(exc).__name__,
                "terminal": True,
            }
            yield f"event: error\ndata: {json.dumps(payload_data)}\n\n"
            yield 'event: done\ndata: {"status":"error"}\n\n'
            return
        finally:
            if not task.done():
                task.cancel()

        yield f"event: analysis\ndata: {json.dumps(trace.analysis.model_dump(mode='json'))}\n\n"
        yield f"event: evidence_path\ndata: {json.dumps([step.model_dump(mode='json') for step in trace.reasoning_path])}\n\n"
        while not stage_queue.empty():
            st = stage_queue.get_nowait()
            yield f"event: stage\ndata: {json.dumps(st)}\n\n"
        if trace.numeric_plan is not None:
            yield f"event: numeric_math\ndata: {json.dumps(trace.numeric_plan.model_dump(mode='json'))}\n\n"
        yield f"event: token\ndata: {json.dumps({'token': trace.final_answer})}\n\n"
        yield f"event: trace\ndata: {json.dumps(trace.model_dump(mode='json'))}\n\n"
        if trace.verification_report is not None:
            yield f"event: verification\ndata: {json.dumps(trace.verification_report.model_dump(mode='json'))}\n\n"
        yield 'event: done\ndata: {"status":"ok"}\n\n'

    return StreamingResponse(event_generator(), media_type="text/event-stream")
