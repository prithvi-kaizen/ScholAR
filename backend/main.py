from __future__ import annotations

import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.services.arxiv_service import search_arxiv
from backend.services.chunking_service import chunk_figures, chunk_pages
from backend.services.ollama_service import (
    OLLAMA_MODEL,
    fallback_goals,
    generate,
    generate_study_goals,
    ollama_available,
    STUDY_GOAL_PROMPT_VERSION,
)
from backend.services.pdf_service import (
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
from backend.services.vision_service import answer_with_figure
from backend.services.web_search_service import search_web, web_search_enabled


app = FastAPI(title="ScholAR API")

app.add_middleware(
    CORSMiddleware,
    # No cookies/session auth are used anywhere in this app, so credentials
    # stay disabled; combined with a wildcard origin, allow_credentials=True
    # would let any website that a user's browser visits make authenticated-
    # looking requests against this local server (browser-as-confused-deputy).
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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
    message: str
    history: list[dict[str, Any]] = Field(default_factory=list)
    web_search: bool = True
    # Multi-document mode: local_ids of secondary papers already ingested
    secondary_paper_ids: list[str] = Field(default_factory=list)


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
        "web_results": [],
        "used_web_search": False,
        "model": OLLAMA_MODEL,
        "error": True,
        "message": message,
    }


def _model_failure_payload(exc: Exception | None = None) -> dict[str, Any]:
    if isinstance(exc, (httpx.ReadTimeout, httpx.TimeoutException)):
        return _error_payload("Local model took too long to answer. Try again with a shorter question.")
    return _error_payload("Local model could not complete this answer. Make sure Ollama is running and the selected model is loaded.")


def _should_search_web(message: str, selected_chunks: list[dict[str, Any]]) -> bool:
    lowered = message.lower()
    explicit_terms = (
        "web",
        "internet",
        "search",
        "latest",
        "recent",
        "current",
        "today",
        "news",
        "who is",
        "what is",
        "compare with",
        "outside this paper",
        "other papers",
        "implementation library",
        "github",
        "dataset link",
    )
    if any(term in lowered for term in explicit_terms):
        return True
    if not selected_chunks:
        return True
    if len(message.split()) > 10 and len(selected_chunks) < 3:
        return True
    return False


def _web_query(message: str, metadata: dict[str, Any]) -> str:
    title = metadata.get("title") or ""
    if any(term in message.lower() for term in ("this paper", "the paper", "authors", "code", "github", "dataset")):
        return f"{title} {message}".strip()
    return message.strip()


def _format_web_context(results: list[dict[str, Any]]) -> str:
    if not results:
        return "No web results were available."
    return "\n\n".join(
        f"[web:{index}]\nTitle: {result.get('title')}\nURL: {result.get('url')}\nSnippet: {result.get('snippet')}"
        for index, result in enumerate(results, start=1)
    )


def _extractive_web_answer(results: list[dict[str, Any]]) -> str:
    if not results:
        return ""
    lines = ["I could not get a full model response, but the web search returned these potentially useful sources:"]
    for index, result in enumerate(results[:5], start=1):
        snippet = result.get("snippet") or "No snippet available."
        lines.append(f"- {result.get('title')} [web:{index}]\n  {snippet}\n  {result.get('url')}")
    return "\n".join(lines)


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

    def replace_evidence_id(match: re.Match[str]) -> str:
        evidence_id = f"E{match.group(1)}".upper()
        item = evidence_by_id.get(evidence_id)
        if not item:
            return ""
        if evidence_id not in used_ids:
            used_ids.append(evidence_id)
        return f"[{used_ids.index(evidence_id) + 1}]"

    normalized = re.sub(r"\[E\s*(\d+)\]", replace_evidence_id, answer_without_direct_pages, flags=re.IGNORECASE)
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
        "web_search_enabled": web_search_enabled(),
        "model": OLLAMA_MODEL,
    }


@app.get("/api/search")
async def search(q: str = "", max_results: int = 12) -> dict[str, Any]:
    try:
        papers = await search_arxiv(q, max_results=max_results)
    except RuntimeError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
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
            chunks = chunk_pages(pages)
            write_json(chunks_path, chunks)
        else:
            chunks = _ensure_chunk_metadata(chunks_path, pages_path)

        # Extract figures/tables (idempotent — skipped if figures.json exists)
        if not figures_path.exists():
            figures = await asyncio.to_thread(extract_figures, pdf_path, figures_dir)
            write_json(figures_path, figures)
        else:
            figures = read_json(figures_path)

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "paper_id": local_id,
        "metadata": metadata,
        "pages": len(pages),
        "chunks": len(chunks),
        "figures": len(figures),
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
        chunks = chunk_pages(pages)
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

    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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
# Reference endpoints (Step 1 & 2 — multi-document mode)
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
            sec_chunks = chunk_pages(sec_pages, source_paper_id=sec_local_id)
            write_json(sec_chunks_path, sec_chunks)
        else:
            sec_chunks = read_json(sec_chunks_path)
            # Back-fill source_paper_id if missing (re-ingestion of older cache)
            if sec_chunks and "source_paper_id" not in sec_chunks[0]:
                sec_chunks = chunk_pages(sec_pages, source_paper_id=sec_local_id)
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
        raise HTTPException(status_code=500, detail=str(exc)) from exc

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


@app.post("/api/papers/{paper_id}/study-goals")
async def study_goals(
    paper_id: str,
    payload: StudyGoalsInput | None = None,
) -> dict[str, Any]:
    payload = payload or StudyGoalsInput()
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

    if not await ollama_available():
        return {
            "goals": fallback_goals(metadata, chunks),
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
        goals = await asyncio.wait_for(generate_study_goals(metadata, chunks), timeout=120.0)
        write_json(goals_path, goals)
    except Exception:
        goals = fallback_goals(metadata, chunks)
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
    # source_paper_id. retrieve_chunks() is unchanged — it operates on any list.
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
        figure_chunks = chunk_figures(figure_records, source_paper_id=paper_id)
        # Tag anchor figure chunks with paper_id (same convention as text chunks)
        figure_chunks = [{**c, "source_paper_id": paper_id} for c in figure_chunks]
        all_chunks = all_chunks + figure_chunks

    page_hints = extract_page_hints(payload.message)
    selected = retrieve_chunks(payload.message, all_chunks, limit=6, preferred_pages=page_hints)
    web_results: list[dict[str, Any]] = []
    used_web_search = False

    if payload.web_search and web_search_enabled() and _should_search_web(payload.message, selected):
        used_web_search = True
        web_results = await search_web(_web_query(payload.message, metadata), limit=5)

    if not selected and not web_results:
        return {
            "answer": "The paper context does not contain enough information to answer that.",
            "citations": [],
            "web_results": [],
            "used_web_search": used_web_search,
        }

    # ── Vision routing ────────────────────────────────────────────────────────
    # When the top-1 retrieved chunk is a figure/table chunk, delegate to the
    # vision path instead of the text-only path. answer_with_figure() itself
    # falls back to a caption-only answer if Ollama is unavailable.
    top_chunk = selected[0] if selected else None
    if top_chunk is not None and top_chunk.get("is_figure_chunk"):
        text_support = [c for c in selected[1:] if not c.get("is_figure_chunk")]
        vision_result = await answer_with_figure(
            question=payload.message,
            figure_chunk=top_chunk,
            context_chunks=text_support,
            paper_id=paper_id,
            paper_metadata=metadata,
        )
        fig_label = vision_result.get("label", "Figure")
        return {
            "answer":            vision_result["answer"],
            "citations":         vision_result["citations"],
            "web_results":       [],
            "used_web_search":   False,
            "model":             vision_result.get("model_used", OLLAMA_MODEL),
            "vision":            True,
            "vision_fallback":   vision_result.get("fallback", False),
            "figure_id":         vision_result.get("figure_id"),
            "figure_label":      fig_label,
            "figure_image_url":  f"/api/papers/{paper_id}/figures/{vision_result.get('figure_id')}.png"
                                  if vision_result.get("figure_id") else None,
        }

    ollama_up = await ollama_available()
    context_chunks: list[dict[str, Any]] = []
    # chunk_id numbering restarts per paper (chunk_001, chunk_002, ...), so in
    # multi-document mode two different papers' chunks can share a chunk_id —
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

    evidence_items = _build_evidence_items(question_text, prompt_chunks, limit=7)
    paper_context = _format_evidence_context(evidence_items, secondary_meta=secondary_meta)
    web_context = _format_web_context(web_results)
    recent_history = "\n".join(
        f"{item.get('role', 'user')}: {str(item.get('content', ''))[:history_char_limit]}"
        for item in payload.history[-6:]
        if isinstance(item, dict)
    )
    prompt = f"""
You are ScholAR, a rigorous research paper tutor with two tools:
1. Paper retrieval: selected chunks from the PDF being studied.
2. Web search: free web results, used only when paper context is insufficient or the question asks for outside/current information.

Think in this order:
1. Identify whether the user is asking about the paper itself, outside context, or both.
2. Use paper context first for paper-specific claims.
3. Use web context only for outside/current/general facts or when the paper context is missing.
4. Reconcile conflicts explicitly instead of blending sources.

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
- Wrap every piece of mathematical notation in single dollar signs, e.g. $x_t$ or $p_\\theta(x_{{t-1}}\\mid x_t)$, so it renders instead of showing raw symbols.
- Cite paper claims inline only with evidence IDs from the Paper evidence list, such as [E1] or [E2].
- Never invent page citations. Do not write [p. 1], [p. 2], or any page number yourself.
- The app will convert evidence IDs into compact numbered references like [1], [2].
- Every paper-specific claim in the Evidence section must cite one of the provided evidence IDs.
- Cite web claims inline as [web:1].
- Never cite web results as if they came from the paper.
- If paper context is insufficient, say what is missing before using web context.
- If web search was unavailable or empty, do not pretend you searched.
- For methods/results questions, include the specific mechanism, dataset, metric, number, or comparison when the evidence provides it.
- Use concise sections: "Answer", "Evidence", "What this means", and "Limits / what to verify" when helpful.
- Answer directly in 180 to 320 words. Do not include hidden reasoning or long deliberation.
- Use only the strongest 2 to 4 evidence points and keep citations close to the claims they support.

Paper evidence:
{paper_context}

Web context:
{web_context}

Question: {question_text}
""".strip()

    if ollama_up:
        try:
            answer = await generate(prompt, temperature=0.1)
        except Exception as exc:
            return {
                **_model_failure_payload(exc),
                "used_web_search": used_web_search,
                "web_results": web_results,
            }
    else:
        answer = ""

    if not answer:
        if ollama_up:
            return {
                **_error_payload("The local model returned an empty answer. Try again."),
                "used_web_search": used_web_search,
                "web_results": web_results,
            }
        answer = _extractive_tutor_answer(payload.message, selected) if selected else ""
        web_answer = _extractive_web_answer(web_results)
        if web_answer:
            answer = f"{answer}\n\n{web_answer}".strip()

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

    return {
        "answer": answer,
        "citations": citations,
        "web_results": web_results,
        "used_web_search": used_web_search,
        "model": OLLAMA_MODEL,
    }
