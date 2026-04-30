from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from backend.services.arxiv_service import search_arxiv
from backend.services.chunking_service import chunk_pages
from backend.services.ollama_service import (
    OLLAMA_MODEL,
    fallback_goals,
    generate,
    generate_study_goals,
    ollama_available,
)
from backend.services.pdf_service import (
    download_pdf,
    extract_pages,
    paper_dir,
    read_json,
    render_page_png,
    safe_paper_id,
    write_json,
)
from backend.services.retrieval_service import retrieve_chunks, short_quote


app = FastAPI(title="ScholAR API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
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


@app.get("/health")
async def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "ollama_available": await ollama_available(),
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

    metadata = paper.model_dump()
    metadata["local_id"] = local_id

    try:
        directory.mkdir(parents=True, exist_ok=True)
        write_json(metadata_path, metadata)

        if not pdf_path.exists():
            await download_pdf(paper.pdf_url, pdf_path)

        if not pages_path.exists():
            pages = extract_pages(pdf_path)
            write_json(pages_path, pages)
        else:
            pages = read_json(pages_path)

        if not chunks_path.exists():
            chunks = chunk_pages(pages)
            write_json(chunks_path, chunks)
        else:
            chunks = read_json(chunks_path)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "paper_id": local_id,
        "metadata": metadata,
        "pages": len(pages),
        "chunks": len(chunks),
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


@app.get("/api/papers/{paper_id}")
async def get_paper(paper_id: str) -> dict[str, Any]:
    metadata_path, pages_path, chunks_path, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    metadata["pages"] = len(read_json(pages_path)) if pages_path.exists() else 0
    metadata["chunks"] = len(read_json(chunks_path)) if chunks_path.exists() else 0
    return metadata


@app.get("/api/papers/{paper_id}/pdf")
async def get_pdf(paper_id: str) -> FileResponse:
    _, _, _, pdf_path = _paths_or_404(paper_id)
    if not pdf_path.exists():
        raise HTTPException(status_code=404, detail="PDF is missing")
    return FileResponse(pdf_path, media_type="application/pdf", filename="paper.pdf")


@app.get("/api/papers/{paper_id}/page/{page_number}.png")
async def get_pdf_page(paper_id: str, page_number: int, zoom: float = 1.8) -> Response:
    _, _, _, pdf_path = _paths_or_404(paper_id)
    try:
        image_bytes = render_page_png(pdf_path, page_number, zoom)
    except RuntimeError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return Response(content=image_bytes, media_type="image/png")


@app.post("/api/papers/{paper_id}/study-goals")
async def study_goals(paper_id: str) -> dict[str, Any]:
    metadata_path, _, chunks_path, _ = _paths_or_404(paper_id)
    metadata = read_json(metadata_path)
    chunks = read_json(chunks_path) if chunks_path.exists() else []
    goals_path = metadata_path.parent / "goals.json"

    if goals_path.exists():
        return {"goals": read_json(goals_path)}

    if not await ollama_available():
        return {"goals": fallback_goals()}

    try:
        goals = await asyncio.wait_for(generate_study_goals(metadata, chunks), timeout=10.0)
        write_json(goals_path, goals)
    except Exception:
        goals = fallback_goals()
    return {"goals": goals}


@app.post("/api/papers/{paper_id}/chat")
async def chat(paper_id: str, payload: ChatInput) -> dict[str, Any]:
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    _, _, chunks_path, _ = _paths_or_404(paper_id)
    chunks = read_json(chunks_path) if chunks_path.exists() else []
    selected = retrieve_chunks(payload.message, chunks, limit=4)

    if not selected:
        return {
            "answer": "The paper context does not contain enough information to answer that.",
            "citations": [],
        }

    citations = [
        {
            "page": chunk.get("page"),
            "chunk_id": chunk.get("chunk_id"),
            "quote": short_quote(chunk, payload.message),
        }
        for chunk in selected
    ]

    context = "\n\n".join(
        f"[{chunk.get('chunk_id')} | p. {chunk.get('page')}]\n{chunk.get('text', '')[:2400]}"
        for chunk in selected
    )
    prompt = f"""
You answer questions about a research paper using only the provided context.
If the context does not contain enough information, say: "The paper context does not contain enough information."
Cite page numbers inline like [p. 2]. Keep the answer concise but useful.

Context:
{context}

Question: {payload.message}
""".strip()

    if await ollama_available():
        try:
            answer = await generate(prompt)
        except Exception:
            answer = ""
    else:
        answer = ""

    if not answer:
        pages = sorted({citation["page"] for citation in citations if citation.get("page")})
        page_text = ", ".join(f"[p. {page}]" for page in pages)
        best_quote = next(
            (citation["quote"] for citation in citations if len(citation.get("quote", "")) > 80),
            citations[0]["quote"],
        )
        answer = (
            "Based on the retrieved paper context, the most relevant passages are on "
            f"{page_text}. {best_quote}"
        )

    return {"answer": answer, "citations": citations}
