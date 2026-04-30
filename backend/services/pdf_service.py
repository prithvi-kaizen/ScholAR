from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import fitz
import httpx


BASE_DIR = Path(__file__).resolve().parents[1]
PAPERS_DIR = BASE_DIR / "data" / "papers"


def safe_paper_id(paper_id: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", paper_id).strip("._-")
    return cleaned or "paper"


def paper_dir(paper_id: str) -> Path:
    return PAPERS_DIR / safe_paper_id(paper_id)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)


async def download_pdf(pdf_url: str, destination: Path) -> None:
    if not pdf_url:
        raise RuntimeError("Paper is missing a PDF URL")

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        async with httpx.AsyncClient(timeout=45.0, follow_redirects=True) as client:
            response = await client.get(pdf_url)
            response.raise_for_status()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"PDF download failed: {exc}") from exc

    content_type = response.headers.get("content-type", "")
    if "pdf" not in content_type.lower() and not response.content.startswith(b"%PDF"):
        raise RuntimeError("Downloaded file does not look like a PDF")

    destination.write_bytes(response.content)


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    if not pdf_path.exists():
        raise RuntimeError("Local PDF does not exist")

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            pages.append({"page": index, "text": text})
    return pages


def render_page_png(pdf_path: Path, page_number: int, zoom: float = 1.8) -> bytes:
    if not pdf_path.exists():
        raise RuntimeError("Local PDF does not exist")

    safe_zoom = min(max(zoom, 0.8), 3.0)
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise RuntimeError("Requested PDF page is out of range")
        page = document.load_page(page_number - 1)
        matrix = fitz.Matrix(safe_zoom, safe_zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
