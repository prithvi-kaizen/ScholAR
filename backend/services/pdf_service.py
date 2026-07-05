from __future__ import annotations

import asyncio
import json
import re
import unicodedata
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


_PDF_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/pdf,*/*",
}


_ARXIV_ID_RE = re.compile(
    r"(?:arxiv\.org/(?:abs|pdf)/|arXiv:)([0-9]{4}\.[0-9]+(?:v\d+)?|[a-z\-]+/\d+)",
    re.IGNORECASE,
)


def _extract_arxiv_id(url_or_id: str) -> str | None:
    """Extract a bare arXiv ID (e.g. '1706.03762') from a URL or ID string."""
    m = _ARXIV_ID_RE.search(url_or_id)
    if m:
        return re.sub(r"v\d+$", "", m.group(1))
    # Maybe it's already a bare id like '1706.03762'
    if re.fullmatch(r"[0-9]{4}\.[0-9]+", url_or_id.strip()):
        return url_or_id.strip()
    return None


def _arxiv_url_candidates(pdf_url: str, oa_pdf_url: str | None = None) -> list[str]:
    """Return URLs to try in order. Prefers known open-access URLs."""
    candidates: list[str] = []
    if oa_pdf_url and oa_pdf_url.startswith("http"):
        candidates.append(oa_pdf_url)

    arxiv_id = _extract_arxiv_id(pdf_url)
    if arxiv_id:
        candidates += [
            f"https://arxiv.org/pdf/{arxiv_id}",
            f"https://export.arxiv.org/pdf/{arxiv_id}",
        ]
    elif pdf_url not in candidates:
        candidates.append(pdf_url)

    return candidates



_MAX_PDF_BYTES = 50 * 1024 * 1024  # 50MB — reject/abort anything larger


async def download_pdf(pdf_url: str, destination: Path) -> None:
    if not pdf_url:
        raise RuntimeError("Paper is missing a PDF URL")

    destination.parent.mkdir(parents=True, exist_ok=True)

    urls_to_try = _arxiv_url_candidates(pdf_url)
    last_exc: Exception | None = None

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True, headers=_PDF_HEADERS) as client:
        for attempt, url in enumerate(urls_to_try):
            try:
                if attempt > 0:
                    await asyncio.sleep(2.0 * attempt)
                async with client.stream("GET", url) as response:
                    if response.status_code == 403 and attempt < len(urls_to_try) - 1:
                        continue  # try next URL
                    response.raise_for_status()

                    content_type = response.headers.get("content-type", "")
                    chunks: list[bytes] = []
                    total = 0
                    async for chunk in response.aiter_bytes():
                        total += len(chunk)
                        if total > _MAX_PDF_BYTES:
                            raise RuntimeError(f"PDF exceeds {_MAX_PDF_BYTES // (1024 * 1024)}MB limit")
                        chunks.append(chunk)
                    content = b"".join(chunks)

                if "pdf" in content_type.lower() or content.startswith(b"%PDF"):
                    destination.write_bytes(content)
                    return
                # Not a PDF — try next
            except httpx.HTTPStatusError as exc:
                last_exc = exc
                if attempt < len(urls_to_try) - 1:
                    continue
            except httpx.HTTPError as exc:
                last_exc = exc
                break

    raise RuntimeError(f"PDF download failed: {last_exc}") from last_exc


def extract_pages(pdf_path: Path) -> list[dict[str, Any]]:
    if not pdf_path.exists():
        raise RuntimeError("Local PDF does not exist")

    pages: list[dict[str, Any]] = []
    with fitz.open(pdf_path) as document:
        for index, page in enumerate(document, start=1):
            text = re.sub(r"\s+", " ", page.get_text("text")).strip()
            pages.append({"page": index, "text": text})
    return pages


# ---------------------------------------------------------------------------
# Figure / Table extraction
# ---------------------------------------------------------------------------

_CAPTION_RE = re.compile(
    r"(?:^|\n)\s*((?:Figure|Fig\.|Table)\s+\d+(?:\.\d+)?)\s*[:.)]?\s*(.{0,300})",
    re.IGNORECASE | re.MULTILINE,
)

# Fraction of page height to capture above/below the caption block
_FIG_ABOVE_FRAC = 0.38
_FIG_BELOW_FRAC = 0.12
_FIG_RENDER_ZOOM = 3.0  # 3× so dense figures (small axis labels, grid thumbnails) stay legible to the vision model


def _caption_bbox(page: "fitz.Page", caption_label: str) -> "fitz.Rect | None":
    """Find the bounding box of the caption label text block on a PDF page."""
    blocks = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get("blocks", [])
    label_lower = caption_label.lower().replace(".", "").strip()
    best: "fitz.Rect | None" = None
    for block in blocks:
        if block.get("type") != 0:
            continue
        block_text = " ".join(
            span.get("text", "")
            for line in block.get("lines", [])
            for span in line.get("spans", [])
        ).lower().replace(".", "")
        if label_lower in block_text:
            best = fitz.Rect(block["bbox"])
            break
    return best


def extract_figures(
    pdf_path: Path,
    figures_dir: Path,
) -> list[dict[str, Any]]:
    """Extract figure/table image chips and metadata from a PDF.

    For each detected Figure N / Table N caption:
    - Finds the caption bounding box on the page.
    - Renders a clip region that captures the image region *above* the caption
      (plus a small strip below it) at 2× zoom.
    - Saves the clip as a PNG file under ``figures_dir``.
    - Returns a list of figure metadata dicts suitable for ``figures.json``.

    Uses page-region clip rendering as primary strategy so vector figures
    (charts, diagrams, tables drawn with lines) are captured correctly.
    ``get_images()`` only returns embedded bitmaps (photos), so vector figures
    would be missed if that were the only approach.
    """
    if not pdf_path.exists():
        raise RuntimeError("Local PDF does not exist")

    figures_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    fig_counter = 0

    with fitz.open(pdf_path) as doc:
        for page_idx, page in enumerate(doc):
            page_num = page_idx + 1
            page_text = page.get_text("text")
            page_rect = page.rect  # (x0=0, y0=0, x1=width, y1=height)

            for match in _CAPTION_RE.finditer(page_text):
                label_raw = match.group(1).strip()   # e.g. "Figure 1"
                caption_text = re.sub(r"\s+", " ", match.group(2)).strip()

                # Determine type: figure vs table
                figure_type = "table" if label_raw.lower().startswith("table") else "figure"

                # Normalise label for display ("Fig. 3" → "Figure 3")
                label_norm = re.sub(r"^Fig\.", "Figure", label_raw, flags=re.IGNORECASE).strip()

                # Locate caption bbox on the page so we know where to clip
                cap_bbox = _caption_bbox(page, label_raw)

                if cap_bbox:
                    # Clip: from (above_frac × page height) above caption top
                    #        to (below_frac × page height) below caption bottom
                    clip_y0 = max(0.0, cap_bbox.y0 - _FIG_ABOVE_FRAC * page_rect.height)
                    clip_y1 = min(page_rect.height, cap_bbox.y1 + _FIG_BELOW_FRAC * page_rect.height)
                    clip = fitz.Rect(0, clip_y0, page_rect.width, clip_y1)
                else:
                    # Fallback: render the whole page
                    clip = page_rect

                mat = fitz.Matrix(_FIG_RENDER_ZOOM, _FIG_RENDER_ZOOM)
                pix = page.get_pixmap(matrix=mat, clip=clip, alpha=False)
                png_bytes = pix.tobytes("png")

                fig_counter += 1
                fig_id = f"fig_{page_num:02d}_{fig_counter:03d}"
                rel_filename = f"{fig_id}.png"
                (figures_dir / rel_filename).write_bytes(png_bytes)

                bbox_dict = {
                    "x0": round(clip.x0, 1),
                    "y0": round(clip.y0, 1),
                    "x1": round(clip.x1, 1),
                    "y1": round(clip.y1, 1),
                } if cap_bbox else None

                records.append({
                    "figure_id":   fig_id,
                    "figure_type": figure_type,   # "figure" | "table"
                    "label":       label_norm,     # "Figure 1", "Table 2", …
                    "caption":     caption_text,
                    "page":        page_num,
                    "bbox":        bbox_dict,
                    "image_file":  rel_filename,
                    "width_px":    pix.width,
                    "height_px":   pix.height,
                })

    return records


def infer_uploaded_metadata(pages: list[dict[str, Any]], fallback_title: str) -> dict[str, Any]:
    first_page = pages[0].get("text", "") if pages else ""
    first_page = re.sub(r"\s+", " ", first_page).strip()

    title = fallback_title
    abstract = ""

    abstract_match = re.search(r"\bAbstract\b\s*(.+?)(?:\b1\s+Introduction\b|\bIntroduction\b|\bKeywords\b|$)", first_page, re.IGNORECASE)
    if abstract_match:
        abstract = abstract_match.group(1).strip()

    title_match = re.search(r"^(.+?)(?:\[Author|Author|April|Abstract)", first_page, re.IGNORECASE)
    if title_match:
        raw_title = title_match.group(1).strip()
        raw_title = re.sub(r"\s+", " ", raw_title)
        if 12 <= len(raw_title) <= 220:
            title = raw_title

    return {
        "title": title or fallback_title,
        "summary": abstract[:1800] or "Custom PDF uploaded for local study.",
    }


def _highlight_quote(page: fitz.Page, quote: str) -> None:
    cleaned = re.sub(r"\s+", " ", quote).strip()
    if not cleaned:
        return

    candidates = [cleaned]
    if len(cleaned) > 160:
        candidates.append(cleaned[:160])
    if "." in cleaned:
        candidates.extend(part.strip() for part in cleaned.split(".") if len(part.strip()) > 40)

    seen: set[str] = set()
    for candidate in candidates:
        candidate = candidate.strip(" .")
        if candidate in seen or len(candidate) < 24:
            continue
        seen.add(candidate)
        rectangles = page.search_for(candidate)
        if not rectangles:
            continue
        for rect in rectangles[:8]:
            page.draw_rect(rect + (-2, -1, 2, 1), color=(0.12, 0.45, 1), fill=(0.58, 0.76, 1), fill_opacity=0.36, width=1.0)
        return
    _highlight_quote_words(page, cleaned)


def _normalize_pdf_text(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text)
    normalized = (
        normalized.replace("ﬁ", "fi")
        .replace("ﬂ", "fl")
        .replace("−", "-")
        .replace("–", "-")
        .replace("—", "-")
        .replace("‐", "-")
    )
    return normalized.lower()


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize_pdf_text(text))


def _draw_word_highlight(page: fitz.Page, word_rows: list[tuple[fitz.Rect, int, int]]) -> None:
    if not word_rows:
        return
    # Merge words on the same text line so the highlight looks like a paragraph mark,
    # not a noisy set of disconnected boxes.
    grouped: dict[tuple[int, int], fitz.Rect] = {}
    for rect, block_no, line_no in word_rows[:42]:
        key = (block_no, line_no)
        grouped[key] = rect if key not in grouped else grouped[key] | rect
    for rect in grouped.values():
        page.draw_rect(rect + (-1.5, -1.0, 1.5, 1.0), color=(0.12, 0.45, 1), fill=(0.58, 0.76, 1), fill_opacity=0.38, width=0.9)


def _highlight_quote_words(page: fitz.Page, quote: str) -> None:
    quote_tokens = _tokens(quote)
    if len(quote_tokens) < 5:
        return

    page_words_raw = page.get_text("words")
    page_words: list[tuple[str, fitz.Rect, int, int]] = []
    for row in page_words_raw:
        if len(row) < 8:
            continue
        token_list = _tokens(str(row[4]))
        if not token_list:
            continue
        # Keep one rectangle for split punctuation variants. Most PDF words map to one token.
        for token in token_list:
            page_words.append((token, fitz.Rect(row[:4]), int(row[5]), int(row[6])))

    page_tokens = [token for token, _, _, _ in page_words]
    if not page_tokens:
        return

    # First try exact ordered snippets from the quote. Starting with shorter snippets
    # makes highlights robust when the citation quote spans multiple PDF text blocks.
    max_quote_start = min(max(len(quote_tokens) - 5, 0), 40)
    for window_size in range(min(18, len(quote_tokens)), 5, -1):
        for quote_start in range(max_quote_start + 1):
            snippet = quote_tokens[quote_start : quote_start + window_size]
            if len(snippet) < window_size:
                continue
            for page_start in range(0, len(page_tokens) - window_size + 1):
                if page_tokens[page_start : page_start + window_size] == snippet:
                    matches = [
                        (page_words[index][1], page_words[index][2], page_words[index][3])
                        for index in range(page_start, page_start + window_size)
                    ]
                    _draw_word_highlight(page, matches)
                    return

    # Fuzzy fallback: use a strong overlap window when exact matching fails because
    # extraction split or joined a few words differently.
    best_score = 0.0
    best_rows: list[tuple[fitz.Rect, int, int]] = []
    target = quote_tokens[: min(len(quote_tokens), 22)]
    target_set = set(target)
    window_size = min(max(len(target), 8), 24)
    for page_start in range(0, max(len(page_tokens) - window_size + 1, 0)):
        window = page_tokens[page_start : page_start + window_size]
        overlap = len(target_set.intersection(window))
        ordered = sum(1 for left, right in zip(target, window) if left == right)
        score = overlap / max(len(target_set), 1) + ordered / max(len(target), 1)
        if score > best_score:
            best_score = score
            best_rows = [
                (page_words[index][1], page_words[index][2], page_words[index][3])
                for index in range(page_start, page_start + window_size)
            ]
    if best_score >= 0.72:
        _draw_word_highlight(page, best_rows)


def render_page_png(pdf_path: Path, page_number: int, zoom: float = 1.8, highlight: str = "") -> bytes:
    if not pdf_path.exists():
        raise RuntimeError("Local PDF does not exist")

    safe_zoom = min(max(zoom, 0.8), 3.0)
    with fitz.open(pdf_path) as document:
        if page_number < 1 or page_number > document.page_count:
            raise RuntimeError("Requested PDF page is out of range")
        page = document.load_page(page_number - 1)
        _highlight_quote(page, highlight)
        matrix = fitz.Matrix(safe_zoom, safe_zoom)
        pixmap = page.get_pixmap(matrix=matrix, alpha=False)
        return pixmap.tobytes("png")
