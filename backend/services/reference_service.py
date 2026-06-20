"""
reference_service.py
--------------------
Resolves a paper's bibliography using:
  1. Semantic Scholar API (primary, keyless, rate-limited)
  2. arXiv title-search fallback (for papers not indexed by S2, or uploads)

Caches results to backend/data/papers/<local_id>/references.json.
"""
from __future__ import annotations

import asyncio
import json
import re
import time
from pathlib import Path
from typing import Any

import httpx

from backend.services.pdf_service import paper_dir, read_json, write_json

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

S2_BASE = "https://api.semanticscholar.org/graph/v1"
S2_REF_FIELDS = "title,authors,abstract,externalIds,year,openAccessPdf,url"
S2_MIN_INTERVAL = 1.1          # seconds between S2 requests (polite, keyless)
ARXIV_PDF_BASE  = "https://arxiv.org/pdf/"
ARXIV_ABS_BASE  = "https://arxiv.org/abs/"

_s2_lock = asyncio.Lock()
_s2_last_at = 0.0


# ---------------------------------------------------------------------------
# Reference cache helpers
# ---------------------------------------------------------------------------

def references_path(local_id: str) -> Path:
    return paper_dir(local_id) / "references.json"


def load_references(local_id: str) -> list[dict[str, Any]]:
    path = references_path(local_id)
    return read_json(path) if path.exists() else []


def save_references(local_id: str, refs: list[dict[str, Any]]) -> None:
    write_json(references_path(local_id), refs)


# ---------------------------------------------------------------------------
# Semantic Scholar helpers
# ---------------------------------------------------------------------------

async def _s2_get(url: str, params: dict[str, Any] | None = None) -> dict[str, Any] | None:
    """Rate-limited GET against the S2 API. Returns None on any error."""
    global _s2_last_at
    async with _s2_lock:
        elapsed = time.monotonic() - _s2_last_at
        if elapsed < S2_MIN_INTERVAL:
            await asyncio.sleep(S2_MIN_INTERVAL - elapsed)

        try:
            async with httpx.AsyncClient(timeout=12.0, follow_redirects=True) as client:
                response = await client.get(url, params=params or {})
                _s2_last_at = time.monotonic()
                if response.status_code == 200:
                    return response.json()
                if response.status_code == 429:
                    # Back off and retry once
                    await asyncio.sleep(3.0)
                    response = await client.get(url, params=params or {})
                    if response.status_code == 200:
                        return response.json()
                return None
        except httpx.HTTPError:
            return None


def _arxiv_id_from_external(external_ids: dict[str, Any]) -> str:
    """Extract a clean arXiv ID from S2's externalIds dict."""
    raw = external_ids.get("ArXiv") or ""
    return re.sub(r"^arxiv:", "", raw, flags=re.IGNORECASE).strip()


def _build_ref_entry(
    s2_data: dict[str, Any],
    ref_number: int,
    ingested: bool = False,
    secondary_local_id: str | None = None,
) -> dict[str, Any]:
    """Normalise a single S2 reference object into our schema."""
    paper   = s2_data.get("citedPaper", s2_data)
    ext_ids = paper.get("externalIds") or {}
    arxiv_id = _arxiv_id_from_external(ext_ids)
    pdf_url = ""
    abs_url = ""
    if arxiv_id:
        pdf_url = f"{ARXIV_PDF_BASE}{arxiv_id}"
        abs_url = f"{ARXIV_ABS_BASE}{arxiv_id}"
    elif paper.get("openAccessPdf"):
        pdf_url = paper["openAccessPdf"].get("url", "")

    authors = [
        a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")
    ]

    return {
        "ref_number":          ref_number,
        "title":               (paper.get("title") or "").strip(),
        "authors":             authors,
        "year":                str(paper.get("year") or ""),
        "abstract":            (paper.get("abstract") or "")[:1200].strip(),
        "arxiv_id":            arxiv_id or None,
        "s2_paper_id":         paper.get("paperId") or None,
        "pdf_url":             pdf_url,
        "abs_url":             abs_url,
        "ingested":            ingested,
        "secondary_local_id":  secondary_local_id,
    }


# ---------------------------------------------------------------------------
# Raw-reference text extraction (heuristic, for arXiv papers)
# Used as an ordering signal and ref-number source, not for title matching.
# ---------------------------------------------------------------------------

_REF_LINE_RE = re.compile(
    r"^\s*\[?(\d{1,3})\]?\s+([A-Z].{10,})",
    re.MULTILINE,
)


def _extract_raw_ref_numbers(pages: list[dict[str, Any]]) -> dict[int, str]:
    """Return {number: raw_title_snippet} from the last 4 pages of the paper."""
    last_pages = pages[-4:] if len(pages) >= 4 else pages
    text = " ".join(p.get("text", "") for p in last_pages)
    found: dict[int, str] = {}
    for match in _REF_LINE_RE.finditer(text):
        num  = int(match.group(1))
        snippet = match.group(2)[:120]
        if num not in found:
            found[num] = snippet
    return found


# ---------------------------------------------------------------------------
# Semantic Scholar primary resolution
# ---------------------------------------------------------------------------

async def _resolve_via_s2(arxiv_id: str) -> list[dict[str, Any]]:
    """Fetch references for a paper identified by its arXiv ID from S2."""
    s2_paper_id = f"arXiv:{arxiv_id}"
    url    = f"{S2_BASE}/paper/{s2_paper_id}/references"
    params = {"fields": S2_REF_FIELDS, "limit": 100}

    data = await _s2_get(url, params)
    if not data or not data.get("data"):
        return []

    refs: list[dict[str, Any]] = []
    for index, item in enumerate(data["data"], start=1):
        entry = _build_ref_entry(item, ref_number=index)
        if entry["title"]:
            refs.append(entry)

    # Fetch additional pages if S2 provides offset
    offset = data.get("next")
    page   = 1
    while offset and page < 5:  # cap at 500 references
        params["offset"] = offset
        data = await _s2_get(url, params)
        if not data or not data.get("data"):
            break
        for item in data["data"]:
            entry = _build_ref_entry(item, ref_number=len(refs) + 1)
            if entry["title"]:
                refs.append(entry)
        offset = data.get("next")
        page  += 1

    return refs


# ---------------------------------------------------------------------------
# arXiv title-search fallback
# ---------------------------------------------------------------------------

async def _resolve_via_arxiv_fallback(
    raw_snippets: dict[int, str],
) -> list[dict[str, Any]]:
    """
    For each raw reference text snippet, try an arXiv title search.
    Used when S2 is unavailable or the paper is an uploaded PDF.
    Intentionally limited to the first 20 references to stay within
    rate limits without a key.
    """
    from backend.services.arxiv_service import search_arxiv  # local import to avoid circular

    refs: list[dict[str, Any]] = []
    for num in sorted(raw_snippets.keys())[:20]:
        snippet = raw_snippets[num]
        try:
            results = await search_arxiv(snippet, max_results=1)
        except RuntimeError:
            continue
        if not results:
            continue
        paper = results[0]
        refs.append(
            {
                "ref_number":          num,
                "title":               paper.get("title", ""),
                "authors":             paper.get("authors", []),
                "year":                paper.get("year", ""),
                "abstract":            paper.get("summary", "")[:1200],
                "arxiv_id":            paper.get("id") or None,
                "s2_paper_id":         None,
                "pdf_url":             paper.get("pdf_url", ""),
                "abs_url":             paper.get("abs_url", ""),
                "ingested":            False,
                "secondary_local_id":  None,
            }
        )
    return refs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def resolve_references(
    local_id: str,
    metadata: dict[str, Any],
    pages: list[dict[str, Any]],
    force: bool = False,
) -> list[dict[str, Any]]:
    """
    Resolve a paper's bibliography and cache the result.

    Priority:
      1. Return cached references if present and `force` is False.
      2. Use Semantic Scholar (primary) for arXiv papers.
      3. Fall back to arXiv title-search on heuristic raw-reference snippets.

    Always saves result to disk before returning.
    """
    if not force:
        cached = load_references(local_id)
        if cached:
            return cached

    arxiv_id = metadata.get("id", "")
    is_upload = metadata.get("source") == "upload"

    # Extract raw reference snippets for ordering + fallback
    raw_snippets = _extract_raw_ref_numbers(pages)

    refs: list[dict[str, Any]] = []

    # --- Primary: Semantic Scholar ---
    if arxiv_id and not is_upload:
        refs = await _resolve_via_s2(arxiv_id)

    # --- Fallback: arXiv title search ---
    if not refs and raw_snippets:
        refs = await _resolve_via_arxiv_fallback(raw_snippets)

    # Deduplicate by title (case-insensitive)
    seen_titles: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for ref in refs:
        key = re.sub(r"\s+", " ", (ref["title"] or "").lower()).strip()
        if key and key not in seen_titles:
            seen_titles.add(key)
            deduped.append(ref)

    save_references(local_id, deduped)
    return deduped


def mark_reference_ingested(
    local_id: str,
    ref_index: int,
    secondary_local_id: str,
) -> None:
    """Update references.json to mark a specific reference as ingested."""
    refs = load_references(local_id)
    if 0 <= ref_index < len(refs):
        refs[ref_index]["ingested"]           = True
        refs[ref_index]["secondary_local_id"] = secondary_local_id
        save_references(local_id, refs)
