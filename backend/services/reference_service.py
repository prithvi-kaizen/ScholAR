"""
reference_service.py
--------------------
Resolves a paper's bibliography using:
  1. Semantic Scholar API by arXiv ID (primary, for arXiv papers)
  2. Semantic Scholar API by title search (for uploaded PDFs)
  3. arXiv title-search fallback (when S2 is unavailable)

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

from backend.services.network_policy_service import NetworkPolicyService
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
    """Rate-limited GET against the S2 API. Returns None on HTTP errors."""
    NetworkPolicyService.require_acquisition("resolve-references", url)
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
# Raw-reference text extraction (heuristic)
# Supports both numbered [N] and author-year styles.
# ---------------------------------------------------------------------------

# Numbered style: [1] Vaswani...  or  1. Vaswani...
_REF_NUMBERED_RE = re.compile(
    r"^\s*\[?(\d{1,3})\]?[.\s]+([A-Z].{8,})",
    re.MULTILINE,
)

# Author-year style: Vaswani, A., ... (2017). Attention is all...
_REF_AUTHOR_YEAR_RE = re.compile(
    r"([A-Z][a-z]+(?:,\s+[A-Z]\.?)+.*?\(\d{4}\)[.,]\s+)([A-Z][^.]{10,}\.)",
    re.MULTILINE,
)


def _extract_raw_ref_numbers(pages: list[dict[str, Any]]) -> dict[int, str]:
    """Return {number: raw_title_snippet} from the last 6 pages of the paper.

    Handles both numbered [N] and author-year reference styles.
    """
    last_pages = pages[-6:] if len(pages) >= 6 else pages
    text = " ".join(p.get("text", "") for p in last_pages)
    found: dict[int, str] = {}

    # Try numbered style first
    for match in _REF_NUMBERED_RE.finditer(text):
        num = int(match.group(1))
        snippet = match.group(2)[:120]
        if num not in found:
            found[num] = snippet

    # If numbered style found very little, also try author-year
    if len(found) < 5:
        for i, match in enumerate(_REF_AUTHOR_YEAR_RE.finditer(text), start=1):
            title_snippet = match.group(2)[:120]
            key = len(found) + i
            if key not in found:
                found[key] = title_snippet

    return found


# ---------------------------------------------------------------------------
# Semantic Scholar primary resolution (by arXiv ID)
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
# Semantic Scholar resolution by title (for uploaded PDFs)
# ---------------------------------------------------------------------------

async def _resolve_via_s2_title(title: str) -> list[dict[str, Any]]:
    """Search S2 for a paper by title, then fetch its references.

    This enables multi-doc mode for uploaded PDFs that are indexed by S2
    (most NLP papers published after ~2010 are).
    """
    if not title or len(title.strip()) < 10:
        return []

    # Step 1: find the paper on S2 by title
    search_url = f"{S2_BASE}/paper/search"
    search_params = {
        "query": title.strip()[:200],
        "fields": "paperId,title,externalIds",
        "limit": 5,
    }
    search_data = await _s2_get(search_url, search_params)
    if not search_data or not search_data.get("data"):
        return []

    # Pick the best match: first result whose title is a close substring match
    s2_paper_id: str | None = None
    title_lower = title.lower().strip()
    for candidate in search_data["data"]:
        cand_title = (candidate.get("title") or "").lower().strip()
        # Accept if ≥60% of the query title's words appear in the candidate
        query_words = set(re.findall(r"[a-z]{4,}", title_lower))
        cand_words  = set(re.findall(r"[a-z]{4,}", cand_title))
        if query_words and len(query_words & cand_words) / len(query_words) >= 0.6:
            s2_paper_id = candidate.get("paperId")
            break

    if not s2_paper_id:
        # Fall back to first result
        s2_paper_id = search_data["data"][0].get("paperId")

    if not s2_paper_id:
        return []

    # Step 2: fetch references for this paper
    ref_url    = f"{S2_BASE}/paper/{s2_paper_id}/references"
    ref_params = {"fields": S2_REF_FIELDS, "limit": 100}
    data = await _s2_get(ref_url, ref_params)
    if not data or not data.get("data"):
        return []

    refs: list[dict[str, Any]] = []
    for index, item in enumerate(data["data"], start=1):
        entry = _build_ref_entry(item, ref_number=index)
        if entry["title"]:
            refs.append(entry)

    return refs


# ---------------------------------------------------------------------------
# arXiv title-search fallback
# ---------------------------------------------------------------------------

async def _resolve_via_arxiv_fallback(
    raw_snippets: dict[int, str],
) -> list[dict[str, Any]]:
    """
    For each raw reference text snippet, try an arXiv title search.
    Used when S2 is completely unavailable.
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
      2. arXiv papers: use Semantic Scholar by arXiv ID (fastest, most complete).
      3. Uploaded PDFs: use Semantic Scholar by title search.
      4. Fall back to arXiv title-search on heuristic raw-reference snippets.

    Always saves result to disk before returning.
    """
    if not force and references_path(local_id).exists():
        # A cached file may legitimately be an empty list (paper has zero
        # resolvable references) — an empty list is falsy, so check file
        # existence rather than truthiness or this cache never "sticks".
        return load_references(local_id)

    NetworkPolicyService.require_acquisition("resolve-references")

    arxiv_id = metadata.get("id", "")
    is_upload = metadata.get("source") == "upload"
    paper_title = metadata.get("title", "")

    # Extract raw reference snippets for ordering + fallback
    raw_snippets = _extract_raw_ref_numbers(pages)

    refs: list[dict[str, Any]] = []

    # --- Primary: Semantic Scholar by arXiv ID (arXiv papers) ---
    if arxiv_id and not is_upload:
        refs = await _resolve_via_s2(arxiv_id)

    # --- Secondary: Semantic Scholar by title (uploaded PDFs) ---
    if not refs and paper_title:
        refs = await _resolve_via_s2_title(paper_title)

    # --- Fallback: arXiv title search on raw text snippets ---
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


def traverse_citation_graph(
    paper_id: str,
    query_entities: list[str],
    chunks: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Traverse bibliography references to find anchor text and linked cited works (PaperQA2 style).

    Args:
        paper_id: The anchor document ID.
        query_entities: Target entity/author/model names mentioned in the multi-hop query.
        chunks: List of document chunks to scan for in-text citation anchors.

    Returns:
        List of matched citation records containing target paper metadata and anchor citing passages.
    """
    refs = load_references(paper_id)
    if not refs or not query_entities:
        return []

    matched_records: list[dict[str, Any]] = []
    clean_entities = [e.strip().lower() for e in query_entities if len(e.strip()) >= 3]

    for ref_idx, ref in enumerate(refs, start=1):
        title = (ref.get("title") or "").lower()
        authors = " ".join(ref.get("authors") or []).lower()
        arxiv_id = (ref.get("arxiv_id") or "").lower()
        combined_ref = f"{title} {authors} {arxiv_id}"

        # Match against query entities
        matched_entity = next((e for e in clean_entities if e in combined_ref), None)
        if not matched_entity:
            continue

        # Look for in-text citations in chunks (e.g. "[1]", "[2]", author names, title fragments)
        citing_chunks: list[dict[str, Any]] = []
        author_last = (ref.get("authors") or [""])[0].split()[-1].lower() if ref.get("authors") else ""
        cite_patterns = [
            rf"\[{ref_idx}\]",
            rf"\b{re.escape(author_last)}\b" if len(author_last) >= 3 else r"(?!)",
            rf"\b{re.escape(matched_entity)}\b",
        ]

        for chunk in chunks:
            chunk_text = chunk.get("text", "")
            if any(re.search(pat, chunk_text, re.IGNORECASE) for pat in cite_patterns):
                citing_chunks.append({
                    "chunk_id": chunk.get("chunk_id"),
                    "page": chunk.get("page"),
                    "section": chunk.get("section_title") or chunk.get("section"),
                    "quote": chunk_text[:280].strip(),
                })

        matched_records.append({
            "ref_index": ref_idx,
            "matched_entity": matched_entity,
            "target_title": ref.get("title"),
            "target_authors": ref.get("authors", []),
            "target_year": ref.get("year"),
            "target_arxiv_id": ref.get("arxiv_id"),
            "is_ingested": ref.get("ingested", False),
            "secondary_local_id": ref.get("secondary_local_id"),
            "citing_anchors": citing_chunks[:3],
        })

    return matched_records
