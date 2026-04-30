from __future__ import annotations

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import quote_plus

import httpx


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_HEADERS = {
    "User-Agent": "ScholAR/0.1 (local research assistant; contact: local@example.com)"
}
BASE_DIR = Path(__file__).resolve().parents[1]
SEARCH_CACHE_PATH = BASE_DIR / "data" / "search_cache.json"
MIN_REQUEST_INTERVAL_SECONDS = 3.1

_request_lock = asyncio.Lock()
_last_request_at = 0.0

FALLBACK_PAPERS: list[dict[str, Any]] = [
    {
        "id": "2512.24601",
        "title": "Recursive Language Models",
        "authors": ["Alex L. Zhang", "Tim Kraska", "Omar Khattab"],
        "year": "2025",
        "published": "2025-12-31T00:00:00Z",
        "summary": (
            "Recursive Language Models treat long prompts as part of an external environment, "
            "allowing an LLM to examine, decompose, and recursively call itself over prompt snippets "
            "to handle inputs far beyond normal context windows."
        ),
        "categories": ["cs.CL", "cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2512.24601",
        "abs_url": "https://arxiv.org/abs/2512.24601",
    },
    {
        "id": "1706.03762",
        "title": "Attention Is All You Need",
        "authors": ["Ashish Vaswani", "Noam Shazeer", "Niki Parmar"],
        "year": "2017",
        "published": "2017-06-12T00:00:00Z",
        "summary": "The Transformer architecture replaces recurrence with attention mechanisms for sequence modeling.",
        "categories": ["cs.CL", "cs.LG"],
        "pdf_url": "https://arxiv.org/pdf/1706.03762",
        "abs_url": "https://arxiv.org/abs/1706.03762",
    },
    {
        "id": "2005.11401",
        "title": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        "authors": ["Patrick Lewis", "Ethan Perez", "Aleksandra Piktus"],
        "year": "2020",
        "published": "2020-05-22T00:00:00Z",
        "summary": "RAG combines retrieval with neural generation for knowledge-intensive NLP tasks.",
        "categories": ["cs.CL", "cs.AI"],
        "pdf_url": "https://arxiv.org/pdf/2005.11401",
        "abs_url": "https://arxiv.org/abs/2005.11401",
    },
]


def clean_text(value: str | None) -> str:
    """Normalize arXiv's frequent line breaks and repeated spaces."""
    return re.sub(r"\s+", " ", value or "").strip()


def _entry_text(entry: ET.Element, name: str) -> str:
    node = entry.find(f"atom:{name}", ATOM_NS)
    return clean_text(node.text if node is not None else "")


def _paper_id(abs_url: str) -> str:
    return abs_url.rstrip("/").split("/")[-1]


def _cache_key(query: str, max_results: int) -> str:
    return f"{query.strip().lower()}::{max_results}"


def _read_cache() -> dict[str, Any]:
    if not SEARCH_CACHE_PATH.exists():
        return {}
    try:
        with SEARCH_CACHE_PATH.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError):
        return {}


def _write_cache(cache: dict[str, Any]) -> None:
    SEARCH_CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with SEARCH_CACHE_PATH.open("w", encoding="utf-8") as handle:
        json.dump(cache, handle, indent=2, ensure_ascii=False)


def _fallback_search(query: str, max_results: int) -> list[dict[str, Any]]:
    query_terms = {term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) > 2}
    if not query_terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for paper in FALLBACK_PAPERS:
        haystack = " ".join(
            [
                paper["id"],
                paper["title"],
                " ".join(paper["authors"]),
                paper["summary"],
                " ".join(paper["categories"]),
            ]
        ).lower()
        score = sum(1 for term in query_terms if term in haystack)
        if score:
            scored.append((score, paper))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in scored[:max_results]]


async def _rate_limited_get(url: str) -> httpx.Response:
    global _last_request_at

    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            await asyncio.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

        async with httpx.AsyncClient(timeout=8.0, follow_redirects=True) as client:
            response = await client.get(url, headers=ARXIV_HEADERS)

        _last_request_at = time.monotonic()
        return response


async def search_arxiv(query: str, max_results: int = 12) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    safe_max = max(1, min(max_results, 25))
    key = _cache_key(query, safe_max)
    cache = _read_cache()
    params = (
        f"search_query=all:{quote_plus(query.strip())}"
        f"&start=0&max_results={safe_max}&sortBy=relevance&sortOrder=descending"
    )
    url = f"{ARXIV_API_URL}?{params}"

    try:
        response = await _rate_limited_get(url)
        if response.status_code == 429:
            await asyncio.sleep(MIN_REQUEST_INTERVAL_SECONDS)
            response = await _rate_limited_get(url)
        if response.status_code == 429 and key in cache:
            return cache[key]["papers"]
        if response.status_code == 429:
            fallback = _fallback_search(query, safe_max)
            if fallback:
                return fallback
            raise RuntimeError("arXiv is rate limiting searches right now. Please wait a few seconds and try again.")
        response.raise_for_status()
    except httpx.HTTPError as exc:
        if key in cache:
            return cache[key]["papers"]
        fallback = _fallback_search(query, safe_max)
        if fallback:
            return fallback
        raise RuntimeError("arXiv search is temporarily unavailable. Please try again shortly.") from exc

    try:
        root = ET.fromstring(response.text)
    except ET.ParseError as exc:
        raise RuntimeError("arXiv returned invalid XML") from exc

    papers: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", ATOM_NS):
        abs_url = _entry_text(entry, "id")
        published = _entry_text(entry, "published")
        authors = [
            clean_text(author.findtext("atom:name", default="", namespaces=ATOM_NS))
            for author in entry.findall("atom:author", ATOM_NS)
        ]
        categories = [
            category.attrib.get("term", "")
            for category in entry.findall("atom:category", ATOM_NS)
            if category.attrib.get("term")
        ]

        pdf_url = ""
        for link in entry.findall("atom:link", ATOM_NS):
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = link.attrib.get("href", "")
                break
        if not pdf_url and abs_url:
            pdf_url = abs_url.replace("/abs/", "/pdf/")

        papers.append(
            {
                "id": _paper_id(abs_url),
                "title": _entry_text(entry, "title"),
                "authors": [author for author in authors if author],
                "year": published[:4] if published else "",
                "published": published,
                "summary": _entry_text(entry, "summary"),
                "categories": categories,
                "pdf_url": pdf_url,
                "abs_url": abs_url,
            }
        )

    cache[key] = {"created_at": int(time.time()), "papers": papers}
    _write_cache(cache)
    return papers
