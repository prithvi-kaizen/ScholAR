from __future__ import annotations

import asyncio
import json
import re
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import httpx

from backend.services.network_policy_service import NetworkPolicyService


ARXIV_API_URL = "https://export.arxiv.org/api/query"
ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
ARXIV_HEADERS = {
    "User-Agent": "ScholAR/0.1 (local research assistant; contact: local@example.com)"
}
BASE_DIR = Path(__file__).resolve().parents[1]
SEARCH_CACHE_PATH = BASE_DIR / "data" / "search_cache.json"
MIN_REQUEST_INTERVAL_SECONDS = 3.1
CACHE_VERSION = "v3"
SEARCH_STOP_WORDS = {
    "a",
    "all",
    "an",
    "and",
    "about",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "no",
    "of",
    "on",
    "or",
    "paper",
    "papers",
    "the",
    "this",
    "to",
    "with",
    "you",
}

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
    return f"{CACHE_VERSION}::{query.strip().lower()}::{max_results}"


def _normalize(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def _query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    cleaned: list[str] = []
    for token in tokens:
        if token in SEARCH_STOP_WORDS or len(token) <= 1:
            continue
        if token.endswith("s") and len(token) > 3:
            token = token[:-1]
        cleaned.append(token)
    return cleaned


def _api_query_tokens(query: str) -> list[str]:
    tokens = re.findall(r"[a-z0-9]+", query.lower())
    return [token for token in tokens if token not in SEARCH_STOP_WORDS and len(token) > 1]


def _build_search_queries(query: str) -> list[str]:
    tokens = _api_query_tokens(query)
    if not tokens:
        return [f'all:"{query.strip()}"']

    unique_tokens = list(dict.fromkeys(tokens))
    queries = [" AND ".join(f"all:{token}" for token in unique_tokens[:8])]
    if len(unique_tokens) > 3:
        queries.append(" AND ".join(f"all:{token}" for token in unique_tokens[:3]))
    if len(unique_tokens) > 1:
        queries.append(" OR ".join(f"ti:{token}" for token in unique_tokens[:5]))
    return list(dict.fromkeys(queries))


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
    query_terms = set(_query_tokens(query))
    if not query_terms:
        return []

    scored: list[tuple[int, dict[str, Any]]] = []
    for paper in FALLBACK_PAPERS:
        title_terms = set(_query_tokens(paper["title"]))
        matched_title_terms = query_terms.intersection(title_terms)
        coverage = len(matched_title_terms) / max(len(query_terms), 1)
        if coverage >= 0.7:
            score = len(matched_title_terms)
            scored.append((score, paper))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in scored[:max_results]]


def _local_prepared_papers() -> list[dict[str, Any]]:
    papers_dir = BASE_DIR / "data" / "papers"
    if not papers_dir.exists():
        return []

    papers: list[dict[str, Any]] = []
    for metadata_path in papers_dir.glob("*/metadata.json"):
        try:
            with metadata_path.open("r", encoding="utf-8") as handle:
                metadata = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(metadata, dict) and metadata.get("title"):
            papers.append(metadata)
    return papers


def _exact_title_matches(query: str, max_results: int) -> list[dict[str, Any]]:
    query_terms = set(_query_tokens(query))
    if len(query_terms) < 2:
        return []

    query_norm = _normalize(query)
    candidates = [*_local_prepared_papers(), *FALLBACK_PAPERS]
    scored: list[tuple[float, dict[str, Any]]] = []

    for paper in candidates:
        title = paper.get("title", "")
        title_norm = _normalize(title)
        title_terms = set(_query_tokens(title))
        if not title_terms:
            continue
        coverage = len(query_terms.intersection(title_terms)) / max(len(query_terms), 1)
        title_coverage = len(query_terms.intersection(title_terms)) / max(len(title_terms), 1)
        score = coverage + title_coverage
        if query_norm == title_norm:
            score += 10.0
        elif query_norm in title_norm or title_norm in query_norm:
            score += 4.0
        if coverage >= 0.8 and title_coverage >= 0.45:
            scored.append((score, paper))

    scored.sort(key=lambda item: item[0], reverse=True)
    return [paper for _, paper in scored[:max_results]]


def _merge_unique(primary: list[dict[str, Any]], secondary: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    seen: set[str] = set()
    merged: list[dict[str, Any]] = []
    for paper in [*primary, *secondary]:
        paper_id = paper.get("id") or paper.get("abs_url") or paper.get("title")
        if not paper_id or paper_id in seen:
            continue
        seen.add(paper_id)
        merged.append(paper)
        if len(merged) >= max_results:
            break
    return merged


async def _rate_limited_get(params: dict[str, Any]) -> httpx.Response:
    NetworkPolicyService.require_acquisition("search-arxiv", ARXIV_API_URL)
    global _last_request_at

    async with _request_lock:
        elapsed = time.monotonic() - _last_request_at
        if elapsed < MIN_REQUEST_INTERVAL_SECONDS:
            await asyncio.sleep(MIN_REQUEST_INTERVAL_SECONDS - elapsed)

        async with httpx.AsyncClient(timeout=15.0, follow_redirects=True) as client:
            response = await client.get(ARXIV_API_URL, params=params, headers=ARXIV_HEADERS)

        _last_request_at = time.monotonic()
        return response


def _score_paper(query: str, paper: dict[str, Any]) -> tuple[float, int]:
    tokens = _query_tokens(query)
    if not tokens:
        return (0.0, 0)

    query_norm = _normalize(query)
    title_norm = _normalize(paper.get("title", ""))
    summary_norm = _normalize(paper.get("summary", ""))
    authors_norm = _normalize(" ".join(paper.get("authors", [])))
    categories_norm = _normalize(" ".join(paper.get("categories", [])))
    paper_id = _normalize(paper.get("id", ""))

    title_tokens = set(_query_tokens(paper.get("title", "")))
    summary_tokens = set(_query_tokens(paper.get("summary", "")))
    author_tokens = set(_query_tokens(authors_norm))
    category_tokens = set(_query_tokens(categories_norm))

    title_matches = set(tokens).intersection(title_tokens)
    summary_matches = set(tokens).intersection(summary_tokens)
    author_matches = set(tokens).intersection(author_tokens)
    category_matches = set(tokens).intersection(category_tokens)
    matched_terms = len(title_matches | summary_matches | author_matches | category_matches)

    score = 0.0
    if query_norm and (query_norm == title_norm or query_norm == paper_id):
        score += 200.0
    elif query_norm and query_norm in title_norm:
        score += 90.0
    elif title_norm and title_norm in query_norm:
        score += 70.0

    score += len(title_matches) * 14.0
    score += len(summary_matches) * 3.0
    score += len(author_matches) * 8.0
    score += len(category_matches) * 5.0

    title_coverage = len(title_matches) / max(len(set(tokens)), 1)
    field_coverage = matched_terms / max(len(set(tokens)), 1)
    score += title_coverage * 35.0
    score += field_coverage * 12.0

    if len(tokens) >= 2 and matched_terms < 2:
        score -= 40.0

    return (score, matched_terms)


def _rerank_papers(query: str, papers: list[dict[str, Any]], max_results: int) -> list[dict[str, Any]]:
    scored: list[tuple[float, int, dict[str, Any]]] = []
    for paper in papers:
        score, matched_terms = _score_paper(query, paper)
        if score >= 8.0 and matched_terms > 0:
            scored.append((score, matched_terms, paper))

    scored.sort(key=lambda item: (item[0], item[1], item[2].get("published", "")), reverse=True)
    return [paper for _, _, paper in scored[:max_results]]


def _parse_arxiv_response(response_text: str) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(response_text)
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
    return papers


async def search_arxiv(query: str, max_results: int = 12) -> list[dict[str, Any]]:
    if not query.strip():
        return []

    safe_max = max(1, min(max_results, 25))
    fetch_max = max(30, safe_max * 3)
    key = _cache_key(query, safe_max)
    cache = _read_cache()
    search_errors: list[Exception] = []
    ranked_papers: list[dict[str, Any]] = []
    rate_limited = False
    exact_matches = _exact_title_matches(query, safe_max)
    if exact_matches:
        exact_matches = _merge_unique(exact_matches, [], safe_max)
        cache[key] = {"created_at": int(time.time()), "papers": exact_matches}
        _write_cache(cache)
        return exact_matches

    for search_query in _build_search_queries(query):
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": min(fetch_max, 50),
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        try:
            response = await _rate_limited_get(params)
            if response.status_code == 429:
                await asyncio.sleep(MIN_REQUEST_INTERVAL_SECONDS)
                response = await _rate_limited_get(params)
            if response.status_code == 429:
                rate_limited = True
                break
            response.raise_for_status()
            papers = _parse_arxiv_response(response.text)
            ranked_papers = _rerank_papers(query, papers, safe_max)
            if ranked_papers:
                break
        except (httpx.HTTPError, RuntimeError) as exc:
            search_errors.append(exc)
            continue

    if exact_matches:
        ranked_papers = _merge_unique(exact_matches, ranked_papers, safe_max)

    if not ranked_papers:
        if key in cache and cache[key].get("papers"):
            return cache[key]["papers"]
        fallback = _fallback_search(query, safe_max)
        if fallback:
            return fallback
        if rate_limited:
            raise RuntimeError("arXiv is rate limiting searches right now. Please wait a few seconds and try again.")
        if search_errors:
            raise RuntimeError("arXiv search is temporarily unavailable. Please try again shortly.") from search_errors[-1]

    cache[key] = {"created_at": int(time.time()), "papers": ranked_papers}
    _write_cache(cache)
    return ranked_papers
