from __future__ import annotations

import html
import os
import re
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

ENABLE_WEB_SEARCH = os.getenv("ENABLE_WEB_SEARCH", "true").lower() not in {"0", "false", "no"}
WEB_SEARCH_RESULTS = int(os.getenv("WEB_SEARCH_RESULTS", "5"))


def web_search_enabled() -> bool:
    return ENABLE_WEB_SEARCH


def _clean_text(value: str) -> str:
    value = re.sub(r"<[^>]+>", " ", value)
    value = html.unescape(value)
    return re.sub(r"\s+", " ", value).strip()


def _clean_url(url: str) -> str:
    url = html.unescape(url)
    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "uddg" in query and query["uddg"]:
        return unquote(query["uddg"][0])
    if url.startswith("//"):
        return f"https:{url}"
    return url


async def search_web(query: str, limit: int | None = None) -> list[dict[str, Any]]:
    if not ENABLE_WEB_SEARCH or not query.strip():
        return []

    max_results = max(1, min(limit or WEB_SEARCH_RESULTS, 8))
    url = f"https://duckduckgo.com/html/?q={quote_plus(query)}"
    headers = {
        "User-Agent": "Mozilla/5.0 ScholAR local research assistant",
        "Accept": "text/html,application/xhtml+xml",
    }

    try:
        async with httpx.AsyncClient(timeout=12.0, follow_redirects=True, headers=headers) as client:
            response = await client.get(url)
            response.raise_for_status()
    except httpx.HTTPError:
        return []

    html_text = response.text
    results: list[dict[str, Any]] = []
    blocks = re.split(r'class="result\b', html_text)
    for block in blocks[1:]:
        link_match = re.search(r'class="result__a"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', block, re.S)
        if not link_match:
            continue
        snippet_match = re.search(r'class="result__snippet"[^>]*>(.*?)</a>|class="result__snippet"[^>]*>(.*?)</', block, re.S)
        title = _clean_text(link_match.group(2))
        result_url = _clean_url(link_match.group(1))
        snippet = _clean_text((snippet_match.group(1) or snippet_match.group(2)) if snippet_match else "")
        if not title or not result_url.startswith(("http://", "https://")):
            continue
        results.append(
            {
                "id": f"web_{len(results) + 1}",
                "title": title,
                "url": result_url,
                "snippet": snippet,
            }
        )
        if len(results) >= max_results:
            break
    return results
