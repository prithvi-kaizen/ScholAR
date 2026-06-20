#!/usr/bin/env python3
"""
seed_eval_papers.py
-------------------
Pre-downloads and chunks the 10 secondary papers needed for the
multi-document evaluation benchmark.

It mirrors what the frontend does:
  1. Fetches paper metadata from arXiv API
  2. Calls POST /api/papers/prepare with the full metadata
     (which uses the backend's download pipeline)

Usage:
    # 1. Start the backend:
    #    cd backend && uvicorn main:app --reload
    #
    # 2. In a second terminal:
    python3 evaluation/seed_eval_papers.py

    # Custom backend URL:
    python3 evaluation/seed_eval_papers.py --backend http://localhost:8000
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]

# arXiv IDs of all secondary papers required by multidoc_benchmark.json
SECONDARY_IDS = [
    "1409.0473",   # Bahdanau attention
    "1412.6980",   # Adam optimizer
    "1508.07909",  # BPE / subword units
    "1512.03385",  # ResNet / residual connections
    "1607.06450",  # Layer normalization
    "1702.08734",  # FAISS billion-scale similarity search
    "1705.03551",  # TriviaQA
    "1901.08634",  # Natural Questions
    "1910.13461",  # BART
    "2004.04906",  # DPR dense passage retrieval
]

ARXIV_API = "https://export.arxiv.org/api/query"
ARXIV_NS  = "http://www.w3.org/2005/Atom"


async def fetch_arxiv_metadata(client: httpx.AsyncClient, arxiv_id: str) -> dict | None:
    """Fetch paper metadata from arXiv API (same as arxiv_service.py does)."""
    params = {"id_list": arxiv_id, "max_results": "1"}
    try:
        r = await client.get(ARXIV_API, params=params, timeout=15.0)
        r.raise_for_status()
    except httpx.HTTPError as exc:
        print(f"  arXiv API error for {arxiv_id}: {exc}")
        return None

    root = ET.fromstring(r.text)
    entry = root.find(f"{{{ARXIV_NS}}}entry")
    if entry is None:
        return None

    def _text(tag: str) -> str:
        el = entry.find(f"{{{ARXIV_NS}}}{tag}")
        return (el.text or "").strip() if el is not None else ""

    title   = re.sub(r"\s+", " ", _text("title"))
    summary = re.sub(r"\s+", " ", _text("summary"))

    authors = [
        re.sub(r"\s+", " ", (a.find(f"{{{ARXIV_NS}}}name") or a).text or "").strip()
        for a in entry.findall(f"{{{ARXIV_NS}}}author")
    ]

    published = _text("published")
    year = published[:4] if published else ""

    # PDF URL from arXiv API links
    pdf_url = ""
    for link in entry.findall(f"{{{ARXIV_NS}}}link"):
        if link.get("title") == "pdf":
            pdf_url = link.get("href", "")
            break
    if not pdf_url:
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}"

    categories = [
        c.get("term", "")
        for c in entry.findall("{http://arxiv.org/schemas/atom}primary_category")
        + entry.findall("{http://arxiv.org/schemas/atom}category")
    ]

    return {
        "id":         arxiv_id,
        "title":      title,
        "authors":    authors,
        "year":       year,
        "summary":    summary,
        "categories": [c for c in categories if c][:5],
        "pdf_url":    pdf_url,
        "abs_url":    f"https://arxiv.org/abs/{arxiv_id}",
        "published":  published,
        "source":     "arxiv",
    }


async def seed(backend_url: str) -> None:
    print(f"Seeding {len(SECONDARY_IDS)} secondary papers via {backend_url} …\n")

    async with httpx.AsyncClient(
        timeout=120.0,
        follow_redirects=True,
        headers={"User-Agent": "ScholAR-eval/1.0 (research; contact: see README)"},
    ) as client:
        for arxiv_id in SECONDARY_IDS:
            # Check if already on disk (skip the API call entirely)
            paper_dir = ROOT / "backend" / "data" / "papers" / arxiv_id
            if (paper_dir / "chunks.json").exists():
                n = len(json.loads((paper_dir / "chunks.json").read_text()))
                print(f"  ✓  {arxiv_id}  already ingested ({n} chunks)")
                continue

            # Fetch metadata from arXiv
            print(f"  →  {arxiv_id}  fetching metadata …", end=" ", flush=True)
            meta = await fetch_arxiv_metadata(client, arxiv_id)
            if not meta:
                print("SKIP (no metadata)")
                continue

            print(f"'{meta['title'][:50]}' … preparing …", end=" ", flush=True)

            # Call the backend prepare endpoint
            try:
                r = await client.post(f"{backend_url}/api/papers/prepare", json=meta)
                if r.status_code == 200:
                    payload = r.json()
                    n_chunks = payload.get("chunks", "?")
                    print(f"OK ({n_chunks} chunks)")
                else:
                    print(f"HTTP {r.status_code}: {r.text[:100]}")
            except httpx.HTTPError as exc:
                print(f"ERROR: {exc}")

            # Be polite to both arXiv and the backend
            await asyncio.sleep(1.5)

    print("\nDone. Now run: python3 evaluation/run_multidoc_eval.py --no-ingest")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pre-seed secondary papers for multi-doc eval")
    parser.add_argument("--backend", default="http://localhost:8000", help="Backend base URL")
    args = parser.parse_args()
    asyncio.run(seed(args.backend))
