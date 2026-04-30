from __future__ import annotations

import json
import os
import re
from typing import Any

import httpx
from dotenv import load_dotenv


load_dotenv()

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen3:9b")


async def ollama_available() -> bool:
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            response = await client.get(f"{OLLAMA_BASE_URL}/api/tags")
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def generate(prompt: str, temperature: float = 0.2) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature},
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(f"{OLLAMA_BASE_URL}/api/generate", json=payload)
        response.raise_for_status()
    return response.json().get("response", "").strip()


def fallback_goals() -> list[dict[str, Any]]:
    titles = [
        "Define problem and motivation",
        "Summarize core idea",
        "Explain methodology",
        "Identify algorithm or architecture",
        "Understand experimental setup",
        "Report key results",
        "Discuss limitations",
        "Convert paper into implementation plan",
    ]
    return [
        {
            "id": f"goal_{index}",
            "title": title,
            "description": f"Study how the paper addresses: {title.lower()}.",
            "source_pages": [1],
            "status": "not_started",
        }
        for index, title in enumerate(titles, start=1)
    ]


def _extract_json_array(text: str) -> list[dict[str, Any]] | None:
    match = re.search(r"\[[\s\S]*\]", text)
    if not match:
        return None
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, list) else None


async def generate_study_goals(metadata: dict[str, Any], chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    context = "\n\n".join(
        f"[p. {chunk.get('page')}] {chunk.get('text', '')[:1800]}" for chunk in chunks[:5]
    )
    prompt = f"""
You are a research paper tutor. Generate exactly 8 study goals for the paper below.
Return only a JSON array. Each object must contain title, description, and source_pages.

Title: {metadata.get("title")}
Authors: {", ".join(metadata.get("authors", []))}
Abstract: {metadata.get("summary")}

Paper context:
{context}
""".strip()

    response = await generate(prompt)
    parsed = _extract_json_array(response) or []
    goals: list[dict[str, Any]] = []
    for index, item in enumerate(parsed[:8], start=1):
        if not isinstance(item, dict):
            continue
        source_pages = item.get("source_pages") or [1]
        goals.append(
            {
                "id": f"goal_{index}",
                "title": str(item.get("title") or fallback_goals()[index - 1]["title"]),
                "description": str(item.get("description") or ""),
                "source_pages": source_pages if isinstance(source_pages, list) else [source_pages],
                "status": "not_started",
            }
        )

    if len(goals) != 8:
        defaults = fallback_goals()
        goals.extend(defaults[len(goals) :])
    return goals[:8]
