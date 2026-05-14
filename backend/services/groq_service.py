from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import load_dotenv


BACKEND_DIR = Path(__file__).resolve().parents[1]
load_dotenv(BACKEND_DIR / ".env")

GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def groq_configured() -> bool:
    return bool(GROQ_API_KEY.strip())


async def groq_available() -> bool:
    if not groq_configured():
        return False
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(
                f"{GROQ_BASE_URL}/models",
                headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
            )
            return response.status_code == 200
    except httpx.HTTPError:
        return False


async def generate_with_groq(prompt: str, temperature: float = 0.2) -> str:
    if not groq_configured():
        raise RuntimeError("GROQ_API_KEY is not configured")

    payload: dict[str, Any] = {
        "model": GROQ_MODEL,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a rigorous research paper tutor. Be precise, grounded, "
                    "and evidence-driven. Do not invent claims that are not supported "
                    "by the provided paper context."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_completion_tokens": 4096,
    }
    async with httpx.AsyncClient(timeout=180.0) as client:
        response = await client.post(
            f"{GROQ_BASE_URL}/chat/completions",
            headers={
                "Authorization": f"Bearer {GROQ_API_KEY}",
                "Content-Type": "application/json",
            },
            json=payload,
        )
        response.raise_for_status()

    choices = response.json().get("choices", [])
    if not choices:
        return ""
    return choices[0].get("message", {}).get("content", "").strip()
