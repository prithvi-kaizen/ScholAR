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


# ---------------------------------------------------------------------------
# Vision (multimodal) generation
# ---------------------------------------------------------------------------

# Kept separate from GROQ_MODEL so that text-only answers never accidentally
# hit the vision model (and vice versa — vision model pricing differs).
GROQ_VISION_MODEL = os.getenv(
    "GROQ_VISION_MODEL",
    "meta-llama/llama-4-scout-17b-16e-instruct",
)


async def generate_with_groq_vision(
    prompt: str,
    image_b64: str,
    system_prompt: str = "",
    temperature: float = 0.1,
) -> str:
    """Send a text prompt + base64-encoded PNG image to the Groq vision model.

    The image is passed via the OpenAI ``image_url`` content-part format using a
    ``data:image/png;base64,...`` URI, which Groq supports for Llama 4 models.

    Args:
        prompt: The user question / instruction.
        image_b64: Base64-encoded PNG bytes (no data-URI prefix).
        system_prompt: Optional system message override.
        temperature: Sampling temperature (low = more deterministic).

    Returns:
        The model's text response.

    Raises:
        RuntimeError: When GROQ_API_KEY is not configured.
        httpx.HTTPStatusError: On non-2xx API response.
    """
    if not groq_configured():
        raise RuntimeError("GROQ_API_KEY is not configured")

    sys_text = system_prompt or (
        "You are a rigorous research paper tutor with vision capabilities. "
        "Answer questions by reading the provided figure or table image carefully. "
        "Be precise, extract exact numbers and labels, and ground every claim in "
        "what is visually shown. Do not invent data."
    )

    payload: dict[str, Any] = {
        "model": GROQ_VISION_MODEL,
        "messages": [
            {"role": "system", "content": sys_text},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{image_b64}",
                        },
                    },
                    {"type": "text", "text": prompt},
                ],
            },
        ],
        "temperature": temperature,
        "max_completion_tokens": 1024,
    }

    async with httpx.AsyncClient(timeout=120.0) as client:
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

