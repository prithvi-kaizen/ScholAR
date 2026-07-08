"""vision_service.py — orchestrates visual question-answering for ScholAR.

When the retriever returns a figure or table chunk as the most relevant hit,
``answer_with_figure`` is called instead of the normal text-generation path.
It:
  1. Loads the PNG file for the figure from disk.
  2. Base64-encodes it.
  3. Builds a prompt that includes the caption, relevant text context, and question.
  4. Sends everything to the local Ollama vision model (qwen3.5:9b, natively multimodal).
  5. Returns a structured VisionAnswer.

If Ollama is unavailable or the image is missing the function degrades gracefully:
it returns a caption-only answer with a ``fallback=True`` flag so the caller
can decide how to present it.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from backend.services.ollama_service import OLLAMA_MODEL, generate, ollama_available
from backend.services.pdf_service import paper_dir

logger = logging.getLogger(__name__)

# Maximum PNG file size to forward to the vision model
_MAX_IMAGE_BYTES = 18 * 1024 * 1024

# Number of supporting text chunks to include alongside the image
_MAX_TEXT_CONTEXT_CHUNKS = 3


def _load_figure_png(paper_id: str, image_file: str) -> bytes | None:
    """Load a figure PNG from the paper's figures/ directory.

    Returns None (not raises) if the file is missing or too large.
    """
    path = paper_dir(paper_id) / "figures" / image_file
    if not path.exists():
        logger.warning("Figure image not found: %s", path)
        return None
    data = path.read_bytes()
    if len(data) > _MAX_IMAGE_BYTES:
        logger.warning("Figure image too large (%d bytes), skipping vision call", len(data))
        return None
    return data


def _build_vision_prompt(
    question: str,
    figure_label: str,
    caption: str,
    text_context: str,
    paper_title: str,
) -> str:
    parts = [
        "You are a rigorous research paper tutor with vision capabilities. "
        "Answer questions by reading the provided figure or table image carefully. "
        "Be precise, extract exact numbers and labels, and ground every claim in "
        "what is visually shown. Do not invent data.",
        "",
        "Paper: " + paper_title if paper_title else "",
        "You are looking at " + figure_label + "." if figure_label else "",
        "Caption: " + caption if caption else "",
        "",
        "Supporting text context from the paper:",
        text_context or "(no additional text context available)",
        "",
        "Based on the image above and the supporting context, answer the following question.",
        "Cite specific values, labels, and visual elements you can read from the image.",
        "Wrap every piece of mathematical notation in single dollar signs, e.g. $x_t$ or $p_\\theta(x_{t-1}\\mid x_t)$, so it renders instead of showing raw symbols.",
        "Use the format:",
        "**Answer**",
        "<your answer>",
        "**Visual evidence**",
        "- <what you can read directly from the image>",
        "**Limits**",
        "- <what cannot be determined from this figure alone>",
        "",
        "Question: " + question,
    ]
    return "\n".join(p for p in parts if p is not None)


async def answer_with_figure(
    question: str,
    figure_chunk: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    paper_id: str,
    paper_metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a question grounded in a specific figure or table image.

    Args:
        question:       The user's question.
        figure_chunk:   The figure/table chunk dict from the retriever.
        context_chunks: Additional text chunks for supporting context.
        paper_id:       The local paper ID (used to resolve the image path).
        paper_metadata: Optional paper metadata for the prompt header.

    Returns a dict with keys:
        answer          str   -- model answer text (or caption-only fallback)
        figure_id       str   -- identifier of the figure used
        label           str   -- display label e.g. "Figure 1"
        caption         str   -- figure caption
        image_file      str   -- relative filename inside figures/
        page            int   -- page number
        model_used      str   -- vision model ID or "caption_fallback"
        fallback        bool  -- True when vision call was skipped / failed
        citations       list  -- single citation dict pointing to this figure
    """
    figure_id   = figure_chunk.get("figure_id", "")
    label       = figure_chunk.get("label", "")
    caption     = figure_chunk.get("caption", "")
    image_file  = figure_chunk.get("image_file", "")
    page        = figure_chunk.get("page", 1)
    paper_title = (paper_metadata or {}).get("title", "")

    # Build supporting text context (exclude the figure chunk itself)
    text_chunks = [
        c for c in context_chunks
        if not c.get("is_figure_chunk") and c.get("text", "").strip()
    ][:_MAX_TEXT_CONTEXT_CHUNKS]
    text_context = "\n\n".join(
        "[p. " + str(c.get("page")) + "] " + c.get("text", "")[:400]
        for c in text_chunks
    )

    # Build the single citation entry for this figure
    figure_citation = {
        "ref_id":        1,
        "page":          page,
        "chunk_id":      figure_chunk.get("chunk_id", "fig_" + figure_id),
        "section_title": label or ("Figure" if figure_chunk.get("figure_type") == "figure" else "Table"),
        "chunk_type":    figure_chunk.get("figure_type", "figure"),
        "quote":         caption[:520] if caption else label,
        "figure_id":     figure_id,
        "image_file":    image_file,
        "label":         label,
        "is_figure":     True,
    }

    def _caption_fallback(reason: str) -> dict[str, Any]:
        """Return a degraded text-only answer using the caption."""
        caption_note = ' The caption reads: "' + caption + '"' if caption else ""
        fallback_answer = (
            "**Answer**\n"
            "This question refers to " + label + "."
            + caption_note
            + "\n\n(Vision analysis unavailable: " + reason + ". "
            "Please open page " + str(page) + " of the PDF to view the figure directly.)"
        )
        return {
            "answer":     fallback_answer,
            "figure_id":  figure_id,
            "label":      label,
            "caption":    caption,
            "image_file": image_file,
            "page":       page,
            "model_used": "caption_fallback",
            "fallback":   True,
            "citations":  [figure_citation],
        }

    # -- Guard: vision requires a running local Ollama ------------------------
    if not await ollama_available():
        return _caption_fallback("Ollama is not running")

    # -- Load and encode the PNG ----------------------------------------------
    if not image_file:
        return _caption_fallback("figure image not extracted")

    png_bytes = _load_figure_png(paper_id, image_file)
    if png_bytes is None:
        return _caption_fallback("figure image file not found on disk")

    image_b64 = base64.b64encode(png_bytes).decode("ascii")

    # -- Call vision model ----------------------------------------------------
    prompt = _build_vision_prompt(question, label, caption, text_context, paper_title)
    try:
        answer_text = await generate(prompt, temperature=0.1, images=[image_b64], model=model)
    except Exception as exc:
        logger.warning("Vision model call failed: %s", exc)
        return _caption_fallback("vision model error: " + type(exc).__name__)

    if not answer_text.strip():
        return _caption_fallback("vision model returned empty response")

    return {
        "answer":     answer_text.strip(),
        "figure_id":  figure_id,
        "label":      label,
        "caption":    caption,
        "image_file": image_file,
        "page":       page,
        "model_used": model or OLLAMA_MODEL,
        "fallback":   False,
        "citations":  [figure_citation],
    }
