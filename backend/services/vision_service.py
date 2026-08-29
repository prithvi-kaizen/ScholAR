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


def _build_multi_vision_prompt(
    question: str,
    figures: list[dict[str, Any]],
    text_context: str,
    paper_title: str,
) -> str:
    parts = [
        "You are a rigorous research paper tutor with vision capabilities.",
        "Answer questions by reading the provided figure or table images carefully.",
        "When multiple figures/images are provided, perform comparative analysis across all of them.",
        "Be precise, extract exact numbers and labels, and ground every claim in what is visually shown.",
        "",
        f"Paper: {paper_title}" if paper_title else "",
        "",
        "Provided visual evidence:",
    ]
    for i, fig in enumerate(figures, start=1):
        label = fig.get("label", f"Image {i}")
        caption = fig.get("caption", "")
        parts.append(f"[Image {i}: {label}]")
        if caption:
            parts.append(f"Caption: {caption}")
        parts.append("")

    is_table_query = any(
        f.get("figure_type") == "table" or "table" in f.get("label", "").lower()
        for f in figures
    ) or "table" in question.lower()

    if is_table_query:
        parts.extend([
            "Supporting text context from the paper:",
            text_context or "(no additional text context available)",
            "",
            "CRITICAL INSTRUCTIONS FOR TABULAR EVIDENCE:",
            "1. Read the numbers and text from the table image with extreme precision.",
            "2. Reconstruct the table as a clean, complete Markdown table with all columns and rows.",
            "3. Format model names, configurations, and abbreviations naturally as clean readable text (e.g. BERT-Base, BERT-Large, Transformer-Big) - NEVER write pseudo-LaTeX like $\\text{BERT}_{\\text{BASE}}$.",
            "4. Cite the table as [1] in your **Answer** section (e.g. Table 1 [1] summarizes...).",
            "5. Provide a row-by-row / model-by-model analysis detailing metric differences, gains, and tradeoffs.",
            "6. Use single dollar signs ONLY for actual mathematical equations or scientific notations (e.g. $1.0 \\times 10^{20}$, $\\alpha = 0.1$).",
            "",
            "Use the exact format:",
            "**Answer**",
            "<high-level overview of what the table evaluates and main conclusions [1]>",
            "",
            "**Structured Table**",
            "| Model / Configuration | Metric 1 | Metric 2 | Training Cost / Notes |",
            "| :--- | :--- | :--- | :--- |",
            "<fill in all rows faithfully from the image>",
            "",
            "**Row-by-Row Analysis & Metric Deltas**",
            "- **<Model 1>**: <exact metrics from table and comparison to others>",
            "- **<Model 2>**: <exact metrics from table and comparison to others>",
            "",
            "**Key Insights & Tradeoffs**",
            "- <efficiency, FLOPs, parameter count, or performance breakthroughs>",
            "",
            "**Limits**",
            "- <what cannot be determined from this table alone>",
            "",
            f"Question: {question}",
        ])
    else:
        parts.extend([
            "Supporting text context from the paper:",
            text_context or "(no additional text context available)",
            "",
            "CRITICAL INSTRUCTIONS FOR MULTIMODAL SYNTHESIS:",
            "1. Synthesize findings from BOTH the visual figures and the supporting text passages from the paper.",
            "2. If specific quantitative thresholds, numbers, or empirical findings are stated in the text context (such as from Results, Observations, or Discussion sections), state them clearly and combine them with the visual trends in the figures.",
            "3. Cite the evidence in your **Answer** section (e.g. Figure 1 [1], Section 4 [2]).",
            "4. Format model names and words naturally (e.g. BERT-Base, ResNet-50) rather than LaTeX commands. Use dollar signs ONLY for true mathematical formulas (e.g. $2^{14}$, $p_\\theta(x_t)$).",
            "5. If the figures are architectural diagrams and do not contain performance numbers, explain the architecture from the figures and extract the performance comparison / thresholds directly from the supporting text context.",
            "",
            "Use the format:",
            "**Answer**",
            "<your comprehensive synthesized answer/comparison citing [1] and text evidence>",
            "",
            "**Key Findings & Evidence**",
            "- **Visual Evidence**: <what is depicted in the figures>",
            "- **Quantitative Findings / Thresholds**: <empirical metrics or thresholds from the paper>",
            "",
            "**Limits**",
            "- <what cannot be determined from these figures/sections alone>",
            "",
            f"Question: {question}",
        ])
    return "\n".join(p for p in parts if p is not None)


def is_uninformative_visual_answer(answer: str) -> bool:
    """Detect if the vision model reported an uninformative refusal due to diagram-only images."""
    if not answer or len(answer.strip()) < 20:
        return True
    lowered = answer.lower()
    refusal_signals = [
        "not possible to determine",
        "not possible to infer",
        "cannot determine",
        "cannot be determined from",
        "do not provide comparative",
        "does not provide comparative",
        "no performance data",
        "no performance metrics",
        "actual results or charts comparing",
        "do not contain a chart or table",
        "does not contain a chart or table",
        "contains no performance data",
        "contain no performance data",
        "not included in the provided images",
        "not present in the image",
        "not present in the provided image",
    ]
    return any(sig in lowered for sig in refusal_signals)


async def answer_with_multimodal_evidence(
    question: str,
    figure_chunks: list[dict[str, Any]],
    context_chunks: list[dict[str, Any]],
    paper_id: str,
    paper_metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a question grounded in one or multiple figure/table images."""
    if not figure_chunks:
        raise ValueError("figure_chunks must not be empty")

    paper_title = (paper_metadata or {}).get("title", "")
    primary_fig = figure_chunks[0]
    figure_id = primary_fig.get("figure_id", "")
    label = primary_fig.get("label", "")
    caption = primary_fig.get("caption", "")
    image_file = primary_fig.get("image_file", "")
    page = primary_fig.get("page", 1)

    # Build supporting text context
    text_chunks = [
        c for c in context_chunks
        if not c.get("is_figure_chunk") and c.get("text", "").strip()
    ][:_MAX_TEXT_CONTEXT_CHUNKS]
    text_context = "\n\n".join(
        f"[Section: {c.get('section_title') or c.get('section') or 'Text'}, p. {c.get('page')}]\n" + c.get("text", "")[:2500]
        for c in text_chunks
    )

    def _caption_fallback(reason: str) -> dict[str, Any]:
        """Return a degraded text-only answer using captions."""
        labels_joined = ", ".join(f.get("label", "Figure") for f in figure_chunks)
        captions_joined = "\n\n".join(f"[{f.get('label')}]: {f.get('caption', '')}" for f in figure_chunks if f.get("caption"))
        fallback_answer = (
            f"**Answer**\nThis question refers to {labels_joined}.\n\n"
            f"{captions_joined}\n\n"
            f"(Vision analysis unavailable: {reason}. Please open page {page} of the PDF to view the figures directly.)"
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
            "citations":  [
                {
                    "ref_id": i + 1,
                    "page": f.get("page", 1),
                    "chunk_id": f.get("chunk_id", f"fig_{f.get('figure_id')}"),
                    "section_title": f.get("label", "Figure"),
                    "chunk_type": f.get("figure_type", "figure"),
                    "quote": (f.get("caption") or f.get("label") or "")[:520],
                    "figure_id": f.get("figure_id"),
                    "image_file": f.get("image_file"),
                    "label": f.get("label"),
                    "is_figure": True,
                }
                for i, f in enumerate(figure_chunks)
            ],
        }

    # -- Guard: vision requires a running local Ollama ------------------------
    if not await ollama_available():
        return _caption_fallback("Ollama is not running")

    # -- Load and base64-encode all available figure PNGs ---------------------
    images_b64: list[str] = []
    loaded_figures: list[dict[str, Any]] = []

    for fig in figure_chunks:
        img_f = fig.get("image_file", "")
        if not img_f:
            continue
        png_bytes = _load_figure_png(paper_id, img_f)
        if png_bytes is not None:
            images_b64.append(base64.b64encode(png_bytes).decode("ascii"))
            loaded_figures.append(fig)

    if not images_b64:
        return _caption_fallback("figure image files not found on disk")

    # -- Call vision model with all loaded images ------------------------------
    prompt = _build_multi_vision_prompt(question, loaded_figures, text_context, paper_title)
    try:
        answer_text = await generate(prompt, temperature=0.1, images=images_b64, model=model)
    except Exception as exc:
        logger.warning("Vision model call failed: %s", exc)
        return _caption_fallback("vision model error: " + type(exc).__name__)

    if not answer_text.strip():
        return _caption_fallback("vision model returned empty response")

    # -- Extract and resolve subregion proposals for each figure ---------------
    from backend.schemas.document import BoundingBox
    from backend.services.grounding_service import VisualGroundingService

    all_citations: list[dict[str, Any]] = []
    for idx, fig in enumerate(loaded_figures, start=1):
        f_id = fig.get("figure_id", str(idx))
        raw_bbox = fig.get("bbox_normalized") or fig.get("bbox") or {}
        x0_val = float(raw_bbox.get("x0", 0.0))
        y0_val = float(raw_bbox.get("y0", 0.0))
        x1_val = float(raw_bbox.get("x1", 1.0))
        y1_val = float(raw_bbox.get("y1", 1.0))
        if max(x0_val, y0_val, x1_val, y1_val) > 1.0:
            # Fallback normalization for unscaled PDF points
            x0_val = min(1.0, max(0.0, x0_val / 612.0))
            y0_val = min(1.0, max(0.0, y0_val / 792.0))
            x1_val = min(1.0, max(0.0, x1_val / 612.0))
            y1_val = min(1.0, max(0.0, y1_val / 792.0))

        parent_box = BoundingBox(
            x0=round(x0_val, 4),
            y0=round(y0_val, 4),
            x1=round(x1_val, 4),
            y1=round(y1_val, 4),
            coordinate_space="normalized_page",
        )
        proposal = VisualGroundingService.extract_subregion_proposals(
            answer_text,
            evidence_id=f_id,
            model_id=model or OLLAMA_MODEL,
        )
        resolved_subregions = VisualGroundingService.resolve_regions_from_proposal(
            proposal=proposal,
            parent_page_box=parent_box,
            document_id=paper_id,
            page_number=fig.get("page", 1),
        )
        all_citations.append({
            "ref_id": idx,
            "page": fig.get("page", 1),
            "chunk_id": fig.get("chunk_id", f"fig_{f_id}"),
            "section_title": fig.get("label") or ("Figure" if fig.get("figure_type") == "figure" else "Table"),
            "chunk_type": fig.get("figure_type", "figure"),
            "quote": (fig.get("caption") or fig.get("label") or "")[:520],
            "figure_id": f_id,
            "image_file": fig.get("image_file"),
            "label": fig.get("label"),
            "is_figure": True,
            "bbox_normalized": parent_box.model_dump(),
            "subregions": [
                {
                    "region_id": r.region_id,
                    "role": r.role,
                    "bbox": r.bbox_page_normalized.model_dump(),
                    "verification": r.verification,
                }
                for r in resolved_subregions
            ],
        })

    # Append top supporting text chunks to citations pool
    start_ref = len(all_citations) + 1
    for t_idx, tc in enumerate(context_chunks[:4], start=start_ref):
        all_citations.append({
            "ref_id": t_idx,
            "page": tc.get("page", 1),
            "chunk_id": tc.get("chunk_id", f"text_{t_idx}"),
            "section_title": tc.get("section_title") or tc.get("section") or "Text Context",
            "chunk_type": tc.get("chunk_type", "text"),
            "quote": (tc.get("text") or tc.get("content") or "")[:400],
            "is_figure": False,
        })

    import re
    cleaned_answer = re.sub(r"```json[\s\S]*?```", "", answer_text)
    cleaned_answer = re.sub(r"\*\*Subregions\*\*[\s\S]*", "", cleaned_answer).strip()
    # Normalize pseudo-LaTeX model names like $\text{BERT}_{\text{BASE}}$ -> BERT-BASE
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{([^}]+)\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\\text\{([^}]+)\}", r"\1", cleaned_answer)

    labels_all = ", ".join(f.get("label", "Figure") for f in loaded_figures)
    return {
        "answer": cleaned_answer or answer_text.strip(),
        "figure_id": figure_id,
        "label": labels_all,
        "caption": caption,
        "image_file": image_file,
        "page": page,
        "model_used": model or OLLAMA_MODEL,
        "fallback": False,
        "citations": all_citations,
        "loaded_figures": loaded_figures,
    }


async def answer_with_figure(
    question: str,
    figure_chunk: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    paper_id: str,
    paper_metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Single-figure wrapper around answer_with_multimodal_evidence."""
    return await answer_with_multimodal_evidence(
        question=question,
        figure_chunks=[figure_chunk],
        context_chunks=context_chunks,
        paper_id=paper_id,
        paper_metadata=paper_metadata,
        model=model,
    )


async def answer_with_custom_snippet(
    question: str,
    snippet_id: str,
    page_number: int,
    bbox_norm: list[float],
    snippet_text: str,
    paper_id: str,
    paper_metadata: dict[str, Any] | None = None,
    model: str | None = None,
) -> dict[str, Any]:
    """Answer a user question about a manually cropped visual snippet from a specific page."""
    model_to_use = model or OLLAMA_MODEL
    snippet_path = paper_dir(paper_id) / "snippets" / f"{snippet_id}.png"

    if not snippet_path.exists():
        return {
            "answer": f"The visual snippet on page {page_number} could not be loaded.",
            "citations": [],
            "fallback": True,
        }

    image_b64 = _image_to_base64(snippet_path)
    if not image_b64:
        return {
            "answer": f"Failed to encode snippet on page {page_number}.",
            "citations": [],
            "fallback": True,
        }

    paper_title = (paper_metadata or {}).get("title", "Research Paper")

    prompt = f"""You are ScholAR, a research paper tutor analyzing a specific user-cropped region (snippet) from the paper:
Title: {paper_title}
Source: Page {page_number}

Extracted text from this cropped region:
{snippet_text or "(No raw text recognized - graphical or mathematical notation)"}

CRITICAL INSTRUCTIONS:
1. Examine the cropped image with extreme precision (equations, formulas, curves, pseudocode, or diagram labels).
2. Answer the user's specific question directly based on this cropped visual snippet.
3. Cite the snippet as [1] in your **Answer** section (e.g. In this snippet [1]...).
4. Wrap every mathematical variable or formula in single dollar signs, e.g. $x_t$, $\\alpha = 0.1$.
5. Format model and layer names naturally as clean readable text (e.g. BERT-Base, ResNet-50) rather than LaTeX commands.

Use the format:
**Answer**
<detailed, precise explanation of what is in this snippet and answer to the question citing [1]>

**Key Elements**
- <element or notation 1>: <explanation>
- <element or notation 2>: <explanation>

**Limits**
- <what cannot be determined from this snippet alone>

Question: {question}"""

    try:
        response_text = await generate_vision(
            prompt=prompt,
            images_b64=[image_b64],
            model=model_to_use,
            temperature=0.1,
        )
    except Exception as exc:
        logger.warning("Custom snippet vision inference failed: %s", exc)
        return {
            "answer": f"Vision analysis failed for this snippet: {exc}",
            "citations": [],
            "fallback": True,
        }

    import re
    cleaned_answer = re.sub(r"```json[\s\S]*?```", "", response_text)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{([^}]+)\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\\text\{([^}]+)\}", r"\1", cleaned_answer)

    citations = [
        {
            "ref_id": 1,
            "page": page_number,
            "chunk_id": f"snippet_{snippet_id}",
            "section_title": f"Snippet (Page {page_number})",
            "chunk_type": "snippet",
            "quote": (snippet_text or f"Custom cropped snippet from page {page_number}")[:520],
            "figure_id": snippet_id,
            "label": f"Snippet (Page {page_number})",
            "is_figure": True,
            "is_snippet": True,
            "image_url": f"/api/papers/{paper_id}/snippets/{snippet_id}.png",
            "bbox_normalized": {
                "x0": bbox_norm[0],
                "y0": bbox_norm[1],
                "x1": bbox_norm[2],
                "y1": bbox_norm[3],
            },
        }
    ]

    return {
        "answer": cleaned_answer.strip() or response_text.strip(),
        "snippet_id": snippet_id,
        "label": f"Snippet (Page {page_number})",
        "page": page_number,
        "model_used": model_to_use,
        "fallback": False,
        "citations": citations,
        "bbox_normalized": bbox_norm,
        "image_url": f"/api/papers/{paper_id}/snippets/{snippet_id}.png",
    }
