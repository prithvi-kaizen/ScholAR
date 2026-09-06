"""Source-scoped local visual question answering for crops and full pages.

The production path accepts visual evidence selected by retrieval, loads only safe
paper-relative PNGs, asks the configured Ollama vision model to transcribe relevant
pixels, then synthesizes an answer from that observation and nearby text. Failures are
reported as labeled fallbacks so the answer pipeline can abstain or continue according
to its execution policy.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import math
import re
from pathlib import Path, PurePosixPath
from typing import Any

from backend.services.ollama_service import OLLAMA_MODEL, generate_result, ollama_available
from backend.services.pdf_service import paper_dir

logger = logging.getLogger(__name__)

# Maximum PNG file size to forward to the vision model
_MAX_IMAGE_BYTES = 18 * 1024 * 1024

# Number of supporting text chunks to include alongside the image
_MAX_TEXT_CONTEXT_CHUNKS = 3


def _retrieval_region_bbox(figure: dict[str, Any]) -> list[float] | None:
    """Return the highest-ranked valid retrieval crop in normalized coordinates."""
    regions = figure.get("candidate_regions")
    if not isinstance(regions, list):
        return None
    for region in regions:
        if not isinstance(region, dict):
            continue
        raw = region.get("bbox_norm")
        if not isinstance(raw, (list, tuple)) or len(raw) != 4:
            continue
        try:
            x0, y0, x1, y1 = (float(value) for value in raw)
        except (TypeError, ValueError):
            continue
        if (
            0.0 <= x0 < x1 <= 1.0
            and 0.0 <= y0 < y1 <= 1.0
            and (x1 - x0) * (y1 - y0) >= 0.0025
        ):
            return [x0, y0, x1, y1]
    return None


def _crop_visual_png(png_bytes: bytes, bbox_norm: list[float]) -> bytes | None:
    """Crop a retrieval-proposed region without creating a derived disk artifact."""
    try:
        from PIL import Image

        with Image.open(io.BytesIO(png_bytes)) as image:
            image.load()
            width, height = image.size
            x0, y0, x1, y1 = bbox_norm
            box = (
                max(0, min(int(width * x0), width - 1)),
                max(0, min(int(height * y0), height - 1)),
                max(1, min(int(math.ceil(width * x1)), width)),
                max(1, min(int(math.ceil(height * y1)), height)),
            )
            if box[2] <= box[0] or box[3] <= box[1]:
                return None
            cropped = image.convert("RGB").crop(box)
            output = io.BytesIO()
            cropped.save(output, format="PNG", optimize=True)
            value = output.getvalue()
            return value if len(value) <= _MAX_IMAGE_BYTES else None
    except Exception as exc:
        logger.warning("Could not crop retrieval-proposed visual region: %s", exc)
        return None


def _load_visual_png(
    source_paper_id: str,
    image_file: str,
    image_relpath: str | None = None,
) -> bytes | None:
    """Safely load a bounded paper-relative page, figure, or table image."""
    root = paper_dir(source_paper_id).resolve()
    relative = str(image_relpath or "").strip()
    if relative:
        parsed = PurePosixPath(relative)
        if parsed.is_absolute() or ".." in parsed.parts or not parsed.parts:
            logger.warning("Rejected unsafe visual image path: %r", image_relpath)
            return None
        path = root.joinpath(*parsed.parts).resolve()
        try:
            path.relative_to(root)
        except ValueError:
            logger.warning("Rejected visual image outside paper directory: %s", path)
            return None
    else:
        image_name = str(image_file or "").strip()
        if not image_name or Path(image_name).name != image_name:
            logger.warning("Rejected unsafe figure image filename: %r", image_file)
            return None
        path = (root / "figures" / image_name).resolve()

    try:
        if not path.is_file():
            logger.warning("Visual image not found: %s", path)
            return None
        if path.stat().st_size > _MAX_IMAGE_BYTES:
            logger.warning("Visual image too large (%d bytes), skipping vision call", path.stat().st_size)
            return None
        return path.read_bytes()
    except OSError as exc:
        logger.warning("Could not read visual image [%s]: %s", path, exc)
        return None


def _load_figure_png(source_paper_id: str, image_file: str) -> bytes | None:
    """Backward-compatible loader for legacy figure-only callers and tests."""
    return _load_visual_png(source_paper_id, image_file)


def _image_to_base64(path: Path) -> str | None:
    """Encode a bounded existing image for the user-snippet vision path."""
    try:
        if not path.is_file() or path.stat().st_size > _MAX_IMAGE_BYTES:
            return None
        return base64.b64encode(path.read_bytes()).decode("ascii")
    except OSError as exc:
        logger.warning("Could not encode visual snippet [%s]: %s", path, exc)
        return None


def _figure_source_id(figure: dict[str, Any], default_paper_id: str) -> str:
    return str(
        figure.get("source_paper_id")
        or figure.get("document_id")
        or default_paper_id
    )


def _visual_image_url(figure: dict[str, Any], default_paper_id: str) -> str | None:
    """Return the API URL for the exact source-scoped visual shown to the model."""
    source_id = _figure_source_id(figure, default_paper_id)
    page = figure.get("page")
    if figure.get("is_page_visual_chunk") or figure.get("figure_type") == "page":
        if isinstance(page, int) and page >= 1:
            return f"/api/papers/{source_id}/page/{page}.png?zoom=1.6"
        return None
    figure_id = str(figure.get("figure_id") or "").strip()
    if figure_id:
        return f"/api/papers/{source_id}/figures/{figure_id}.png"
    return None


def _source_title(
    source_paper_id: str,
    source_metadata: dict[str, dict[str, Any]] | None,
) -> str:
    metadata = (source_metadata or {}).get(source_paper_id, {})
    return str(metadata.get("title") or source_paper_id or "(title unavailable)")


def _build_visual_observation_prompt(
    question: str,
    figures: list[dict[str, Any]],
    source_metadata: dict[str, dict[str, Any]] | None = None,
) -> str:
    evidence = "\n".join(
        (
            f"- {figure.get('_vision_evidence_id', f'V{index}')} "
            f"(citation [{figure.get('_vision_ref_id', index)}], "
            f"source_id={_figure_source_id(figure, '')!r}, "
            f"source_title={figure.get('_source_title') or _source_title(_figure_source_id(figure, ''), source_metadata)!r}): "
            f"{figure.get('label', 'Figure')} — {figure.get('caption', '')}; "
            f"vision_input={figure.get('_vision_input_kind', 'full_visual')}; "
            f"page_bbox={figure.get('_retrieval_bbox_normalized') or 'full'}"
        )
        for index, figure in enumerate(figures, start=1)
    )
    allowed_ids = [
        str(figure.get("_vision_evidence_id") or f"V{index}")
        for index, figure in enumerate(figures, start=1)
    ]
    return f"""You are extracting evidence from scientific figure and table pixels.
Each image has a globally unambiguous visual evidence ID and source label:
{evidence}

Question that will later be answered: {question}

Transcribe only visually observable evidence relevant to the question. Record exact labels, axes, legends, values, trends, diagram relationships, and uncertainty. Do not answer the question, infer causes, merge evidence across images, or add outside knowledge. If an element is unreadable, say so explicitly.

Return only this JSON shape, with exactly one object per supplied ID {allowed_ids!r}:
{{"observations":[{{"evidence_id":"V1","observation":"pixel-grounded observation for V1 only"}}]}}"""


def _parse_visual_observations(
    raw_response: str,
    allowed_ids: set[str],
) -> dict[str, str]:
    """Validate first-pass output and keep only source-specific observations."""
    text = raw_response.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        payload = json.loads(text[start : end + 1])
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    items = payload.get("observations") if isinstance(payload, dict) else None
    if not isinstance(items, list):
        return {}

    observations: dict[str, str] = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        evidence_id = str(item.get("evidence_id") or "").strip()
        observation = str(item.get("observation") or "").strip()
        if evidence_id not in allowed_ids or not observation or evidence_id in observations:
            continue
        observations[evidence_id] = observation[:4000]
    return observations


def _build_multi_vision_prompt(
    question: str,
    figures: list[dict[str, Any]],
    text_context: str,
    paper_title: str,
    visual_observations: dict[str, str] | None = None,
    source_metadata: dict[str, dict[str, Any]] | None = None,
) -> str:
    parts = [
        "You are a rigorous research paper tutor with vision capabilities.",
        "Answer questions by reading the provided figure or table images carefully.",
        "When multiple figures/images are provided, compare them without merging their provenance.",
        "Be precise, extract exact numbers and labels, and ground every claim in the numbered evidence reference shown below.",
        "",
        f"Anchor paper: {paper_title}" if paper_title else "",
        "",
        "Provided visual evidence:",
    ]
    for i, fig in enumerate(figures, start=1):
        label = fig.get("label", f"Image {i}")
        caption = fig.get("caption", "")
        evidence_id = str(fig.get("_vision_evidence_id") or f"V{i}")
        ref_id = int(fig.get("_vision_ref_id") or i)
        source_id = _figure_source_id(fig, "")
        source_title = str(
            fig.get("_source_title") or _source_title(source_id, source_metadata)
        )
        parts.append(
            f"[{evidence_id} -> citation [{ref_id}] | source_id={source_id} | "
            f"source_title={source_title} | {label} | "
            f"vision_input={fig.get('_vision_input_kind', 'full_visual')} | "
            f"page_bbox={fig.get('_retrieval_bbox_normalized') or 'full'}]"
        )
        if caption:
            parts.append(f"Caption: {caption}")
        structured_text = str(fig.get("text") or fig.get("body_text") or "").strip()
        if structured_text and structured_text != caption:
            parts.append(f"Extracted content/grid: {structured_text[:1200]}")
        observation = (visual_observations or {}).get(evidence_id)
        if observation:
            parts.append(
                "First-pass model-generated transcription for this image only "
                f"(verify against pixels): {observation}"
            )
        parts.append("")

    parts.extend([
        "Evidence citation policy:",
        "- Cite visual and text claims only with the exact numeric citation IDs assigned below.",
        "- Never attribute one image's transcription, labels, or values to another image or source.",
        "- A first-pass transcription is model-generated proxy evidence, not independent validation.",
        "",
    ])

    has_table = any(
        f.get("figure_type") == "table" or "table" in f.get("label", "").lower()
        for f in figures
    ) or "table" in question.lower()
    has_diagram_or_multiple = len(figures) > 1 or any(
        f.get("figure_type") != "table" and "table" not in f.get("label", "").lower()
        for f in figures
    ) or "figure" in question.lower() or "diagram" in question.lower() or "relate" in question.lower() or "mechanism" in question.lower()

    q_lower = question.lower()
    wants_full_table = any(kw in q_lower for kw in [
        "full table", "all rows", "reconstruct the table", "reconstruct table",
        "entire table", "show the table", "structured table", "list all models",
        "list all rows", "tabular summary of all", "row-by-row", "every row"
    ])

    if has_table and wants_full_table:
        parts.extend([
            "Supporting text context from the paper:",
            text_context or "(no additional text context available)",
            "",
            "CRITICAL INSTRUCTIONS FOR FULL TABULAR RECONSTRUCTION:",
            "1. Read the numbers and text from the table image with extreme precision.",
            "2. Reconstruct the table as a clean, complete Markdown table with all columns and rows faithfully from the visual evidence.",
            "3. Provide a concise row-by-row / model-by-model analysis detailing metric differences, gains, and tradeoffs.",
            "4. Cite every table claim with that table's exact assigned numeric citation marker (e.g. [1]).",
            "5. Format model names and abbreviations naturally as clean readable text (e.g. BERT-Base, Transformer-Big).",
            "",
            "Use the format:",
            "**Answer**",
            "<high-level overview of what the table evaluates and main conclusions with exact citations>",
            "",
            "**Structured Table**",
            "| Model / Configuration | Metric 1 | Metric 2 | Training Cost / Notes |",
            "| :--- | :--- | :--- | :--- |",
            "<fill in all rows faithfully from the image>",
            "",
            "**Row-by-Row Analysis & Metric Deltas**",
            "- **<Model 1>**: <exact metrics from table and comparison with citation>",
            "- **<Model 2>**: <exact metrics from table and comparison with citation>",
            "",
            f"Question: {question}",
        ])
    elif has_table and has_diagram_or_multiple:
        parts.extend([
            "Supporting text context from the paper:",
            text_context or "(no additional text context available)",
            "",
            "CRITICAL INSTRUCTIONS FOR TARGETED MULTIMODAL SYNTHESIS:",
            "1. Answer the user's specific question directly, synthesizing evidence from BOTH the diagram/figures and the table/text context.",
            "2. RELEVANCE RULE: Focus exclusively on the specific models, parameters, equations, or rows relevant to the question. Do NOT dump entire unrelated table sections (e.g. do not discuss attention head variations when the question is about positional encodings).",
            "3. If comparing metrics, cite them directly in your explanation with the assigned numeric citation markers (e.g. [1], [2]). You may include a small, focused comparison of only the relevant models if helpful.",
            "4. Explain how the architectural mechanism shown in the diagram figure connects to or explains the empirical findings.",
            "5. Do NOT include empty headers or boilerplate sections. If there are no limits to report, omit the Limits section entirely.",
            "",
            "Use the format:",
            "**Answer**",
            "<direct, clear synthesized answer addressing the user's specific question in 1-2 focused paragraphs with exact citations>",
            "",
            "**Key Findings & Mechanism**",
            "- **Architectural Link**: <how the structural mechanism in the diagram explains or connects to the finding with citations>",
            "- **Empirical Evidence**: <specific relevant metrics or tradeoffs from the paper/table with citations>",
            "",
            f"Question: {question}",
        ])
    elif has_table:
        parts.extend([
            "Supporting text context from the paper:",
            text_context or "(no additional text context available)",
            "",
            "CRITICAL INSTRUCTIONS FOR TABULAR EVIDENCE:",
            "1. Answer the user's specific question directly using the numbers and text from the table image and text context.",
            "2. RELEVANCE RULE: Focus exclusively on the specific models, metrics, or rows relevant to the user's question. Do NOT dump entire unrelated tables or rows.",
            "3. Cite every claim with that table's exact assigned numeric citation ID (e.g. [1] or [2]).",
            "4. Format model names and numbers naturally (e.g. BERT-Base, Transformer-Big).",
            "5. Do NOT include empty headers or boilerplate sections. If there are no limits, omit the Limits section entirely.",
            "",
            "Use the format:",
            "**Answer**",
            "<direct, comprehensive answer addressing the user's question with exact citations and relevant metrics>",
            "",
            "**Key Insights**",
            "- <1-2 concise bullet points highlighting key empirical takeaways or tradeoffs relevant to the question>",
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
            "2. If specific quantitative thresholds, numbers, or empirical findings are stated in the text context, state them clearly and combine them with the visual trends in the figures.",
            "3. Cite every single factual statement or bullet point with its supporting numeric citation marker (e.g. [1], [2]). Put the citation marker directly inside the bullet point.",
            "4. Format model names and words naturally (e.g. BERT-Base, ResNet-50) rather than LaTeX commands. Use dollar signs ONLY for true mathematical formulas (e.g. $2^{14}$, $p_\\theta(x_t)$).",
            "5. If the figures are architectural diagrams and do not contain performance numbers, explain the architecture from the figures and extract the performance comparison / thresholds directly from the supporting text context.",
            "6. Do NOT include empty headers. If there are no limitations, omit the Limits section entirely.",
            "",
            "Use the format:",
            "**Answer**",
            "<your comprehensive synthesized answer/comparison citing [1] and text evidence>",
            "",
            "**Key Findings & Evidence**",
            "- **Visual Evidence**: <what is depicted in the figures with citations like [1]>",
            "- **Quantitative Findings / Thresholds**: <empirical metrics or thresholds from the paper with citations like [2]>",
            "",
            f"Question: {question}",
        ])
    return "\n".join(p for p in parts if p is not None)


def is_uninformative_visual_answer(answer: str) -> bool:
    """Detect if the vision model reported an uninformative refusal due to diagram-only images."""
    if not answer or len(answer.strip()) < 20:
        return True
    if "| :---" in answer or "**Structured Table**" in answer or "**Row-by-Row Analysis" in answer:
        return False
    body = re.split(r"\*\*(?:Limits|Limitations)\*\*", answer, flags=re.IGNORECASE)[0]
    lowered = body.lower()
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
    source_metadata: dict[str, dict[str, Any]] | None = None,
    model: str | None = None,
    seed: int | None = None,
    decoding_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answer a question grounded in one or multiple figure/table images."""
    if not figure_chunks:
        raise ValueError("figure_chunks must not be empty")

    paper_title = (paper_metadata or {}).get("title", "")
    source_registry = dict(source_metadata or {})
    source_registry.setdefault(paper_id, paper_metadata or {})
    primary_fig = figure_chunks[0]
    primary_source_id = _figure_source_id(primary_fig, paper_id)
    figure_id = primary_fig.get("figure_id", "")
    label = primary_fig.get("label", "")
    caption = primary_fig.get("caption", "")
    image_file = primary_fig.get("image_file", "")
    page = primary_fig.get("page", 1)

    # Keep the exact supporting chunks; citation IDs are assigned after the
    # successfully loaded images so prompt references and returned citations align.
    text_chunks = [
        c for c in context_chunks
        if not c.get("is_figure_chunk") and c.get("text", "").strip()
    ][:_MAX_TEXT_CONTEXT_CHUNKS]

    def _caption_fallback(reason: str) -> dict[str, Any]:
        """Return a structured multi-level reasoning answer using text context and figure evidence."""
        if text_chunks:
            from backend.services.mlr_synthesis_service import MLRSynthesisService
            mlr_res = MLRSynthesisService.synthesize_extractive_mlr(question, text_chunks, figure_chunks)
            mlr_citations: list[dict[str, Any]] = []
            for c in mlr_res["citations"]:
                cid = str(c.get("chunk_id", ""))
                is_fig = "fig_" in cid or any(f.get("figure_id") and f.get("figure_id") in cid for f in figure_chunks)
                fig_match = next((f for f in figure_chunks if f.get("figure_id") and f.get("figure_id") in cid), primary_fig)
                mlr_citations.append({
                    "ref_id": c.get("ref_id"),
                    "page": c.get("page", 1),
                    "chunk_id": cid or f"fig_{fig_match.get('figure_id')}",
                    "section_title": c.get("section", "Body"),
                    "chunk_type": "figure" if is_fig else "text",
                    "quote": (c.get("quote") or "")[:520],
                    "figure_id": fig_match.get("figure_id"),
                    "image_file": fig_match.get("image_file"),
                    "image_relpath": fig_match.get("image_relpath"),
                    "image_url": _visual_image_url(fig_match, paper_id),
                    "label": c.get("section") if ("Figure" in str(c.get("section", "")) or "Table" in str(c.get("section", ""))) else fig_match.get("label"),
                    "is_figure": is_fig,
                    "is_page_visual": bool(fig_match.get("is_page_visual_chunk")),
                    "source_paper_id": _figure_source_id(fig_match, paper_id),
                    "document_id": _figure_source_id(fig_match, paper_id),
                })
            return {
                "answer":     mlr_res["answer"],
                "figure_id":  figure_id,
                "label":      label,
                "caption":    caption,
                "image_file": image_file,
                "image_relpath": primary_fig.get("image_relpath"),
                "image_url":  _visual_image_url(primary_fig, paper_id),
                "page":       page,
                "source_paper_id": primary_source_id,
                "model_used": "mlr_multimodal_synthesis",
                "fallback":   True,
                "citations":  mlr_citations or [
                    {
                        "ref_id": i + 1,
                        "page": f.get("page", 1),
                        "chunk_id": f.get("chunk_id", f"fig_{f.get('figure_id')}"),
                        "section_title": f.get("label", "Figure"),
                        "chunk_type": f.get("figure_type", "figure"),
                        "quote": (f.get("caption") or f.get("label") or "")[:520],
                        "figure_id": f.get("figure_id"),
                        "image_file": f.get("image_file"),
                        "image_relpath": f.get("image_relpath"),
                        "image_url": _visual_image_url(f, paper_id),
                        "label": f.get("label"),
                        "is_figure": True,
                        "is_page_visual": bool(f.get("is_page_visual_chunk")),
                        "source_paper_id": _figure_source_id(f, paper_id),
                        "document_id": _figure_source_id(f, paper_id),
                    }
                    for i, f in enumerate(figure_chunks)
                ],
            }

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
            "image_relpath": primary_fig.get("image_relpath"),
            "image_url": _visual_image_url(primary_fig, paper_id),
            "page":       page,
            "source_paper_id": primary_source_id,
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
                    "image_relpath": f.get("image_relpath"),
                    "image_url": _visual_image_url(f, paper_id),
                    "label": f.get("label"),
                    "is_figure": True,
                    "is_page_visual": bool(f.get("is_page_visual_chunk")),
                    "source_paper_id": _figure_source_id(f, paper_id),
                    "document_id": _figure_source_id(f, paper_id),
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
        source_id = _figure_source_id(fig, paper_id)
        image_relpath = fig.get("image_relpath")
        png_bytes = (
            _load_visual_png(source_id, img_f, str(image_relpath))
            if image_relpath
            else _load_figure_png(source_id, img_f)
        )
        if png_bytes is not None:
            retrieval_bbox = (
                _retrieval_region_bbox(fig)
                if fig.get("is_page_visual_chunk") else None
            )
            vision_bytes = png_bytes
            vision_input_kind = "full_visual"
            if retrieval_bbox is not None:
                cropped = _crop_visual_png(png_bytes, retrieval_bbox)
                if cropped is not None:
                    vision_bytes = cropped
                    vision_input_kind = "retrieval_crop"
            visual_index = len(loaded_figures) + 1
            images_b64.append(base64.b64encode(vision_bytes).decode("ascii"))
            loaded_figures.append({
                **fig,
                "source_paper_id": source_id,
                "_vision_evidence_id": f"V{visual_index}",
                "_vision_ref_id": visual_index,
                "_source_title": _source_title(source_id, source_registry),
                "_vision_input_kind": vision_input_kind,
                "_retrieval_bbox_normalized": (
                    {
                        "x0": retrieval_bbox[0],
                        "y0": retrieval_bbox[1],
                        "x1": retrieval_bbox[2],
                        "y1": retrieval_bbox[3],
                    }
                    if retrieval_bbox is not None else None
                ),
            })

    if not images_b64:
        return _caption_fallback("figure image files not found on disk")

    primary_fig = loaded_figures[0]
    primary_source_id = _figure_source_id(primary_fig, paper_id)
    figure_id = primary_fig.get("figure_id", "")
    label = primary_fig.get("label", "")
    caption = primary_fig.get("caption", "")
    image_file = primary_fig.get("image_file", "")
    page = primary_fig.get("page", 1)

    text_context_parts: list[str] = []
    for offset, chunk in enumerate(text_chunks, start=1):
        ref_id = len(loaded_figures) + offset
        source_id = _figure_source_id(chunk, paper_id)
        source_title = _source_title(source_id, source_registry)
        section = chunk.get("section_title") or chunk.get("section") or "Text"
        text_context_parts.append(
            f"[T{offset} -> citation [{ref_id}] | source_id={source_id} | "
            f"source_title={source_title} | section={section} | p. {chunk.get('page')}]\n"
            + str(chunk.get("text") or "")[:2500]
        )
    text_context = "\n\n".join(text_context_parts)

    # -- Two-pass visual analysis: transcribe pixels, then synthesize an answer --
    visual_observations: dict[str, str] = {}
    try:
        observation_generation = await generate_result(
            _build_visual_observation_prompt(
                question,
                loaded_figures,
                source_registry,
            ),
            temperature=0.0,
            images=images_b64,
            model=model,
            seed=seed,
            top_p=(decoding_options or {}).get("top_p"),
            num_ctx=(decoding_options or {}).get("num_ctx"),
            num_predict=min(
                int((decoding_options or {}).get("num_predict") or 900),
                900,
            ),
        )
        visual_observations = _parse_visual_observations(
            observation_generation.response,
            {
                str(figure["_vision_evidence_id"])
                for figure in loaded_figures
            },
        )
        if not visual_observations:
            logger.warning(
                "First-pass visual transcription did not return valid source-scoped JSON"
            )
    except Exception as exc:
        logger.warning("First-pass visual transcription failed: %s", exc)

    prompt = _build_multi_vision_prompt(
        question,
        loaded_figures,
        text_context,
        paper_title,
        visual_observations,
        source_registry,
    )
    try:
        generation = await generate_result(
            prompt,
            temperature=float((decoding_options or {}).get("temperature", 0.1)),
            images=images_b64,
            model=model,
            seed=seed,
            top_p=(decoding_options or {}).get("top_p"),
            num_ctx=(decoding_options or {}).get("num_ctx"),
            num_predict=(decoding_options or {}).get("num_predict"),
        )
        answer_text = generation.response
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
        visual_evidence_id = str(fig.get("_vision_evidence_id") or f"V{idx}")
        visual_observation = visual_observations.get(visual_evidence_id, "")
        source_id = _figure_source_id(fig, paper_id)
        raw_bbox = (
            fig.get("_retrieval_bbox_normalized")
            or fig.get("bbox_normalized")
            or fig.get("bbox")
            or {}
        )
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
            evidence_id=visual_evidence_id,
            model_id=model or OLLAMA_MODEL,
        )
        resolved_subregions = VisualGroundingService.resolve_regions_from_proposal(
            proposal=proposal,
            parent_page_box=parent_box,
            document_id=source_id,
            page_number=fig.get("page", 1),
        )
        all_citations.append({
            "ref_id": idx,
            "page": fig.get("page", 1),
            "chunk_id": fig.get("chunk_id", f"fig_{f_id}"),
            "section_title": fig.get("label") or ("Figure" if fig.get("figure_type") == "figure" else "Table"),
            "chunk_type": fig.get("figure_type", "figure"),
            "quote": (str(fig.get("caption") or fig.get("label") or ""))[:520],
            "visual_evidence_id": visual_evidence_id,
            "visual_observation": visual_observation or None,
            "visual_observation_model_generated": bool(visual_observation),
            "figure_id": f_id,
            "image_file": fig.get("image_file"),
            "image_relpath": fig.get("image_relpath"),
            "image_url": _visual_image_url(fig, paper_id),
            "label": fig.get("label"),
            "is_figure": True,
            "is_page_visual": bool(fig.get("is_page_visual_chunk")),
            "source_paper_id": source_id,
            "source_title": fig.get("_source_title") or _source_title(source_id, source_registry),
            "document_id": source_id,
            "bbox_normalized": parent_box.model_dump(),
            "vision_input_kind": fig.get("_vision_input_kind", "full_visual"),
            "retrieval_candidate_regions": list(fig.get("candidate_regions") or []),
            "visual_retrieval_backend": fig.get("visual_retrieval_backend"),
            "visual_retrieval_model": fig.get("visual_retrieval_model"),
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

    # Append the exact text chunks shown in the prompt using the same IDs.
    start_ref = len(all_citations) + 1
    for offset, tc in enumerate(text_chunks, start=1):
        ref_id = start_ref + offset - 1
        source_id = _figure_source_id(tc, paper_id)
        all_citations.append({
            "ref_id": ref_id,
            "page": tc.get("page", 1),
            "chunk_id": tc.get("chunk_id", f"text_{ref_id}"),
            "section_title": tc.get("section_title") or tc.get("section") or "Text Context",
            "chunk_type": tc.get("chunk_type", "text"),
            "quote": (tc.get("text") or tc.get("content") or "")[:400],
            "is_figure": False,
            "text_evidence_id": f"T{offset}",
            "source_paper_id": source_id,
            "source_title": _source_title(source_id, source_registry),
            "document_id": tc.get("document_id") or source_id,
        })

    import re
    cleaned_answer = re.sub(r"```json[\s\S]*?```", "", answer_text)
    cleaned_answer = re.sub(r"\*\*Subregions\*\*[\s\S]*", "", cleaned_answer).strip()
    # Normalize pseudo-LaTeX model names like $\text{BERT}_{\text{BASE}}$ -> BERT-BASE
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{?\\text\{([^}]+)\}?\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}_\{([^}]+)\}\$", r"\1-\2", cleaned_answer)
    cleaned_answer = re.sub(r"\$\\text\{([^}]+)\}\$", r"\1", cleaned_answer)
    cleaned_answer = re.sub(r"\\text\{([^}]+)\}", r"\1", cleaned_answer)

    # Strip empty or trailing boilerplate sections
    cleaned_answer = re.sub(
        r"\n+\*\*(?:Limits|Limitations)\*\*\s*(?:-\s*(?:none|n/a|\s*))?(?=\n\*\*|\Z)",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = re.sub(
        r"\n+\*\*Row-by-Row Analysis[^\n]*\*\*\s*(?=\n\*\*|\Z)",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = re.sub(
        r"\n+\*\*Relation to Architectural Mechanism[^\n]*\*\*\s*(?=\n\*\*|\Z)",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = re.sub(
        r"\n+\*\*Key Insights[^\n]*\*\*\s*(?=\n\*\*|\Z)",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = re.sub(
        r"\n+\*\*Key Findings & Mechanism[^\n]*\*\*\s*(?=\n\*\*|\Z)",
        "",
        cleaned_answer,
        flags=re.IGNORECASE,
    )
    cleaned_answer = cleaned_answer.strip()

    labels_all = ", ".join(f.get("label", "Figure") for f in loaded_figures)
    return {
        "answer": cleaned_answer or answer_text.strip(),
        "figure_id": figure_id,
        "label": labels_all,
        "caption": caption,
        "image_file": image_file,
        "image_relpath": primary_fig.get("image_relpath"),
        "image_url": _visual_image_url(primary_fig, paper_id),
        "page": page,
        "source_paper_id": primary_source_id,
        "model_used": generation.resolved_model or model or OLLAMA_MODEL,
        "model_digest": generation.model_digest,
        "quantization": generation.quantization,
        "generation_options": generation.options,
        "fallback": False,
        "citations": all_citations,
        "loaded_figures": loaded_figures,
        "visual_observations": visual_observations,
        "visual_observation_model_generated": bool(visual_observations),
    }


async def answer_with_figure(
    question: str,
    figure_chunk: dict[str, Any],
    context_chunks: list[dict[str, Any]],
    paper_id: str,
    paper_metadata: dict[str, Any] | None = None,
    model: str | None = None,
    seed: int | None = None,
    decoding_options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single-figure wrapper around answer_with_multimodal_evidence."""
    return await answer_with_multimodal_evidence(
        question=question,
        figure_chunks=[figure_chunk],
        context_chunks=context_chunks,
        paper_id=paper_id,
        paper_metadata=paper_metadata,
        model=model,
        seed=seed,
        decoding_options=decoding_options,
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
    seed: int | None = None,
    decoding_options: dict[str, Any] | None = None,
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
        generation = await generate_result(
            prompt=prompt,
            images=[image_b64],
            model=model_to_use,
            temperature=float((decoding_options or {}).get("temperature", 0.1)),
            seed=seed,
            top_p=(decoding_options or {}).get("top_p"),
            num_ctx=(decoding_options or {}).get("num_ctx"),
            num_predict=(decoding_options or {}).get("num_predict"),
        )
        response_text = generation.response
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
        "model_used": generation.resolved_model or model_to_use,
        "model_digest": generation.model_digest,
        "quantization": generation.quantization,
        "generation_options": generation.options,
        "fallback": False,
        "citations": citations,
        "bbox_normalized": bbox_norm,
        "image_url": f"/api/papers/{paper_id}/snippets/{snippet_id}.png",
    }
