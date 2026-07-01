from __future__ import annotations

import re
from typing import Any


SECTION_PATTERNS: list[tuple[str, str]] = [
    ("abstract", r"\babstract\b"),
    ("introduction", r"\b(?:\d+\.?\s*)?introduction\b"),
    ("method", r"\b(?:method|methodology|approach|model|architecture|framework)\b"),
    ("experiment", r"\b(?:experiment|experimental setup|evaluation|dataset|benchmark)\b"),
    ("result", r"\b(?:result|results|analysis|findings)\b"),
    ("limitation", r"\b(?:limitation|limitations|discussion|future work|conclusion)\b"),
]


def _section_title(page_text: str, page_number: int) -> str:
    lowered = page_text.lower()
    if page_number == 1 and "abstract" in lowered:
        return "Abstract"
    candidates = re.findall(
        r"(?:^|\s)(?:\d+(?:\.\d+)?\s+)?(Abstract|Introduction|Background|Related Work|Method|Methodology|Approach|Model|Architecture|Framework|Experiments?|Experimental Setup|Evaluation|Results?|Analysis|Discussion|Limitations?|Conclusion|Future Work)\b",
        page_text,
        flags=re.IGNORECASE,
    )
    return candidates[-1].strip().title() if candidates else ("Abstract" if page_number == 1 else "Body")


def _chunk_type(text: str, section_title: str, page_number: int) -> str:
    combined = f"{section_title} {text[:900]}".lower()
    if page_number == 1 and "abstract" in combined:
        return "abstract"
    for chunk_type, pattern in SECTION_PATTERNS:
        if re.search(pattern, combined, flags=re.IGNORECASE):
            return chunk_type
    if any(term in combined for term in ("outperform", "achieve", "accuracy", "bleu", "score", "significant")):
        return "result"
    return "body"


def _paragraph_text(text: str, max_chars: int = 900) -> str:
    cleaned = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    paragraph = " ".join(sentences[:5]).strip()
    return paragraph[:max_chars] or cleaned[:max_chars]


def chunk_pages(
    pages: list[dict[str, Any]],
    target_words: int = 1400,
    overlap_words: int = 120,
    source_paper_id: str | None = None,
) -> list[dict[str, Any]]:
    """Create simple page-preserving chunks suitable for transparent MVP retrieval.

    Args:
        pages: List of {page, text} dicts from pdf_service.extract_pages().
        target_words: Target chunk size in words.
        overlap_words: Overlap between consecutive chunks on the same page.
        source_paper_id: When set, every chunk gets a ``source_paper_id`` field
            so cross-paper provenance is unambiguous in multi-document sessions.
            The single-paper flow passes ``None`` and no field is added.
    """
    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for page in pages:
        text = page.get("text", "")
        page_number = page.get("page", 1)
        section_title = _section_title(text, page_number)
        words = text.split()
        if not words:
            continue

        start_word = 0
        while start_word < len(words):
            end_word = min(start_word + target_words, len(words))
            chunk_text = " ".join(words[start_word:end_word]).strip()
            char_start = len(" ".join(words[:start_word]))
            char_end = char_start + len(chunk_text)

            chunk: dict[str, Any] = {
                "chunk_id": f"chunk_{chunk_number:03d}",
                "page": page_number,
                "text": chunk_text,
                "paragraph_text": _paragraph_text(chunk_text),
                "section_title": section_title,
                "chunk_type": _chunk_type(chunk_text, section_title, page_number),
                "char_start": char_start,
                "char_end": char_end,
            }
            if source_paper_id is not None:
                chunk["source_paper_id"] = source_paper_id

            chunks.append(chunk)
            chunk_number += 1

            if end_word >= len(words):
                break
            start_word = max(end_word - overlap_words, start_word + 1)

    return chunks


def chunk_figures(
    figures: list[dict],
    source_paper_id: str | None = None,
) -> list[dict]:
    """Convert figure/table metadata records into retrieval-compatible chunks.

    Each figure chunk has:
    - ``chunk_type``: "figure" or "table"
    - ``text``: the caption text (used for BM25 matching and visual-cue boosting)
    - ``figure_id``, ``label``, ``image_file``, ``bbox``, ``page``: figure metadata
    - ``is_figure_chunk``: True — used by retrieval and vision path to identify
      these chunks without string-matching chunk_type

    Figure chunks are appended to the normal text chunk pool so they participate
    in a single unified BM25 ranking pass.
    """
    chunks: list[dict] = []
    for fig in figures:
        caption = fig.get("caption", "").strip()
        label = fig.get("label", "")
        # BM25 text = label + caption — richer signal than caption alone
        text = f"{label}: {caption}" if caption else label

        chunk: dict = {
            "chunk_id":       f"fig_{fig.get('figure_id')}",
            "page":           fig.get("page", 1),
            "text":           text,
            "paragraph_text": text[:300],
            "section_title":  "Figure" if fig.get("figure_type") == "figure" else "Table",
            "chunk_type":     fig.get("figure_type", "figure"),  # "figure" | "table"
            "char_start":     0,
            "char_end":       len(text),
            "is_figure_chunk": True,
            "figure_id":      fig.get("figure_id"),
            "label":          label,
            "image_file":     fig.get("image_file"),
            "bbox":           fig.get("bbox"),
            "caption":        caption,
        }
        if source_paper_id is not None:
            chunk["source_paper_id"] = source_paper_id

        chunks.append(chunk)
    return chunks

