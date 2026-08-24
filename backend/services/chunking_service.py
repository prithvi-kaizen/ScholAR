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


def _extract_headings(page_text: str) -> list[tuple[str, int]]:
    """Extract section and subsection headings with their hierarchy levels (1=Section, 2=Subsection)."""
    headings: list[tuple[str, int]] = []
    lines = page_text.splitlines()

    # Regex for numbered headings: "3 Methods", "3.2 Attention Mechanism", "3.2.1 Scaled Dot-Product"
    numbered_re = re.compile(
        r"^\s*(\d+(?:\.\d+)*)\.?\s+([A-Z][A-Za-z0-9\s\-–,:(/)]+)\s*$"
    )
    # Common unnumbered standard section names
    unnumbered_re = re.compile(
        r"^\s*(Abstract|Introduction|Background|Related Work|Methodology|Methods|Approach|Architecture|Framework|Experiments?|Experimental Setup|Evaluation|Results?|Analysis|Discussion|Limitations?|Conclusions?|Future Work|Broader Impacts?|References|Appendix)\s*$",
        re.IGNORECASE,
    )

    for line in lines:
        cleaned = line.strip()
        if not cleaned or len(cleaned) > 80:
            continue

        num_m = numbered_re.match(cleaned)
        if num_m:
            num_str, title_str = num_m.group(1), num_m.group(2).strip()
            # Count dots to determine level: "3" -> level 1, "3.2" -> level 2, "3.2.1" -> level 3
            level = num_str.count(".") + 1
            full_title = f"{num_str} {title_str.title()}"
            headings.append((full_title, level))
            continue

        unnum_m = unnumbered_re.match(cleaned)
        if unnum_m:
            headings.append((unnum_m.group(1).title(), 1))

    return headings


def _section_title(page_text: str, page_number: int) -> str:
    """Return the most specific section or subsection title for the page."""
    lowered = page_text.lower()
    if page_number == 1 and "abstract" in lowered:
        return "Abstract"

    headings = _extract_headings(page_text)
    if headings:
        # Return the most specific (latest) heading on the page
        return headings[-1][0]

    # Fallback to keyword matching
    candidates = re.findall(
        r"(?:^|\s)(?:\d+(?:\.\d+)?\s+)?(Abstract|Introduction|Background|Related Work|Method|Methodology|Approach|Model|Architecture|Framework|Experiments?|Experimental Setup|Evaluation|Results?|Analysis|Discussion|Limitations?|Conclusion|Future Work)\b",
        page_text,
        flags=re.IGNORECASE,
    )
    return candidates[-1].strip().title() if candidates else ("Abstract" if page_number == 1 else "Body")


def _section_path(page_text: str, page_number: int) -> list[str]:
    """Extract hierarchical section outline path [Section, Subsection] from page text."""
    if page_number == 1 and "abstract" in page_text.lower():
        return ["Abstract"]

    headings = _extract_headings(page_text)
    if not headings:
        title = _section_title(page_text, page_number)
        return [title] if title else ["Body"]

    # Build hierarchical path preserving parent -> child
    path: list[str] = []
    main_section: str | None = None
    sub_section: str | None = None

    for title, level in headings:
        if level == 1:
            main_section = title
            sub_section = None
        elif level >= 2:
            if not main_section:
                main_section = f"Section {title.split('.')[0] if '.' in title else ''}".strip()
            sub_section = title

    if main_section:
        path.append(main_section)
    if sub_section and sub_section != main_section:
        path.append(sub_section)

    return path or [_section_title(page_text, page_number)]


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
    """Create page-preserving chunks with hierarchical section-path prefixing.

    Args:
        pages: List of {page, text} dicts from pdf_service.extract_pages().
        target_words: Target chunk size in words.
        overlap_words: Overlap between consecutive chunks on the same page.
        source_paper_id: When set, every chunk gets a ``source_paper_id`` field
            so cross-paper provenance is unambiguous in multi-document sessions.
    """
    chunks: list[dict[str, Any]] = []
    chunk_number = 1

    for page in pages:
        text = page.get("text", "")
        page_number = page.get("page", 1)
        section_title = _section_title(text, page_number)
        sec_path = _section_path(text, page_number)
        section_prefix = " > ".join(sec_path)
        words = text.split()
        if not words:
            continue

        start_word = 0
        while start_word < len(words):
            end_word = min(start_word + target_words, len(words))
            chunk_text = " ".join(words[start_word:end_word]).strip()
            char_start = len(" ".join(words[:start_word]))
            char_end = char_start + len(chunk_text)

            # Prefix the section hierarchy for retrieval while preserving clean original text
            retrieval_text = f"{section_prefix}. {chunk_text}" if section_prefix else chunk_text

            chunk: dict[str, Any] = {
                "chunk_id": f"chunk_{chunk_number:03d}",
                "page": page_number,
                "text": chunk_text,
                "original_text": chunk_text,
                "retrieval_text": retrieval_text,
                "paragraph_text": _paragraph_text(chunk_text),
                "section_title": section_title,
                "section_path": sec_path,
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
    """Convert figure/table metadata records into retrieval-compatible chunks."""
    chunks: list[dict] = []
    for fig in figures:
        caption = fig.get("caption", "").strip()
        label = fig.get("label", "")
        fig_type = fig.get("figure_type", "figure")
        sec_title = "Figure" if fig_type == "figure" else "Table"
        sec_path = [sec_title]
        
        text = f"{label}: {caption}" if caption else label
        retrieval_text = f"{sec_title} > {text}"

        chunk: dict = {
            "chunk_id": f"fig_{fig.get('figure_id')}",
            "page": fig.get("page", 1),
            "text": text,
            "original_text": text,
            "retrieval_text": retrieval_text,
            "paragraph_text": text[:300],
            "section_title": sec_title,
            "section_path": sec_path,
            "chunk_type": fig_type,  # "figure" | "table"
            "char_start": 0,
            "char_end": len(text),
            "is_figure_chunk": True,
            "figure_id": fig.get("figure_id"),
            "label": label,
            "image_file": fig.get("image_file"),
            "bbox": fig.get("bbox"),
            "caption": caption,
        }
        if source_paper_id is not None:
            chunk["source_paper_id"] = source_paper_id

        chunks.append(chunk)
    return chunks
