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
) -> list[dict[str, Any]]:
    """Create simple page-preserving chunks suitable for transparent MVP retrieval."""
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

            chunks.append(
                {
                    "chunk_id": f"chunk_{chunk_number:03d}",
                    "page": page_number,
                    "text": chunk_text,
                    "paragraph_text": _paragraph_text(chunk_text),
                    "section_title": section_title,
                    "chunk_type": _chunk_type(chunk_text, section_title, page_number),
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            chunk_number += 1

            if end_word >= len(words):
                break
            start_word = max(end_word - overlap_words, start_word + 1)

    return chunks
