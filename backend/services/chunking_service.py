from __future__ import annotations

from typing import Any


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
                    "page": page.get("page", 1),
                    "text": chunk_text,
                    "char_start": char_start,
                    "char_end": char_end,
                }
            )
            chunk_number += 1

            if end_word >= len(words):
                break
            start_word = max(end_word - overlap_words, start_word + 1)

    return chunks
