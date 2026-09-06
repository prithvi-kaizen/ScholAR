"""Unit tests for layout-aware table extraction and full bounding box coverage."""

from pathlib import Path
import pytest
import fitz

from backend.services.pdf_service import (
    _caption_bbox,
    _find_matching_table_bbox,
    extract_figures,
)


def test_find_matching_table_bbox_attention_table3():
    pdf_path = Path("backend/data/papers/1706.03762/paper.pdf")
    assert pdf_path.exists(), "Test PDF 1706.03762 must exist"

    doc = fitz.open(pdf_path)
    page_9 = doc[8]  # 0-indexed page 9

    cap_bbox = _caption_bbox(page_9, "Table 3")
    assert cap_bbox is not None, "Should locate Table 3 caption on page 9"

    table_rect = _find_matching_table_bbox(page_9, cap_bbox)
    assert table_rect is not None, "Should detect table bounding box via PyMuPDF"

    # The table should span from near the caption down past row (E) (y1 > 380)
    assert table_rect.y0 < cap_bbox.y1
    assert table_rect.y1 >= 380.0, f"Table rect y1 should be at least 380 to include row E, got {table_rect.y1}"

    # Verify text inside this clip includes row (E)
    clip_text = page_9.get_text("text", clip=table_rect)
    assert "(E)" in clip_text
    assert "positional embedding" in clip_text
    assert "4.92" in clip_text
    assert "25.7" in clip_text


def test_extract_figures_table3_full_content():
    pdf_path = Path("backend/data/papers/1706.03762/paper.pdf")
    scratch_dir = Path("scratch/test_figures_verify")
    scratch_dir.mkdir(parents=True, exist_ok=True)

    records = extract_figures(pdf_path, scratch_dir)
    table_3_record = next((r for r in records if "Table 3" in r.get("label", "")), None)

    assert table_3_record is not None, "Table 3 must be extracted"
    body_text = table_3_record.get("body_text", "")

    # Must contain base, (A), (B), (C), (D), (E), and big model
    assert "(A)" in body_text
    assert "(B)" in body_text
    assert "(C)" in body_text
    assert "(D)" in body_text
    assert "(E)" in body_text
    assert "positional embedding instead of sinusoids" in body_text
    assert "big" in body_text
