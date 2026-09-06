"""Unit tests for adaptive vision prompt construction and empty header stripping."""

import re
import pytest
from backend.services.vision_service import _build_multi_vision_prompt


def test_build_prompt_conceptual_intent():
    figures = [
        {"figure_type": "table", "label": "Table 3", "caption": "Variations on the Transformer architecture"},
        {"figure_type": "figure", "label": "Figure 1", "caption": "The Transformer architecture"},
    ]
    question = "Why did the authors prefer sinusoidal positional encoding over learned positional embeddings?"
    
    prompt = _build_multi_vision_prompt(
        question=question,
        figures=figures,
        text_context="Section 3.5 positional encodings...",
        paper_title="Attention Is All You Need",
    )

    # Conceptual question should NOT force a rigid Markdown table reconstruction or full row dump
    assert "CRITICAL INSTRUCTIONS FOR FULL TABULAR RECONSTRUCTION" not in prompt
    assert "CRITICAL INSTRUCTIONS FOR TARGETED MULTIMODAL SYNTHESIS" in prompt
    assert "RELEVANCE RULE" in prompt
    # Instructions specify to omit Limits if empty
    assert "omit the Limits section entirely" in prompt


def test_build_prompt_wants_full_table_intent():
    figures = [
        {"figure_type": "table", "label": "Table 3", "caption": "Variations on the Transformer architecture"},
    ]
    question = "Can you reconstruct the table of results showing all rows from Table 3?"
    
    prompt = _build_multi_vision_prompt(
        question=question,
        figures=figures,
        text_context="Section 6 results...",
        paper_title="Attention Is All You Need",
    )

    # When explicitly asking for the table, it SHOULD request the structured table
    assert "CRITICAL INSTRUCTIONS FOR FULL TABULAR RECONSTRUCTION" in prompt
    assert "**Structured Table**" in prompt


def test_empty_section_stripping():
    # Simulate a raw model response that left Limits empty or with placeholder
    raw_answer = (
        "**Answer**\n"
        "The authors chose sinusoidal encodings because it allows sequence extrapolation [1].\n\n"
        "**Key Findings & Mechanism**\n"
        "- **Architectural Link**: Sinusoids encode relative position smoothly [2].\n\n"
        "**Limits**\n\n"
        "**Cited Sources:**\n"
        "- [1] Page 6"
    )

    # Run the exact regex cleaners used in vision_service
    cleaned = re.sub(
        r"\n+\*\*(?:Limits|Limitations)\*\*\s*(?:-\s*(?:none|n/a|\s*))?(?=\n\*\*|\Z)",
        "",
        raw_answer,
        flags=re.IGNORECASE,
    )

    assert "**Limits**" not in cleaned
    assert "The authors chose sinusoidal encodings" in cleaned
    assert "**Cited Sources:**" in cleaned
