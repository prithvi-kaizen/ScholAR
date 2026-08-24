"""Deterministic Tabular Arithmetic & NumericPlan Schemas for ScholAR.

Decouples mathematical operations over table cells from LLM text generation:
- NumericOp: Supported arithmetic operations (difference, ratio, percent_change, sum, mean, max, min)
- CellOperand: Typed cell value extracted from structured TableData
- NumericPlan: Deterministic calculation specification
- NumericExecutionResult: Exact decimal result and formatted claim injection
"""

from __future__ import annotations

from decimal import Decimal
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field


class NumericOp(str, Enum):
    """Supported deterministic tabular arithmetic operations."""
    DIFFERENCE = "difference"              # operand_a - operand_b
    PERCENT_CHANGE = "percent_change"      # ((operand_a - operand_b) / operand_b) * 100
    RATIO = "ratio"                        # operand_a / operand_b
    SUM = "sum"                            # sum(operands)
    MEAN = "mean"                          # mean(operands)
    MAX = "max"                            # max(operands)
    MIN = "min"                            # min(operands)


class CellOperand(BaseModel):
    """Specific cell operand located inside a structured table."""
    table_id: str                          # e.g., "tab_001" or "E_TAB_01"
    row: int
    col: int
    raw_text: str = ""
    parsed_value: float = 0.0
    label: str = ""                        # e.g., "Transformer (big) BLEU"


class NumericPlan(BaseModel):
    """Deterministic calculation plan for table reasoning."""
    operation: NumericOp = NumericOp.DIFFERENCE
    operands: list[CellOperand] = Field(default_factory=list)
    target_metric: str = ""                # e.g., "BLEU-4 score", "parameter count"
    description: str = ""


class NumericExecutionResult(BaseModel):
    """Exact result of deterministic calculation plan."""
    operation: NumericOp
    computed_value: float
    formatted_value: str                   # e.g., "+4.054%", "-1.10", "1.18x"
    formatted_statement: str
    is_exact: bool = True
    evidence_ids: list[str] = Field(default_factory=list)
