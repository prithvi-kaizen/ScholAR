"""Deterministic Table Arithmetic & NumericPlan Execution Engine for ScholAR.

Executes exact decimal arithmetic over structured table cells:
- Eliminates LLM arithmetic and rounding hallucinations
- Supports difference, percent change, ratio, sum, mean, min, max
- Cleans scientific cell values (removes error margins +/- 0.2, footnotes, units)
"""

from __future__ import annotations

import logging
import re
from decimal import Decimal, ROUND_HALF_UP
from typing import Any

from backend.schemas.numeric_plan import (
    CellOperand,
    NumericExecutionResult,
    NumericOp,
    NumericPlan,
)

logger = logging.getLogger("scholar.table_arithmetic")


class TableArithmeticService:
    """Performs deterministic calculations over scientific table cells."""

    @classmethod
    def clean_and_parse_number(cls, raw: str) -> Decimal | None:
        """Parse raw table cell text into an exact Decimal.

        Handles:
        - Percentages: '84.7%' -> 84.7
        - Error bounds: '28.4 +/- 0.2' -> 28.4
        - Footnotes: '88.4*' or '88.4^a' -> 88.4
        - Commas: '1,234.56' -> 1234.56
        - Exponents: '3.3e18' -> 3300000000000000000
        """
        if not raw:
            return None

        # Remove LaTeX wrappers and footnotes
        cleaned = re.sub(r"[\$\\^\*†‡§]", "", str(raw)).strip()
        # Remove trailing units and notes in brackets
        cleaned = re.sub(r"\[.*?\]|\(.*?\)", "", cleaned).strip()
        # Take base value if +/- present
        if "±" in cleaned:
            cleaned = cleaned.split("±")[0].strip()
        elif "+/-" in cleaned:
            cleaned = cleaned.split("+/-")[0].strip()

        # Remove commas and percentage signs
        cleaned = cleaned.replace(",", "").replace("%", "").strip()

        # Find first floating point or scientific number
        match = re.search(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", cleaned)
        if not match:
            return None

        try:
            val_str = match.group(0)
            return Decimal(val_str)
        except Exception:
            return None

    @classmethod
    def execute_plan(cls, plan: NumericPlan) -> NumericExecutionResult:
        """Execute a NumericPlan with exact Python Decimal arithmetic."""
        operands = plan.operands
        if not operands:
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=0.0,
                formatted_value="N/A",
                formatted_statement="No operands provided for calculation.",
                is_exact=False,
            )

        dec_values = []
        for op in operands:
            if op.raw_text:
                parsed = cls.clean_and_parse_number(op.raw_text)
                if parsed is not None:
                    dec_values.append(parsed)
                    continue
            dec_values.append(Decimal(str(op.parsed_value)))

        table_ids = list({op.table_id for op in operands if op.table_id})

        if plan.operation == NumericOp.DIFFERENCE:
            if len(dec_values) < 2:
                return NumericExecutionResult(
                    operation=plan.operation,
                    computed_value=float(dec_values[0]) if dec_values else 0.0,
                    formatted_value="Error: Arity < 2",
                    formatted_statement="Difference requires at least two operands.",
                    is_exact=False,
                    evidence_ids=table_ids,
                )
            diff = dec_values[0] - dec_values[1]
            diff_float = float(diff)
            sign = "+" if diff_float > 0 else ""
            fmt_val = f"{sign}{diff_float:.4g}"
            stmt = f"The difference in {plan.target_metric or 'performance'} is {fmt_val} ({operands[0].label or 'A'}: {operands[0].parsed_value} vs {operands[1].label or 'B'}: {operands[1].parsed_value})."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=diff_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.PERCENT_CHANGE:
            if len(dec_values) < 2:
                return NumericExecutionResult(
                    operation=plan.operation,
                    computed_value=0.0,
                    formatted_value="Error: Arity < 2",
                    formatted_statement="Percent change requires at least two operands.",
                    is_exact=False,
                    evidence_ids=table_ids,
                )
            if dec_values[1] == 0:
                return NumericExecutionResult(
                    operation=plan.operation,
                    computed_value=0.0,
                    formatted_value="Undefined (division by zero)",
                    formatted_statement=f"Cannot compute percent change: baseline ({operands[1].label or 'baseline'}) is zero.",
                    is_exact=False,
                    evidence_ids=table_ids,
                )
            pct = ((dec_values[0] - dec_values[1]) / abs(dec_values[1])) * Decimal(100)
            pct_float = float(pct)
            sign = "+" if pct_float > 0 else ""
            fmt_val = f"{sign}{pct_float:.3f}%"
            stmt = f"A change of {fmt_val} in {plan.target_metric or 'metric'} ({operands[0].label or 'A'}: {operands[0].parsed_value} compared to baseline {operands[1].label or 'B'}: {operands[1].parsed_value})."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=pct_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.RATIO:
            if len(dec_values) < 2:
                return NumericExecutionResult(
                    operation=plan.operation,
                    computed_value=0.0,
                    formatted_value="Error: Arity < 2",
                    formatted_statement="Ratio requires at least two operands.",
                    is_exact=False,
                    evidence_ids=table_ids,
                )
            if dec_values[1] == 0:
                return NumericExecutionResult(
                    operation=plan.operation,
                    computed_value=0.0,
                    formatted_value="Undefined (division by zero)",
                    formatted_statement=f"Cannot compute ratio: denominator ({operands[1].label or 'denominator'}) is zero.",
                    is_exact=False,
                    evidence_ids=table_ids,
                )
            ratio = dec_values[0] / dec_values[1]
            ratio_float = float(ratio)
            fmt_val = f"{ratio_float:.3f}x"
            stmt = f"The ratio of {operands[0].label or 'A'} to {operands[1].label or 'B'} is {fmt_val}."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=ratio_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.MEAN:
            mean_val = sum(dec_values) / Decimal(len(dec_values))
            mean_float = float(mean_val)
            fmt_val = f"{mean_float:.4g}"
            stmt = f"The average {plan.target_metric or 'value'} across {len(dec_values)} entries is {fmt_val}."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=mean_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.MIN:
            min_val = min(dec_values)
            min_float = float(min_val)
            fmt_val = f"{min_float:.4g}"
            stmt = f"The minimum {plan.target_metric or 'value'} across {len(dec_values)} entries is {fmt_val}."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=min_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.MAX:
            max_val = max(dec_values)
            max_float = float(max_val)
            fmt_val = f"{max_float:.4g}"
            stmt = f"The maximum {plan.target_metric or 'value'} across {len(dec_values)} entries is {fmt_val}."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=max_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        elif plan.operation == NumericOp.SUM:
            sum_val = sum(dec_values)
            sum_float = float(sum_val)
            fmt_val = f"{sum_float:.4g}"
            stmt = f"The total {plan.target_metric or 'sum'} is {fmt_val}."
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=sum_float,
                formatted_value=fmt_val,
                formatted_statement=stmt,
                is_exact=True,
                evidence_ids=table_ids,
            )

        else:
            return NumericExecutionResult(
                operation=plan.operation,
                computed_value=0.0,
                formatted_value="Unsupported",
                formatted_statement=f"Unsupported operation: {plan.operation}",
                is_exact=False,
                evidence_ids=table_ids,
            )

    @classmethod
    def extract_and_calculate_from_table_text(
        cls,
        table_text: str,
        entity_a: str,
        entity_b: str,
        op: NumericOp = NumericOp.PERCENT_CHANGE,
        metric_keyword: str = "",
        table_id: str = "E_TAB_01",
    ) -> NumericExecutionResult | None:
        """Extract two numerical values corresponding to two entities from a markdown table and compute arithmetic."""
        lines = [line.strip() for line in table_text.splitlines() if "|" in line]
        if len(lines) < 2:
            return None

        # Parse header
        headers = [h.strip() for h in lines[0].split("|") if h.strip()]
        target_col_idx: int | None = None

        if metric_keyword:
            for idx, h in enumerate(headers):
                if metric_keyword.lower() in h.lower():
                    target_col_idx = idx
                    break

        val_a: Decimal | None = None
        val_b: Decimal | None = None

        for line in lines[1:]:
            # Skip separator line |---|---|
            if set(line.replace("|", "").strip()) <= {"-", ":"}:
                continue

            line_lowered = line.lower()
            cells = [c.strip() for c in line.split("|") if c.strip()]
            if not cells:
                continue

            if entity_a.lower() in line_lowered and val_a is None:
                if target_col_idx is not None and target_col_idx < len(cells):
                    parsed = cls.clean_and_parse_number(cells[target_col_idx])
                    if parsed is not None:
                        val_a = parsed
                if val_a is None:
                    # Fallback: first numeric cell after entity name
                    for cell in cells[1:]:
                        parsed = cls.clean_and_parse_number(cell)
                        if parsed is not None:
                            val_a = parsed
                            break

            if entity_b.lower() in line_lowered and val_b is None:
                if target_col_idx is not None and target_col_idx < len(cells):
                    parsed = cls.clean_and_parse_number(cells[target_col_idx])
                    if parsed is not None:
                        val_b = parsed
                if val_b is None:
                    for cell in cells[1:]:
                        parsed = cls.clean_and_parse_number(cell)
                        if parsed is not None:
                            val_b = parsed
                            break

        if val_a is None or val_b is None:
            return None

        plan = NumericPlan(
            operation=op,
            operands=[
                CellOperand(table_id=table_id, row=1, col=1, raw_text=str(val_a), parsed_value=float(val_a), label=entity_a),
                CellOperand(table_id=table_id, row=2, col=1, raw_text=str(val_b), parsed_value=float(val_b), label=entity_b),
            ],
            target_metric=metric_keyword or "reported score",
        )
        return cls.execute_plan(plan)
