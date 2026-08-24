"""Unit tests for Phase 6: Deterministic Tabular Reasoning & NumericPlan Engine."""

import unittest
from decimal import Decimal
from backend.schemas.numeric_plan import (
    CellOperand,
    NumericOp,
    NumericPlan,
)
from backend.services.table_arithmetic_service import TableArithmeticService


class TestTableArithmetic(unittest.TestCase):

    def test_clean_and_parse_numbers(self):
        """Verify parsing across complex scientific notation."""
        self.assertEqual(TableArithmeticService.clean_and_parse_number("84.7%"), Decimal("84.7"))
        self.assertEqual(TableArithmeticService.clean_and_parse_number("28.4 +/- 0.2"), Decimal("28.4"))
        self.assertEqual(TableArithmeticService.clean_and_parse_number("28.4 ± 0.15"), Decimal("28.4"))
        self.assertEqual(TableArithmeticService.clean_and_parse_number("1,234.5"), Decimal("1234.5"))
        self.assertEqual(TableArithmeticService.clean_and_parse_number("88.4*"), Decimal("88.4"))
        self.assertEqual(TableArithmeticService.clean_and_parse_number("3.3e18"), Decimal("3300000000000000000"))

    def test_difference_calculation(self):
        """Verify exact difference computation."""
        plan = NumericPlan(
            operation=NumericOp.DIFFERENCE,
            operands=[
                CellOperand(table_id="tab_1", row=1, col=2, parsed_value=28.4, label="Transformer (big)"),
                CellOperand(table_id="tab_1", row=2, col=2, parsed_value=25.16, label="ConvS2S"),
            ],
            target_metric="BLEU score",
        )
        res = TableArithmeticService.execute_plan(plan)
        self.assertAlmostEqual(res.computed_value, 3.24, places=2)
        self.assertEqual(res.formatted_value, "+3.24")
        self.assertIn("3.24", res.formatted_statement)
        self.assertTrue(res.is_exact)

    def test_percent_change_calculation(self):
        """Verify exact percentage change computation."""
        # 84.7 vs 81.4 -> (84.7 - 81.4) / 81.4 * 100 = 4.054%
        plan = NumericPlan(
            operation=NumericOp.PERCENT_CHANGE,
            operands=[
                CellOperand(table_id="tab_1", row=1, col=2, parsed_value=84.7, label="Proposed"),
                CellOperand(table_id="tab_1", row=2, col=2, parsed_value=81.4, label="Baseline"),
            ],
            target_metric="accuracy",
        )
        res = TableArithmeticService.execute_plan(plan)
        self.assertAlmostEqual(res.computed_value, 4.054, places=3)
        self.assertEqual(res.formatted_value, "+4.054%")
        self.assertIn("+4.054%", res.formatted_statement)

    def test_extract_and_calculate_from_table_text(self):
        """Verify automated table cell extraction and calculation."""
        table_md = """
| Model | Accuracy (%) | Parameters |
| ResNet-50 | 76.15 | 25.6M |
| ResNet-152 | 78.31 | 60.2M |
| ViT-B/16 | 77.91 | 86.6M |
"""
        res = TableArithmeticService.extract_and_calculate_from_table_text(
            table_text=table_md,
            entity_a="ResNet-152",
            entity_b="ResNet-50",
            op=NumericOp.DIFFERENCE,
        )
        self.assertIsNotNone(res)
        # 78.31 - 76.15 = 2.16
        self.assertAlmostEqual(res.computed_value, 2.16, places=2)


if __name__ == "__main__":
    unittest.main()
