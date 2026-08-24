"""Unit tests for ExportService: Markdown & LaTeX Reasoning Reports."""

import unittest
from backend.schemas.claims import AtomicClaim, EntailmentStatus, VerificationReport
from backend.schemas.evidence_graph import ReasoningPath, ReasoningPathStep
from backend.schemas.numeric_plan import NumericExecutionResult, NumericOp
from backend.schemas.reasoning import QuestionAnalysis, ReasoningLevel, TargetModality
from backend.services.export_service import ExportService


class TestExportService(unittest.TestCase):

    def setUp(self):
        self.analysis = QuestionAnalysis(
            original_query="Why does the Transformer outperform ConvS2S based on architecture and results?",
            reasoning_level=ReasoningLevel.L5_MULTI_HOP_SYNTHESIS,
            target_modalities=[TargetModality.TEXT, TargetModality.TABLE],
            requires_arithmetic=True,
        )

        self.path = ReasoningPath(
            query=self.analysis.original_query,
            reasoning_level=self.analysis.reasoning_level.value,
            steps=[
                ReasoningPathStep(step_index=1, evidence_id="E_001", section="Model Architecture", page=3, modality="text", role="method_definition", claim_contribution="Defines multi-head attention."),
                ReasoningPathStep(step_index=2, evidence_id="E_002", section="Ablation Studies", page=7, modality="text", role="ablation_support", claim_contribution="Isolates attention head impact."),
                ReasoningPathStep(step_index=3, evidence_id="E_TAB_01", section="Translation Results", page=8, modality="table", role="final_result", claim_contribution="Reports 28.4 BLEU."),
            ],
        )

        self.numeric_res = NumericExecutionResult(
            operation=NumericOp.DIFFERENCE,
            computed_value=3.24,
            formatted_value="+3.24",
            formatted_statement="The difference in BLEU score is +3.24 (Transformer: 28.4 vs ConvS2S: 25.16).",
            is_exact=True,
        )

        self.verification = VerificationReport(
            claims=[
                AtomicClaim(claim_id="C1", text="The Transformer achieves 28.4 BLEU.", entailment_status=EntailmentStatus.SUPPORTED),
                AtomicClaim(claim_id="C2", text="Multi-head attention eliminates recurrence.", entailment_status=EntailmentStatus.SUPPORTED),
            ],
            overall_supported=True,
            supported_count=2,
            unsupported_count=0,
            contradicted_count=0,
        )

    def test_export_markdown(self):
        """Verify Markdown output contains all structured reasoning sections."""
        md = ExportService.export_to_markdown(
            paper_id="attention_vaswani_2017",
            query=self.analysis.original_query,
            answer="The Transformer outperforms ConvS2S [E_001]...",
            analysis=self.analysis,
            path=self.path,
            numeric_res=self.numeric_res,
            verification=self.verification,
        )

        self.assertIn("# ScholAR Multi-Level Reasoning Report", md)
        self.assertIn("L5_MULTI_HOP_SYNTHESIS", md)
        self.assertIn("Deterministic Tabular Arithmetic Proof", md)
        self.assertIn("+3.24", md)
        self.assertIn("Evidence Reasoning Path (DAG)", md)
        self.assertIn("3-Way Atomic Claim Entailment Audit", md)

    def test_export_latex(self):
        """Verify LaTeX output contains TikZ figure and document structure."""
        tex = ExportService.export_to_latex(
            paper_id="attention_vaswani_2017",
            query=self.analysis.original_query,
            answer="The Transformer outperforms ConvS2S [E_001]...",
            analysis=self.analysis,
            path=self.path,
            numeric_res=self.numeric_res,
            verification=self.verification,
        )

        self.assertIn("\\documentclass{article}", tex)
        self.assertIn("\\usepackage{tikz}", tex)
        self.assertIn("\\begin{tikzpicture}", tex)
        self.assertIn("ScholAR Multi-Level Reasoning Report", tex)


if __name__ == "__main__":
    unittest.main()
