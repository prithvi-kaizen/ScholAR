"""Unit tests for Phase 1: Dual-Engine Ingestion Pipeline & Canonical Evidence AST."""

import json
import tempfile
import unittest
from pathlib import Path

import fitz

from backend.schemas.evidence import (
    EvidenceAST,
    EvidenceBlock,
    EvidenceModality,
    ParserAblationConfig,
    SectionNode,
    TableCell,
    TableData,
)
from backend.services.ingestion_service import (
    DualEngineIngestionService,
    PARSER_ABLATIONS,
)


class TestDualEngineIngestion(unittest.TestCase):

    def test_evidence_ast_schema_and_methods(self):
        """Verify EvidenceAST serialization, lookup, and filtering."""
        block1 = EvidenceBlock(
            evidence_id="E_001",
            document_id="test_doc",
            page=1,
            section_path=["1 Introduction"],
            modality=EvidenceModality.TEXT,
            bbox=[0.1, 0.1, 0.9, 0.3],
            text="This is an introduction paragraph.",
        )
        block2 = EvidenceBlock(
            evidence_id="E_TAB_01",
            document_id="test_doc",
            page=2,
            section_path=["2 Experiments"],
            modality=EvidenceModality.TABLE,
            bbox=[0.1, 0.4, 0.9, 0.8],
            text="| Model | BLEU |\n|---|---|\n| Transformer | 28.4 |",
            table_data=TableData(
                table_id="tab_001",
                headers=["Model", "BLEU"],
                rows=[["Transformer", "28.4"]],
                cells=[
                    TableCell(row=0, col=0, value="Model", is_header=True),
                    TableCell(row=0, col=1, value="BLEU", is_header=True),
                    TableCell(row=1, col=0, value="Transformer"),
                    TableCell(row=1, col=1, value="28.4"),
                ],
                num_rows=2,
                num_cols=2,
            ),
        )
        block3 = EvidenceBlock(
            evidence_id="VIS_F1",
            document_id="test_doc",
            page=3,
            section_path=["3 Results"],
            modality=EvidenceModality.VISUAL,
            bbox=[0.1, 0.2, 0.9, 0.7],
            text="Figure 1: Attention weight visualization",
            figure_id="fig_001",
        )

        ast = EvidenceAST(
            document_id="test_doc",
            title="Test Scientific Paper",
            authors=["Alice", "Bob"],
            year=2026,
            page_count=3,
            blocks=[block1, block2, block3],
            sections=[
                SectionNode(title="1 Introduction", level=1, page_start=1, page_end=1),
                SectionNode(title="2 Experiments", level=1, page_start=2, page_end=2),
                SectionNode(title="3 Results", level=1, page_start=3, page_end=3),
            ],
            parser_engine="docling+pymupdf",
            degraded_mode=False,
        )

        # Test Lookups
        self.assertEqual(ast.get_block("E_001"), block1)
        self.assertEqual(ast.get_block("VIS_F1"), block3)
        self.assertIsNone(ast.get_block("NON_EXISTENT"))

        # Test filtering by modality
        self.assertEqual(len(ast.get_text_blocks()), 1)
        self.assertEqual(len(ast.get_table_blocks()), 1)
        self.assertEqual(len(ast.get_visual_blocks()), 1)

        # Test page filtering
        self.assertEqual(len(ast.get_blocks_for_page(2)), 1)
        self.assertEqual(ast.get_blocks_for_page(2)[0].modality, EvidenceModality.TABLE)

    def test_parser_ablation_matrix_configs(self):
        """Verify that all 5 parser ablation configurations are defined and valid."""
        expected_configs = {"P0", "P1", "P2", "P3", "P4"}
        self.assertEqual(set(PARSER_ABLATIONS.keys()), expected_configs)

        for cfg_id, cfg in PARSER_ABLATIONS.items():
            self.assertIsInstance(cfg, ParserAblationConfig)
            self.assertEqual(cfg.config_id, cfg_id)
            self.assertTrue(len(cfg.parser_name) > 0)
            self.assertTrue(cfg.chunk_token_size > 0)
            self.assertTrue(cfg.chunk_overlap > 0)

    def test_dual_engine_synthetic_pdf_ingestion(self):
        """Create a synthetic multi-section PDF, run ingestion, and verify EvidenceAST outputs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample_paper.pdf"
            doc = fitz.open()

            # Page 1: Title & Section 1
            page1 = doc.new_page(width=600, height=800)
            page1.insert_text((50, 80), "ScholAR: A Capability-Adaptive Scientific Document Assistant", fontsize=16)
            page1.insert_text((50, 130), "1. Introduction", fontsize=14)
            page1.insert_text((50, 160), "Scientific document understanding requires precise grounding across text and figures.", fontsize=11)

            # Page 2: Section 2 & Table
            page2 = doc.new_page(width=600, height=800)
            page2.insert_text((50, 80), "2. Experimental Methodology", fontsize=14)
            page2.insert_text((50, 110), "We evaluate across multiple local hardware profiles.", fontsize=11)
            page2.insert_text((50, 150), "| System | Latency | Faithfulness |\n| ScholAR | 1.2s | 0.88 |", fontsize=10)

            doc.save(str(pdf_path))
            doc.close()

            # Execute Ingestion
            ast = DualEngineIngestionService.ingest_paper(
                pdf_path=pdf_path,
                document_id="synthetic_test_01",
                title="ScholAR Synthetic Test",
                authors=["Test Author"],
                year=2026,
            )

            # Verifications
            self.assertEqual(ast.document_id, "synthetic_test_01")
            self.assertEqual(ast.page_count, 2)
            self.assertTrue(len(ast.blocks) > 0)

            # Verify coordinate normalization in [0.0, 1.0]
            for block in ast.blocks:
                self.assertEqual(len(block.bbox), 4)
                for coord in block.bbox:
                    self.assertGreaterEqual(coord, 0.0)
                    self.assertLessEqual(coord, 1.0)

            # Verify generated files exist in local paper storage
            from backend.services.pdf_service import paper_dir
            p_dir = paper_dir("synthetic_test_01")
            self.assertTrue((p_dir / "evidence_ast.json").exists())
            self.assertTrue((p_dir / "chunks.json").exists())
            self.assertTrue((p_dir / "pages.json").exists())
            self.assertTrue((p_dir / "metadata.json").exists())

            # Read back metadata and check fields
            meta = json.loads((p_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["id"], "synthetic_test_01")
            self.assertIn("parser_engine", meta)
            self.assertIn("degraded_mode", meta)


if __name__ == "__main__":
    unittest.main()
