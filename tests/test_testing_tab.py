"""Unit tests to verify that the Testing tab papers match the requested 10 benchmark papers."""

import unittest
from pathlib import Path
from backend.services.pdf_service import paper_dir


class TestTestingTabPapers(unittest.TestCase):

    def test_all_ten_testing_papers_exist_locally(self):
        """Verify all 10 requested benchmark papers are present on disk in backend/data/papers."""
        requested_paper_ids = [
            "1406.2661",          # GAN
            "1412.6980",          # Adam
            "2112.10752",         # Stable Diffusion
            "1706.03762",         # Attention Is All You Need
            "2406.08394",         # VisionLLM v2
            "2104.08663",         # BEIR
            "2603.14257",         # Inter-doc Multi-hop Scientific QA
            "2025.emnlp-main.77", # MEBench Cross-Doc Multi-Entity QA
            "yale_thesis_1003",   # Towards Multimodal Multi-Doc / M3SciQA
            "2410.00526",         # Conversational QA in Multi-instructional Docs
        ]

        for pid in requested_paper_ids:
            p_dir = paper_dir(pid)
            self.assertTrue(
                p_dir.exists(),
                f"Paper directory for [{pid}] must exist in backend/data/papers",
            )
            self.assertTrue(
                (p_dir / "paper.pdf").exists() or (p_dir / "pages.json").exists(),
                f"Paper [{pid}] must have either paper.pdf or pages.json",
            )
            self.assertTrue(
                (p_dir / "chunks.json").exists(),
                f"Paper [{pid}] must have chunks.json",
            )
            self.assertTrue(
                (p_dir / "metadata.json").exists(),
                f"Paper [{pid}] must have metadata.json",
            )


if __name__ == "__main__":
    unittest.main()
