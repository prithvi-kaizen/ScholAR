"""Portable source-contract test for the ten-paper Testing catalog.

Paper acquisition is deliberately not a unit-test prerequisite; model/data-backed
availability is checked by explicit evaluation preflights instead.
"""

import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class TestTestingTabPapers(unittest.TestCase):
    def test_testing_catalog_declares_all_ten_paper_ids_without_requiring_cache(self) -> None:
        requested_paper_ids = [
            "1406.2661",
            "1412.6980",
            "2112.10752",
            "1706.03762",
            "2406.08394",
            "2104.08663",
            "2603.14257",
            "2025.emnlp-main.77",
            "yale_thesis_1003",
            "2410.00526",
        ]
        source = (ROOT / "frontend" / "app" / "page.tsx").read_text(encoding="utf-8")
        testing_start = source.index("const testingPapers")
        testing_end = source.index("export default function", testing_start)
        testing_catalog = source[testing_start:testing_end]
        for paper_id in requested_paper_ids:
            with self.subTest(paper_id=paper_id):
                self.assertEqual(
                    testing_catalog.count(f'id: "{paper_id}"'),
                    1,
                    f"Testing catalog must declare {paper_id} exactly once",
                )
        self.assertEqual(testing_catalog.count("  {\n    id:"), 10)


if __name__ == "__main__":
    unittest.main()
