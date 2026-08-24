import unittest
from pathlib import Path
from backend.services.chunking_service import _extract_headings, _section_path, chunk_pages, chunk_figures
from backend.services.storage_service import StorageService
from backend.schemas.document import ScientificDocument


class TestStorageAndAst(unittest.TestCase):

    def test_section_path_hierarchy_extraction(self):
        sample_page_text = """
        3 Model Architecture
        3.1 Encoder and Decoder Stacks
        The encoder is composed of a stack of N = 6 identical layers.
        3.2 Attention Mechanism
        An attention function can be described as mapping a query and a set of key-value pairs to an output.
        """
        headings = _extract_headings(sample_page_text)
        self.assertEqual(len(headings), 3)
        self.assertEqual(headings[0], ("3 Model Architecture", 1))
        self.assertEqual(headings[1], ("3.1 Encoder And Decoder Stacks", 2))
        self.assertEqual(headings[2], ("3.2 Attention Mechanism", 2))

        path = _section_path(sample_page_text, page_number=3)
        self.assertIn("3 Model Architecture", path[0])
        self.assertIn("3.2 Attention Mechanism", path[1])

    def test_chunk_pages_retrieval_text_prefixing(self):
        pages = [
            {
                "page": 1,
                "text": "Abstract\nWe propose the Transformer, a model architecture eschewing recurrence.",
            },
            {
                "page": 3,
                "text": "3 Model Architecture\n3.2 Attention\nMulti-head attention allows the model to jointly attend.",
            },
        ]
        chunks = chunk_pages(pages, target_words=100)
        self.assertEqual(len(chunks), 2)
        
        # Check abstract chunk
        self.assertEqual(chunks[0]["page"], 1)
        self.assertEqual(chunks[0]["section_path"], ["Abstract"])
        self.assertTrue(chunks[0]["retrieval_text"].startswith("Abstract."))

        # Check section chunk
        self.assertEqual(chunks[1]["page"], 3)
        self.assertIn("3 Model Architecture", chunks[1]["section_path"][0])
        self.assertTrue(" > " in chunks[1]["retrieval_text"] or "3 Model Architecture" in chunks[1]["retrieval_text"])

    def test_storage_service_sync_and_query(self):
        paper_id = "test_paper_01"
        metadata = {
            "title": "Attention Is All You Need",
            "authors": ["Ashish Vaswani", "Noam Shazeer"],
            "year": "2017",
            "summary": "Transformer model paper",
            "pages": 11,
            "source": "arxiv",
        }
        chunks = [
            {
                "chunk_id": "chunk_001",
                "page": 1,
                "section_title": "Abstract",
                "section_path": ["Abstract"],
                "chunk_type": "abstract",
                "text": "We introduce the Transformer architecture.",
                "retrieval_text": "Abstract. We introduce the Transformer architecture.",
                "paragraph_text": "We introduce the Transformer architecture.",
            },
            {
                "chunk_id": "chunk_002",
                "page": 3,
                "section_title": "3.2 Attention",
                "section_path": ["3 Model Architecture", "3.2 Attention"],
                "chunk_type": "method",
                "text": "Multi-head attention details.",
                "retrieval_text": "3 Model Architecture > 3.2 Attention. Multi-head attention details.",
                "paragraph_text": "Multi-head attention details.",
            },
        ]
        figures = [
            {
                "figure_id": "fig_01_001",
                "page": 3,
                "label": "Figure 1",
                "figure_type": "figure",
                "caption": "The Transformer model architecture.",
                "image_file": "fig_01_001.png",
                "bbox": {"x0": 0.1, "y0": 0.2, "x1": 0.9, "y1": 0.8},
            }
        ]

        # Sync to DB
        StorageService.sync_paper_to_db(paper_id, metadata, chunks, figures)

        # Query via SQL
        res = StorageService.query_sql(paper_id, "SELECT * FROM chunks WHERE page = ?", (3,))
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["chunk_id"], "chunk_002")
        self.assertEqual(res[0]["section_title"], "3.2 Attention")

        fig_res = StorageService.query_sql(paper_id, "SELECT * FROM figures WHERE label = ?", ("Figure 1",))
        self.assertEqual(len(fig_res), 1)
        self.assertEqual(fig_res[0]["figure_id"], "fig_01_001")


if __name__ == "__main__":
    unittest.main()
