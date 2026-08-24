import unittest
from backend.schemas.capabilities import CapabilityMode, ModelCapabilities, ModelRegistry
from backend.schemas.document import (
    BoundingBox,
    CoordinateTransform,
    DocumentMetadata,
    EvidenceBlock,
    PageRender,
    ScientificDocument,
    SectionNode,
    TableBlock,
    TableCell,
    VisualEvidence,
    VisualRegion,
)


class TestSchemasAndCapabilities(unittest.TestCase):

    def test_bounding_box_validation_and_ordering(self):
        # Even if coordinates are inverted (x1 < x0), model_validator normalizes them
        box = BoundingBox(x0=0.8, y0=0.9, x1=0.2, y1=0.1, coordinate_space="normalized_page")
        self.assertEqual(box.x0, 0.2)
        self.assertEqual(box.y0, 0.1)
        self.assertEqual(box.x1, 0.8)
        self.assertEqual(box.y1, 0.9)
        self.assertAlmostEqual(box.width, 0.6)
        self.assertAlmostEqual(box.height, 0.8)
        self.assertAlmostEqual(box.area, 0.48)
        self.assertTrue(box.is_valid())

    def test_bounding_box_coordinate_conversions(self):
        # 600 x 800 pt page
        norm_box = BoundingBox(x0=0.1, y0=0.2, x1=0.5, y1=0.6, coordinate_space="normalized_page")
        pdf_box = norm_box.to_pdf_points(page_width=600.0, page_height=800.0)
        
        self.assertEqual(pdf_box.coordinate_space, "pdf_points")
        self.assertEqual(pdf_box.x0, 60.0)
        self.assertEqual(pdf_box.y0, 160.0)
        self.assertEqual(pdf_box.x1, 300.0)
        self.assertEqual(pdf_box.y1, 480.0)

        # Convert back
        roundtrip = pdf_box.to_normalized(page_width=600.0, page_height=800.0)
        self.assertEqual(roundtrip.coordinate_space, "normalized_page")
        self.assertAlmostEqual(roundtrip.x0, 0.1)
        self.assertAlmostEqual(roundtrip.y0, 0.2)
        self.assertAlmostEqual(roundtrip.x1, 0.5)
        self.assertAlmostEqual(roundtrip.y1, 0.6)

    def test_bounding_box_iou_and_containment(self):
        box_a = BoundingBox(x0=0.0, y0=0.0, x1=0.5, y1=0.5)
        box_b = BoundingBox(x0=0.25, y0=0.0, x1=0.75, y1=0.5)
        
        # Inter = 0.25 x 0.5 = 0.125
        # Area A = 0.25, Area B = 0.25
        # Union = 0.25 + 0.25 - 0.125 = 0.375
        # IoU = 0.125 / 0.375 = 1/3
        self.assertAlmostEqual(box_a.iou(box_b), 1.0 / 3.0)

        # Disjoint boxes have IoU = 0
        box_c = BoundingBox(x0=0.8, y0=0.8, x1=1.0, y1=1.0)
        self.assertEqual(box_a.iou(box_c), 0.0)

        # Containment
        inner = BoundingBox(x0=0.1, y0=0.1, x1=0.4, y1=0.4)
        self.assertTrue(box_a.contains(inner))
        self.assertFalse(inner.contains(box_a))

    def test_coordinate_transform_affine_mapping(self):
        # Scale by 2, translate by x=+10, y=+20
        # x' = 2*x + 10, y' = 2*y + 20
        transform = CoordinateTransform(
            source_space="pdf_points",
            target_space="render_pixels",
            matrix=[2.0, 0.0, 0.0, 2.0, 10.0, 20.0],
        )
        x_p, y_p = transform.apply(50.0, 100.0)
        self.assertEqual(x_p, 110.0)
        self.assertEqual(y_p, 220.0)

        # Inverse transform
        inv = transform.inverse()
        orig_x, orig_y = inv.apply(x_p, y_p)
        self.assertAlmostEqual(orig_x, 50.0)
        self.assertAlmostEqual(orig_y, 100.0)

    def test_model_capabilities_gating(self):
        # Vision model in AUTO mode
        vlm = ModelRegistry.resolve_capabilities("qwen3.5:9b", mode=CapabilityMode.AUTO)
        self.assertTrue(vlm.supports_vision)
        self.assertTrue(vlm.can_process_images())
        self.assertEqual(vlm.effective_image_limit(), 4)

        # Forcing TEXT_ONLY mode disables image processing
        vlm_text_mode = ModelRegistry.resolve_capabilities("qwen3.5:9b", mode=CapabilityMode.TEXT_ONLY)
        self.assertTrue(vlm_text_mode.supports_vision)
        self.assertFalse(vlm_text_mode.can_process_images())
        self.assertEqual(vlm_text_mode.effective_image_limit(), 0)

        # Pure text model
        text_lm = ModelRegistry.resolve_capabilities("llama3.1:8b", mode=CapabilityMode.AUTO)
        self.assertFalse(text_lm.supports_vision)
        self.assertFalse(text_lm.can_process_images())
        self.assertEqual(text_lm.effective_image_limit(), 0)

    def test_scientific_document_serialization(self):
        meta = DocumentMetadata(
            document_id="doc_12345",
            source_sha256="abcdef123456",
            filename="sample_paper.pdf",
            page_count=10,
            title="Attention Is All You Need",
        )
        block = EvidenceBlock(
            evidence_id="E_001",
            document_id="doc_12345",
            block_type="paragraph",
            original_text="The Transformer is the first transduction model relying entirely on self-attention.",
            retrieval_text="Introduction. The Transformer is the first transduction model relying entirely on self-attention.",
            page=1,
            section_path=["1 Introduction"],
            bbox=BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.3),
        )
        doc = ScientificDocument(
            metadata=meta,
            sections=[
                SectionNode(
                    section_id="S_01",
                    title="1 Introduction",
                    section_path=["1 Introduction"],
                    evidence_ids=["E_001"],
                )
            ],
            evidence_blocks=[block],
        )
        # Serialization & roundtrip
        json_data = doc.model_dump_json()
        restored = ScientificDocument.model_validate_json(json_data)
        self.assertEqual(restored.metadata.document_id, "doc_12345")
        self.assertEqual(len(restored.evidence_blocks), 1)
        self.assertEqual(restored.evidence_blocks[0].evidence_id, "E_001")
        self.assertEqual(restored.evidence_blocks[0].bbox.x0, 0.1)


if __name__ == "__main__":
    unittest.main()
