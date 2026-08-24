"""Unit tests for Phase 7: Closed-Loop Cross-Modal Grounding & Visual Anchor Verifier."""

import unittest
from backend.schemas.visual_grounding import VisualAnchor, VisualInspectionReport
from backend.services.visual_grounding_service import VisualGroundingService


class TestVisualGrounding(unittest.TestCase):

    def test_visual_anchor_model(self):
        """Verify VisualAnchor bounding box normalization and field mapping."""
        anchor = VisualAnchor(
            figure_id="VIS_F2",
            panel_label="(a)",
            bbox_norm=[0.1, 0.2, 0.5, 0.6],
            caption="Figure 2: Loss curves comparing Adam vs SGD",
            image_crop_path="/tmp/crop.png",
        )
        self.assertEqual(anchor.figure_id, "VIS_F2")
        self.assertEqual(anchor.panel_label, "(a)")
        self.assertEqual(len(anchor.bbox_norm), 4)

    def test_verify_visual_claim(self):
        """Verify closed-loop visual claim verification report."""
        anchor = VisualAnchor(
            figure_id="VIS_F2",
            caption="Figure 2: Training loss curves for Adam and SGD on MNIST dataset.",
        )
        report = VisualGroundingService.verify_visual_claim(
            claim_text="Figure 2 shows that Adam converges faster than SGD on training loss.",
            anchor=anchor,
        )
        self.assertTrue(report.is_visually_grounded)
        self.assertGreaterEqual(report.confidence, 0.8)


if __name__ == "__main__":
    unittest.main()
