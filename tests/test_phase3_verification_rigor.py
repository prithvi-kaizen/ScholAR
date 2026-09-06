"""Unit tests for Phase 3: Verification & Grounding Rigor."""

import unittest
from backend.services.verifier_service import LexicalSupportScorer, VerificationLabel


class TestPhase3VerificationRigor(unittest.TestCase):

    def setUp(self):
        self.scorer = LexicalSupportScorer()

    def test_polar_negation_contradiction(self):
        """Negation asymmetry between claim and evidence with high content overlap must yield CONTRADICTED."""
        claim = "Residual learning does not improve training accuracy."
        evidence = "Residual learning improves training accuracy."

        result = self.scorer.score("claim_neg", claim, [evidence], "text", ["ev_1"])
        self.assertEqual(result.label, VerificationLabel.CONTRADICTED)
        self.assertIn("polar negation", result.reason.lower())

    def test_numerical_mismatch_contradiction(self):
        """Claim asserting unsupported numbers alongside matching numbers must yield CONTRADICTED."""
        claim = "Model A scores 80 and Model B scores 999."
        evidence = "Model A scores 80 and Model B scores 75."

        result = self.scorer.score("claim_num", claim, [evidence], "text", ["ev_1"])
        self.assertEqual(result.label, VerificationLabel.CONTRADICTED)
        self.assertIn("numerical contradiction", result.reason.lower())

    def test_supported_claim_with_matching_numbers_and_polarity(self):
        """Claim with matching numbers and polarity must yield SUPPORTED."""
        claim = "Model A achieves 80% accuracy and Model B achieves 75% accuracy."
        evidence = "Our results demonstrate Model A achieves 80% accuracy, whereas Model B achieves 75% accuracy."

        result = self.scorer.score("claim_supp", claim, [evidence], "text", ["ev_1"])
        self.assertEqual(result.label, VerificationLabel.SUPPORTED)

    def test_unsupported_claim_low_overlap(self):
        """Claim with low overlap must yield UNSUPPORTED."""
        claim = "The universe expanded exponentially during the inflationary epoch."
        evidence = "Residual networks ease the training of substantially deeper neural networks."

        result = self.scorer.score("claim_unsupp", claim, [evidence], "text", ["ev_1"])
        self.assertEqual(result.label, VerificationLabel.UNSUPPORTED)


if __name__ == "__main__":
    unittest.main()
