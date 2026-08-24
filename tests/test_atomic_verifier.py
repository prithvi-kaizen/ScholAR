"""Unit tests for Phase 8: Atomic Claim Verification, 3-Way Entailment & 1-Pass Repair."""

import unittest
from backend.schemas.claims import (
    AtomicClaim,
    EntailmentStatus,
    RepairAction,
    VerificationReport,
)
from backend.services.verifier_service import ClaimVerifierService


class TestAtomicVerifier(unittest.TestCase):

    def setUp(self):
        self.sample_chunks = [
            {
                "chunk_id": "c1",
                "evidence_id": "E_001",
                "text": "The Transformer achieves 28.4 BLEU on the English-to-German machine translation task.",
            },
            {
                "chunk_id": "c2",
                "evidence_id": "E_002",
                "text": "Training took 3.5 days on 8 P100 GPUs.",
            },
        ]

    def test_supported_claim_verification(self):
        """Verify factual claim is marked SUPPORTED."""
        response = "The Transformer achieves 28.4 BLEU on English-to-German translation [E_001]."
        report = ClaimVerifierService.generate_atomic_verification_report(
            response_text=response,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertTrue(report.overall_supported)
        self.assertEqual(report.supported_count, 1)
        self.assertEqual(report.claims[0].entailment_status, EntailmentStatus.SUPPORTED)

    def test_contradicted_number_verification(self):
        """Verify claim with wrong numerical statistic is marked CONTRADICTED."""
        response = "The Transformer achieves 99.4 BLEU on the translation task [E_001]."
        report = ClaimVerifierService.generate_atomic_verification_report(
            response_text=response,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertEqual(report.contradicted_count, 1)
        self.assertEqual(report.claims[0].entailment_status, EntailmentStatus.CONTRADICTED)

    def test_calibrated_abstention_on_unsupported_response(self):
        """Verify that a response with zero supporting evidence cleanly abstains."""
        response = "This paper proposes a quantum teleportation algorithm for planetary navigation [E_001]."
        report = ClaimVerifierService.generate_atomic_verification_report(
            response_text=response,
            retrieved_chunks=self.sample_chunks,
        )
        self.assertTrue(report.has_abstained)
        self.assertIn("sufficient evidence", report.final_verified_response.lower())


if __name__ == "__main__":
    unittest.main()
