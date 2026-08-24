import unittest
from backend.services.verifier_service import (
    ClaimVerifierService,
    VerificationLabel,
    ClaimVerificationResult,
)


class TestVerifierAndRepair(unittest.TestCase):

    def test_claim_decomposition(self):
        answer = """
        **Answer**
        The Transformer model uses self-attention mechanisms [1]. It was evaluated on WMT 2014 translation [2].
        """
        claims = ClaimVerifierService.decompose_answer_into_claims(answer)
        self.assertEqual(len(claims), 2)
        self.assertIn("uses self-attention mechanisms", claims[0][0])
        self.assertEqual(claims[0][1], [1])
        self.assertIn("evaluated on WMT 2014 translation", claims[1][0])
        self.assertEqual(claims[1][1], [2])

    def test_claim_verification_categories(self):
        evidence = ["The Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."]

        # 1. Supported
        res_sup = ClaimVerifierService.verify_claim(
            claim_id="C1",
            claim_text="The Transformer achieves 28.4 BLEU on English-to-German translation.",
            evidence_texts=evidence,
        )
        self.assertEqual(res_sup.label, VerificationLabel.SUPPORTED)
        self.assertGreaterEqual(res_sup.confidence, 0.8)

        # 2. Contradicted (number mismatch)
        res_contra = ClaimVerifierService.verify_claim(
            claim_id="C2",
            claim_text="The Transformer achieves 99.5 BLEU on English-to-German translation.",
            evidence_texts=evidence,
        )
        self.assertEqual(res_contra.label, VerificationLabel.CONTRADICTED)

        # 3. Partially supported (1 out of 3 tokens overlap = 33%)
        res_part = ClaimVerifierService.verify_claim(
            claim_id="C3",
            claim_text="General optimization task.",
            evidence_texts=evidence,
        )
        self.assertEqual(res_part.label, VerificationLabel.PARTIALLY_SUPPORTED)

        # 4. Unsupported
        res_unsup = ClaimVerifierService.verify_claim(
            claim_id="C4",
            claim_text="Photosynthesis produces glucose from sunlight and carbon dioxide.",
            evidence_texts=evidence,
        )
        self.assertEqual(res_unsup.label, VerificationLabel.UNSUPPORTED)

    def test_single_repair_policy(self):
        evidence = ["The Transformer uses scaled dot-product attention."]
        partial_res = ClaimVerificationResult(
            claim_id="C_part",
            claim_text="Dot-product attention framework.",
            label=VerificationLabel.PARTIALLY_SUPPORTED,
            confidence=0.4,
        )
        repaired = ClaimVerifierService.apply_single_repair(partial_res, evidence)
        self.assertEqual(repaired.label, VerificationLabel.SUPPORTED)
        self.assertIn("According to the paper evidence", repaired.repaired_text or "")

    def test_verify_and_repair_answer_end_to_end(self):
        answer = "The Transformer achieves 28.4 BLEU on WMT 2014 [1]."
        citations = [
            {
                "ref_id": 1,
                "page": 8,
                "quote": "The Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task.",
            }
        ]
        ans, verified_cits, verified_claims = ClaimVerifierService.verify_and_repair_answer(
            answer=answer,
            citations=citations,
            apply_repair=True,
        )
        self.assertEqual(len(verified_claims), 1)
        self.assertEqual(verified_claims[0].label, VerificationLabel.SUPPORTED)
        self.assertEqual(len(verified_cits), 1)
        self.assertEqual(verified_cits[0]["verification"], "SUPPORTED")
        self.assertGreaterEqual(verified_cits[0]["confidence"], 0.8)


if __name__ == "__main__":
    unittest.main()
