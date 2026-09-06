import re
import unittest

from backend.schemas.answer_trace import InterventionControls, RepairMode
from backend.schemas.claims import EntailmentStatus, RepairAction
from backend.services.verifier_service import (
    ClaimVerificationResult,
    ClaimVerifierService,
    VerificationLabel,
)


class TestVerifierAndRepair(unittest.TestCase):
    def setUp(self) -> None:
        self.attention_citation = {
            "ref_id": 1,
            "evidence_id": "E1",
            "chunk_id": "attention",
            "page": 2,
            "quote": "The Transformer uses scaled dot-product attention.",
        }

    def test_claim_parser_preserves_exact_claim_and_citation_spans(self) -> None:
        answer = (
            "## Answer\n\n"
            "- The method uses α-attention [1].\n"
            "It scores 28.4 BLEU [E_002].\n"
            "```python\nignored = True\n```\n"
        )
        claims = ClaimVerifierService.parse_answer_into_claims(answer)
        self.assertEqual(len(claims), 2)
        for claim in claims:
            self.assertEqual(answer[claim.start:claim.end], claim.text)
            for citation in claim.citations:
                self.assertEqual(answer[citation.start:citation.end], citation.marker)
        self.assertEqual(claims[0].text, "- The method uses α-attention [1].")
        self.assertEqual(claims[0].citations[0].reference_ids, ["1"])
        self.assertEqual(claims[1].citations[0].reference_ids, ["E_002"])

        legacy_claims = ClaimVerifierService.decompose_answer_into_claims(answer)
        self.assertEqual(legacy_claims[0][1], [1])
        self.assertEqual(legacy_claims[1][1], [])

    def test_claim_verification_uses_canonical_four_way_labels(self) -> None:
        evidence = ["The Transformer achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."]
        supported = ClaimVerifierService.verify_claim(
            "C1", "The Transformer achieves 28.4 BLEU on English-to-German translation.", evidence
        )
        contradicted = ClaimVerifierService.verify_claim(
            "C2", "The Transformer achieves 99.5 BLEU on English-to-German translation.", evidence
        )
        partial = ClaimVerifierService.verify_claim("C3", "General optimization task.", evidence)
        unsupported = ClaimVerifierService.verify_claim(
            "C4", "Photosynthesis produces glucose from sunlight and carbon dioxide.", evidence
        )
        self.assertEqual(supported.label, VerificationLabel.SUPPORTED)
        self.assertEqual(contradicted.label, VerificationLabel.CONTRADICTED)
        self.assertEqual(partial.label, VerificationLabel.PARTIAL)
        self.assertEqual(partial.label.value, "PARTIAL")
        self.assertEqual(VerificationLabel.PARTIALLY_SUPPORTED, VerificationLabel.PARTIAL)
        self.assertEqual(unsupported.label, VerificationLabel.UNSUPPORTED)

    def test_single_repair_deletes_unsupported_clause_and_reverifies(self) -> None:
        original = (
            "The Transformer uses scaled dot-product attention and requires quantum entanglement "
            "for planetary navigation across galaxies [1]."
        )
        first_pass = ClaimVerifierService.verify_claim(
            "C1", original, [self.attention_citation["quote"]]
        )
        self.assertEqual(first_pass.label, VerificationLabel.PARTIAL)
        repaired = ClaimVerifierService.apply_single_repair(first_pass, [self.attention_citation["quote"]])
        self.assertEqual(repaired.label, VerificationLabel.SUPPORTED)
        self.assertEqual(repaired.repair_action, RepairAction.CLAIM_NARROWING)
        self.assertEqual(repaired.repaired_text, "The Transformer uses scaled dot-product attention [1].")
        self.assertNotIn("According to the paper", repaired.repaired_text or "")

    def test_supported_claim_and_formatting_are_byte_preserved(self) -> None:
        answer = "**Result**\nThe Transformer uses scaled dot-product attention [1].\n"
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [self.attention_citation], [self.attention_citation]
        )
        self.assertEqual(outcome.final_answer.encode(), answer.encode())
        self.assertEqual(outcome.edits, [])
        self.assertTrue(outcome.reverified)
        self.assertEqual(outcome.final_report.supported_count, 1)
        claim = outcome.final_report.claims[0]
        self.assertEqual(outcome.final_answer[claim.start:claim.end], claim.text)
        self.assertEqual(claim.second_pass_status, EntailmentStatus.SUPPORTED)

    def test_no_repair_and_remap_only_are_behaviorally_distinct(self) -> None:
        answer = "BERT pre-trains deep bidirectional representations [2]."
        wrong = {"ref_id": 2, "chunk_id": "c2", "quote": "We use a WordPiece vocabulary."}
        supporting = {
            "ref_id": 1,
            "chunk_id": "c1",
            "quote": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        }
        none = ClaimVerifierService.verify_and_repair_detailed(
            answer,
            [wrong],
            [supporting, wrong],
            controls=InterventionControls(
                repair_mode=RepairMode.NONE,
                abstain_on_no_supported_claims=False,
            ),
        )
        remap = ClaimVerifierService.verify_and_repair_detailed(
            answer,
            [wrong],
            [supporting, wrong],
            controls=InterventionControls(
                repair_mode=RepairMode.CITATION_REMAP_ONLY,
                abstain_on_no_supported_claims=False,
            ),
        )
        self.assertEqual(none.final_answer, answer)
        self.assertEqual(none.edits, [])
        self.assertEqual(remap.final_answer, "BERT pre-trains deep bidirectional representations [1].")
        self.assertEqual(remap.edits[0].action, RepairAction.CITATION_REMAP)

    def test_partial_claim_is_narrowed_without_new_factual_tokens(self) -> None:
        answer = (
            "## Result\n"
            "The Transformer uses scaled dot-product attention and requires quantum entanglement "
            "for planetary navigation across galaxies [1].\n"
        )
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [self.attention_citation], [self.attention_citation]
        )
        self.assertIn("The Transformer uses scaled dot-product attention [1].", outcome.final_answer)
        self.assertNotIn("quantum", outcome.final_answer)
        self.assertEqual(len(outcome.edits), 1)
        edit = outcome.edits[0]
        self.assertEqual(edit.action, RepairAction.CLAIM_NARROWING)
        self.assertEqual(edit.initial_status, EntailmentStatus.PARTIAL)
        self.assertEqual(edit.second_pass_status, EntailmentStatus.SUPPORTED)
        original_words = set(re.findall(r"[A-Za-z]+", edit.original_text.lower()))
        replacement_words = set(re.findall(r"[A-Za-z]+", edit.replacement_text.lower()))
        self.assertLessEqual(replacement_words, original_words)
        self.assertTrue(outcome.reverified)

    def test_unsupported_claim_is_deleted_when_a_supported_claim_remains(self) -> None:
        answer = (
            "The Transformer uses scaled dot-product attention [1].\n"
            "Photosynthesis powers interstellar quantum navigation [1]."
        )
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [self.attention_citation], [self.attention_citation]
        )
        self.assertIn("scaled dot-product attention [1]", outcome.final_answer)
        self.assertNotIn("Photosynthesis", outcome.final_answer)
        self.assertEqual(outcome.edits[0].action, RepairAction.CLAIM_DELETION)
        self.assertTrue(outcome.reverified)
        self.assertTrue(all(
            claim.entailment_status == EntailmentStatus.SUPPORTED
            for claim in outcome.final_report.claims
            if claim.claim_type == "factual"
        ))

    def test_unsupported_only_answer_abstains_instead_of_retaining_claim(self) -> None:
        answer = "Photosynthesis powers interstellar quantum navigation [1]."
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [self.attention_citation], [self.attention_citation]
        )
        self.assertTrue(outcome.final_report.has_abstained)
        self.assertEqual(
            outcome.final_answer,
            "The paper does not provide sufficient evidence to answer this question.",
        )
        self.assertEqual(outcome.edits[0].action, RepairAction.ABSTAIN)
        self.assertNotIn("Photosynthesis", outcome.final_answer)

    def test_contradicted_number_is_removed_not_hedged(self) -> None:
        citation = {
            "ref_id": 1,
            "evidence_id": "E1",
            "quote": "The Transformer achieves 28.4 BLEU on English-to-German translation.",
        }
        answer = (
            "The Transformer achieves 28.4 BLEU on English-to-German translation [1].\n"
            "The Transformer achieves 99.5 BLEU on English-to-German translation [1]."
        )
        outcome = ClaimVerifierService.verify_and_repair_detailed(answer, [citation], [citation])
        self.assertIn("28.4 BLEU", outcome.final_answer)
        self.assertNotIn("99.5", outcome.final_answer)
        contradicted_edit = next(
            edit for edit in outcome.edits if edit.initial_status == EntailmentStatus.CONTRADICTED
        )
        self.assertEqual(contradicted_edit.action, RepairAction.CLAIM_DELETION)
        self.assertNotIn("According to", outcome.final_answer)

    def test_unsupported_citation_is_remapped_once_and_reverified(self) -> None:
        answer = "BERT pre-trains deep bidirectional representations [2]."
        wrong = {"ref_id": 2, "chunk_id": "c2", "quote": "We use a WordPiece vocabulary."}
        supporting = {
            "ref_id": 1,
            "chunk_id": "c1",
            "quote": "BERT is designed to pre-train deep bidirectional representations from unlabeled text.",
        }
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [wrong], [supporting, wrong]
        )
        self.assertEqual(outcome.final_answer, "BERT pre-trains deep bidirectional representations [1].")
        self.assertEqual(len(outcome.edits), 1)
        self.assertEqual(outcome.edits[0].action, RepairAction.CITATION_REMAP)
        self.assertTrue(outcome.edits[0].remap_attempted)
        self.assertEqual(outcome.edits[0].second_pass_status, EntailmentStatus.SUPPORTED)
        self.assertEqual(outcome.citations[0]["repair_origin"], "REMAPPED")
        self.assertTrue(outcome.reverified)

    def test_every_recorded_edit_changes_text_and_final_claims_are_cited(self) -> None:
        answer = (
            "The Transformer uses scaled dot-product attention [1].\n"
            "An uncited quantum claim crosses galaxies."
        )
        outcome = ClaimVerifierService.verify_and_repair_detailed(
            answer, [self.attention_citation], [self.attention_citation]
        )
        self.assertEqual(bool(outcome.edits), outcome.final_answer != answer)
        self.assertTrue(all(edit.original_text != edit.replacement_text for edit in outcome.edits))
        for claim in outcome.final_report.claims:
            if claim.claim_type == "factual":
                self.assertEqual(claim.entailment_status, EntailmentStatus.SUPPORTED)
                self.assertTrue(claim.citation_spans)
                self.assertTrue(claim.cited_evidence_ids)
        self.assertTrue(outcome.final_report.second_pass_completed)
        self.assertEqual(outcome.final_report.final_verified_response, outcome.final_answer)


if __name__ == "__main__":
    unittest.main()
