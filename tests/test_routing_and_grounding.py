import unittest
from backend.schemas.capabilities import CapabilityMode, ModelCapabilities
from backend.schemas.document import BoundingBox
from backend.services.routing_service import QuestionRouter, QuestionRouteType, QueryDecomposer
from backend.services.grounding_service import GroundingProposal, VisualGroundingService, VisualGroundingValidator
from backend.services.verifier_service import ClaimVerifierService, VerificationLabel


class TestRoutingAndGrounding(unittest.TestCase):

    def test_question_router_classification(self):
        # Visual cues
        self.assertEqual(
            QuestionRouter.classify_question("What does Figure 3 show about attention heads?"),
            QuestionRouteType.FIGURE_VISUAL,
        )
        # Chart + calculation cues
        self.assertEqual(
            QuestionRouter.classify_question("What percentage increase in accuracy is shown in Figure 4?"),
            QuestionRouteType.CHART_NUMERIC,
        )
        # Table cues
        self.assertEqual(
            QuestionRouter.classify_question("What is the BLEU score on WMT 2014 in Table 2?"),
            QuestionRouteType.TABLE_NUMERIC,
        )
        # Comparison cues
        self.assertEqual(
            QuestionRouter.classify_question("Compare the Transformer base model versus the big model"),
            QuestionRouteType.COMPARISON,
        )
        # Direct lookup cues
        self.assertEqual(
            QuestionRouter.classify_question("What learning rate and optimizer were used?"),
            QuestionRouteType.DIRECT_LOOKUP,
        )

    def test_routing_capability_gating(self):
        # VLM model: gets visual items
        vlm = ModelCapabilities(model_id="qwen3.5:9b", display_name="Qwen 3.5", supports_vision=True)
        budget_vlm = QuestionRouter.route("Explain the diagram in Figure 1", capabilities=vlm)
        self.assertEqual(budget_vlm.route_type, QuestionRouteType.FIGURE_VISUAL)
        self.assertEqual(budget_vlm.visual_items, 4)
        self.assertFalse(budget_vlm.capability_fallback)

        # Text-only model: visual items downgraded to 0, fallback enabled
        text_lm = ModelCapabilities(model_id="llama3.1:8b", display_name="Llama 3.1", supports_vision=False)
        budget_text = QuestionRouter.route("Explain the diagram in Figure 1", capabilities=text_lm)
        self.assertEqual(budget_text.route_type, QuestionRouteType.FIGURE_VISUAL)
        self.assertEqual(budget_text.visual_items, 0)
        self.assertTrue(budget_text.capability_fallback)
        self.assertGreaterEqual(budget_text.text_top_k, 5)

    def test_visual_grounding_crop_to_page_mapping(self):
        # Figure located at [0.1, 0.2, 0.9, 0.8] on page (width=0.8, height=0.6)
        figure_page_box = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, coordinate_space="normalized_page")
        
        # Subregion inside crop located at [0.5, 0.5, 1.0, 1.0] (bottom right quarter)
        crop_subregion = BoundingBox(x0=0.5, y0=0.5, x1=1.0, y1=1.0, coordinate_space="normalized_page")

        # Mapped to page: x0 = 0.1 + 0.5*0.8 = 0.5, y0 = 0.2 + 0.5*0.6 = 0.5
        # x1 = 0.1 + 1.0*0.8 = 0.9, y1 = 0.2 + 1.0*0.6 = 0.8
        page_box = VisualGroundingValidator.map_crop_to_page(crop_subregion, figure_page_box)
        self.assertAlmostEqual(page_box.x0, 0.5)
        self.assertAlmostEqual(page_box.y0, 0.5)
        self.assertAlmostEqual(page_box.x1, 0.9)
        self.assertAlmostEqual(page_box.y1, 0.8)

    def test_visual_region_expansion(self):
        box = BoundingBox(x0=0.4, y0=0.4, x1=0.6, y1=0.6)  # 0.2 x 0.2 centered at 0.5, 0.5
        expanded = VisualGroundingValidator.expand_region(box, expand_fraction=0.20)
        # width = 0.2 * 1.2 = 0.24 -> [0.38, 0.38, 0.62, 0.62]
        self.assertAlmostEqual(expanded.x0, 0.38)
        self.assertAlmostEqual(expanded.y0, 0.38)
        self.assertAlmostEqual(expanded.x1, 0.62)
        self.assertAlmostEqual(expanded.y1, 0.62)

    def test_claim_verification_and_repair(self):
        evidence = ["The Transformer model achieves 28.4 BLEU on the WMT 2014 English-to-German translation task."]

        # Supported claim
        v_sup = ClaimVerifierService.verify_claim(
            claim_id="C1",
            claim_text="The Transformer achieved 28.4 BLEU on English-to-German translation.",
            evidence_texts=evidence,
        )
        self.assertEqual(v_sup.label, VerificationLabel.SUPPORTED)

        # Contradicted claim (number mismatch)
        v_contra = ClaimVerifierService.verify_claim(
            claim_id="C2",
            claim_text="The Transformer achieved 95.2% BLEU on English-to-German translation.",
            evidence_texts=evidence,
        )
        self.assertEqual(v_contra.label, VerificationLabel.CONTRADICTED)

        # Partial claim and single repair (1 out of 3 tokens overlap -> 33% overlap)
        v_partial = ClaimVerifierService.verify_claim(
            claim_id="C3",
            claim_text="General optimization convergence task.",
            evidence_texts=evidence,
        )
        self.assertEqual(v_partial.label, VerificationLabel.PARTIALLY_SUPPORTED)
        repaired = ClaimVerifierService.apply_single_repair(v_partial, evidence)
        self.assertEqual(repaired.label, VerificationLabel.SUPPORTED)
        self.assertIn("According to the paper evidence", repaired.repaired_text or "")

    def test_all_ten_question_routes(self):
        # 1. DIRECT_LOOKUP
        self.assertEqual(QuestionRouter.classify_question("What learning rate was used?"), QuestionRouteType.DIRECT_LOOKUP)
        # 2. EXPLANATION
        self.assertEqual(QuestionRouter.classify_question("Explain the self-attention mechanism in detail."), QuestionRouteType.EXPLANATION)
        # 3. COMPARISON
        self.assertEqual(QuestionRouter.classify_question("Compare Transformer vs RNN models."), QuestionRouteType.COMPARISON)
        # 4. MULTI_SECTION
        self.assertEqual(QuestionRouter.classify_question("Trace the method across sections from start to finish."), QuestionRouteType.MULTI_SECTION)
        # 5. TABLE_NUMERIC
        self.assertEqual(QuestionRouter.classify_question("What is the BLEU score in Table 1?"), QuestionRouteType.TABLE_NUMERIC)
        self.assertEqual(QuestionRouter.classify_question("Explain Table 2"), QuestionRouteType.TABLE_NUMERIC)
        # 6. FIGURE_VISUAL
        self.assertEqual(QuestionRouter.classify_question("What is shown in Figure 2?"), QuestionRouteType.FIGURE_VISUAL)
        # 7. CHART_NUMERIC
        self.assertEqual(QuestionRouter.classify_question("What percentage increase is depicted in Figure 3 plot?"), QuestionRouteType.CHART_NUMERIC)
        # 8. MIXED_TEXT_VISUAL
        self.assertEqual(QuestionRouter.classify_question("Compare the text explanation and the diagram in Figure 1."), QuestionRouteType.MIXED_TEXT_VISUAL)
        # 9. CODE_ALGORITHM
        self.assertEqual(QuestionRouter.classify_question("Explain Algorithm 1 and its step-by-step logic"), QuestionRouteType.CODE_ALGORITHM)
        self.assertEqual(QuestionRouter.classify_question("Show the pseudocode implementation for the forward pass"), QuestionRouteType.CODE_ALGORITHM)

    def test_table_route_allocates_vision(self):
        caps = ModelCapabilities(model_id="gemma4:12b", display_name="Gemma 4", supports_vision=True)
        budget = QuestionRouter.route("Explain Table 2", caps)
        self.assertEqual(budget.route_type, QuestionRouteType.TABLE_NUMERIC)
        self.assertTrue(budget.requires_native_vision)
        self.assertGreaterEqual(budget.visual_items, 1)

    def test_evidence_sufficiency_and_abstention(self):
        # Empty evidence -> Insufficient
        suff_empty = ClaimVerifierService.compute_sufficiency("What is BLEU?", [])
        self.assertFalse(suff_empty.is_sufficient)
        self.assertEqual(suff_empty.reason_code, "INSUFFICIENT_TEXT_EVIDENCE")

        # Zero overlap -> Insufficient
        chunks = [{"text": "Photosynthesis is a biological process in plants."}]
        suff_unrel = ClaimVerifierService.compute_sufficiency("What learning rate optimizer was used for Transformer?", chunks)
        self.assertFalse(suff_unrel.is_sufficient)

    def test_subregion_proposal_extraction_and_geometry(self):
        # Sample VLM response with subregions
        vlm_sample = """
        **Answer**
        The blue curve indicates the proposed model achieving 28.4 BLEU.
        **Visual evidence**
        - Peak value is reached at step 100k.
        **Subregions**
        [{"role": "plot", "box": [0.2, 0.3, 0.8, 0.7]}]
        """
        proposal = VisualGroundingService.extract_subregion_proposals(vlm_sample, evidence_id="fig_01")
        self.assertEqual(len(proposal.regions), 1)
        self.assertAlmostEqual(proposal.regions[0].x0, 0.2)
        self.assertAlmostEqual(proposal.regions[0].y0, 0.3)
        self.assertAlmostEqual(proposal.regions[0].x1, 0.8)
        self.assertAlmostEqual(proposal.regions[0].y1, 0.7)

        # Map to page
        figure_page_box = BoundingBox(x0=0.1, y0=0.2, x1=0.9, y1=0.8, coordinate_space="normalized_page")
        resolved = VisualGroundingService.resolve_regions_from_proposal(
            proposal=proposal,
            parent_page_box=figure_page_box,
            document_id="paper_123",
            page_number=4,
        )
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0].page, 4)
        # x0 = 0.1 + 0.2*0.8 = 0.26, y0 = 0.2 + 0.3*0.6 = 0.38
        self.assertAlmostEqual(resolved[0].bbox_page_normalized.x0, 0.26)
        self.assertAlmostEqual(resolved[0].bbox_page_normalized.y0, 0.38)

    def test_query_decomposer(self):
        # Multi-figure comparison
        sub_figs = QueryDecomposer.decompose("compare figure 1 and 2")
        self.assertEqual(sub_figs, ["Figure 1", "Figure 2"])

        sub_figs_3 = QueryDecomposer.decompose("compare figures 1, 2, and 3")
        self.assertEqual(sub_figs_3, ["Figure 1", "Figure 2", "Figure 3"])

        # Multi-table comparison
        sub_tbls = QueryDecomposer.decompose("compare table 1 and table 2")
        self.assertEqual(sub_tbls, ["Table 1", "Table 2"])

        # Concept comparison
        sub_comp = QueryDecomposer.decompose("What is the difference between scaled dot-product attention and multi-head attention?")
        self.assertEqual(sub_comp, ["scaled dot-product attention", "multi-head attention"])

        # Methodology comparison
        sub_meth = QueryDecomposer.decompose("compare methodology 1 and methodology 2")
        self.assertEqual(sub_meth, ["methodology 1", "methodology 2"])

        # Single lookup should not split
        sub_single = QueryDecomposer.decompose("What learning rate was used?")
        self.assertEqual(sub_single, ["What learning rate was used?"])

    def test_multi_image_prompt_building(self):
        from backend.services.vision_service import _build_multi_vision_prompt
        figs = [
            {"label": "Figure 1", "caption": "Overall architecture of RLM."},
            {"label": "Figure 2", "caption": "Detailed REPL loop."},
        ]
        prompt = _build_multi_vision_prompt(
            question="compare figure 1 and 2",
            figures=figs,
            text_context="RLM evaluates recursive prompts.",
            paper_title="Recursive Language Models",
        )
        self.assertIn("[Image 1: Figure 1]", prompt)
        self.assertIn("[Image 2: Figure 2]", prompt)
        self.assertIn("Overall architecture of RLM.", prompt)
        self.assertIn("Detailed REPL loop.", prompt)
        self.assertIn("compare figure 1 and 2", prompt)

    def test_pedagogical_study_goals_structure(self):
        from backend.services.ollama_service import fallback_goals
        meta = {"title": "BERT: Pre-training of Deep Bidirectional Transformers", "summary": "We introduce BERT."}
        chunks = [{"chunk_id": "c1", "page": 1, "section_title": "1. Introduction", "text": "We introduce BERT for NLP."}]
        figs = [{"label": "Figure 1", "caption": "Architecture"}, {"label": "Table 1", "caption": "GLUE results"}]
        goals = fallback_goals(meta, chunks, figs)
        self.assertEqual(len(goals), 8)
        phases = [g["phase"] for g in goals]
        self.assertEqual(phases, ["Foundation", "Foundation", "Architecture", "Architecture", "Benchmarks", "Benchmarks", "Implementation", "Implementation"])
        for g in goals:
            self.assertIn("difficulty", g)
            self.assertIn("estimated_minutes", g)
            self.assertIn("target_evidence", g)
            self.assertIn("key_takeaways", g)
            self.assertGreaterEqual(len(g["subquestions"]), 3)

    def test_citation_pruning_on_negative_disclaimers(self):
        from backend.services.verifier_service import ClaimVerifierService
        answer = "BERT uses masked language modeling [1]. The term RLM is not used in the provided text [3]."
        citations = [
            {"ref_id": 1, "page": 3, "quote": "BERT uses masked language modeling to pre-train bidirectional representations."},
            {"ref_id": 3, "page": 4, "quote": "We use a case-preserving WordPiece model."},
        ]
        aligned_answer, final_cits, _ = ClaimVerifierService.verify_and_repair_answer(
            answer=answer,
            citations=citations,
        )
        self.assertIn("[1]", aligned_answer)
        self.assertNotIn("[3]", aligned_answer)
        self.assertIn("The term RLM is not used in the provided text.", aligned_answer)
        self.assertEqual(len(final_cits), 1)
        self.assertEqual(final_cits[0]["ref_id"], 1)

    def test_citation_auto_remapping(self):
        from backend.services.verifier_service import ClaimVerifierService
        # Answer claims masked LM but cited [2] (which is WordPiece). Candidate pool has [1] (masked LM).
        answer = "BERT pre-trains deep bidirectional representations [2]."
        citations = [
            {"ref_id": 2, "page": 4, "quote": "We use WordPiece vocabulary."},
        ]
        candidate_pool = [
            {"ref_id": 1, "page": 1, "quote": "BERT is designed to pre-train deep bidirectional representations from unlabeled text."},
            {"ref_id": 2, "page": 4, "quote": "We use WordPiece vocabulary."},
        ]
        aligned_answer, final_cits, _ = ClaimVerifierService.verify_and_repair_answer(
            answer=answer,
            citations=citations,
            candidate_pool=candidate_pool,
        )
        self.assertIn("[1]", aligned_answer)
        self.assertEqual(len(final_cits), 1)
        self.assertEqual(final_cits[0]["ref_id"], 1)
        self.assertIn("pre-train deep bidirectional", final_cits[0]["quote"])

    def test_figure_citation_auto_linking_and_preservation(self):
        from backend.services.verifier_service import ClaimVerifierService
        # Vision answer mentioning Figure 1 without explicit bracket numbers
        vision_ans = "**Answer**\nFigure 1 illustrates the impact of aggregation and initialization errors on federated learning.\n\n**Visual evidence**\n- Figure 1(a): Shows error rates."
        citations = [
            {
                "ref_id": 1,
                "page": 4,
                "label": "Figure 1",
                "is_figure": True,
                "quote": "Figure 1: Aggregation error and relative factor-level initialization errors on MNLI.",
                "figure_id": "fig_001",
            }
        ]
        aligned_ans, final_cits, _ = ClaimVerifierService.verify_and_repair_answer(
            answer=vision_ans,
            citations=citations,
        )
        self.assertIn("Figure 1 [1]", aligned_ans)
        self.assertEqual(len(final_cits), 1)
        self.assertEqual(final_cits[0]["ref_id"], 1)
        self.assertEqual(final_cits[0]["page"], 4)
        self.assertTrue(final_cits[0]["is_figure"])

    def test_pdf_region_crop(self):
        import fitz
        from pathlib import Path
        import tempfile
        from backend.services.pdf_service import crop_page_region

        with tempfile.TemporaryDirectory() as tmpdir:
            pdf_path = Path(tmpdir) / "sample.pdf"
            doc = fitz.open()
            page = doc.new_page(width=600, height=800)
            page.insert_text((50, 100), "Equation (1): L = E[log p(x)]", fontsize=14)
            doc.save(str(pdf_path))
            doc.close()

            png_bytes, extracted_text = crop_page_region(
                pdf_path=pdf_path,
                page_number=1,
                bbox_norm=[0.05, 0.05, 0.95, 0.25],
                zoom=2.0,
            )
            self.assertTrue(len(png_bytes) > 0)
            self.assertIn("Equation", extracted_text)

    def test_snippet_chat_payload(self):
        from backend.main import ChatInput, SnippetInput
        snip_in = SnippetInput(page=3, bbox=[0.1, 0.2, 0.5, 0.6])
        self.assertEqual(snip_in.page, 3)
        self.assertEqual(len(snip_in.bbox), 4)

        chat_in = ChatInput(
            message="Explain this formula",
            snippet_id="snip_12345",
            snippet_page=3,
            snippet_bbox=[0.1, 0.2, 0.5, 0.6],
            snippet_text="L = E[log p(x)]",
        )
        self.assertEqual(chat_in.snippet_id, "snip_12345")
        self.assertEqual(chat_in.snippet_page, 3)


if __name__ == "__main__":
    unittest.main()
