"""Tests for Comparative Scaling & Joint Text-Vision Multimodal Fusion."""

import asyncio
import unittest
from unittest.mock import AsyncMock, patch

from backend.services.ollama_service import LocalGenerationResult
from backend.services.retrieval_service import (
    _COMPARATIVE_SCALING_PATTERNS,
    _section_hints,
    retrieve_chunks,
)
from backend.services.vision_service import (
    _build_multi_vision_prompt,
    _build_visual_observation_prompt,
    _parse_visual_observations,
    answer_with_multimodal_evidence,
    is_uninformative_visual_answer,
)


class TestComparativeVisualTextFusion(unittest.TestCase):
    """Test suite for comparative scaling query handling and multimodal text-vision fusion."""

    def test_comparative_scaling_patterns_detection(self):
        """Verify regex patterns match comparative scaling and threshold queries."""
        queries = [
            "what input context at which RLM outperforms the normal model?",
            "at what context length does RLM beat the base model?",
            "what is the crossover threshold where RLM outperforms GPT-5?",
            "how does performance degrade with longer context?",
            "explain the tradeoff point between base LM and RLM",
            "scaling comparison between RLM and vanilla model",
        ]
        for q in queries:
            self.assertTrue(bool(_COMPARATIVE_SCALING_PATTERNS.search(q.lower())), f"Failed to match: {q}")

    def test_section_hints_for_comparative_queries(self):
        """Verify _section_hints extracts result/experiment for comparative queries."""
        query = "what input context at which RLM outperforms the normal model?"
        hints = _section_hints(query)
        self.assertIn("result", hints)
        self.assertIn("experiment", hints)

    def test_uninformative_visual_answer_detection(self):
        """Verify is_uninformative_visual_answer flags refusals and passes good answers."""
        uninformative_samples = [
            "Based on the provided visual evidence, it is not possible to determine specific input contexts or data points where RLM outperforms a normal model. The available images and text describe the mechanism of RLMs but do not provide comparative performance metrics.",
            "The images do not contain a chart or table comparing RLM performance to baseline models.",
            "Figure 2 illustrates the architecture but contains no performance data or comparison graphs.",
            "",
            "Too short",
        ]
        for sample in uninformative_samples:
            self.assertTrue(is_uninformative_visual_answer(sample))

        informative_samples = [
            "**Answer**\nRLM consistently outperforms the base model for context lengths beyond $2^{14}$ (~16k tokens) as detailed in Observation 3 [1] and Figure 1 [2]. Below 16k tokens, the base model slightly outperforms RLM due to REPL initialization overhead.",
            "**Answer**\nTable 1 [1] shows that RAG-Sequence achieves 44.5 on NQ compared to 34.5 for T5-11B.",
        ]
        for sample in informative_samples:
            self.assertFalse(is_uninformative_visual_answer(sample))

    def test_retrieve_chunks_boosts_empirical_results(self):
        """Verify retrieve_chunks prioritizes empirical results sections and comparison plots."""
        chunks = [
            {
                "chunk_id": "chunk_sec2_arch",
                "section_title": "2. Recursive Language Models",
                "section": "2. Recursive Language Models",
                "chunk_type": "method",
                "text": "A Recursive Language Model (RLM) is an inference-time scaffold around M that treats the user prompt as part of the environment.",
                "page": 2,
            },
            {
                "chunk_id": "fig_02_arch",
                "label": "Figure 2",
                "caption": "A Recursive Language Model (RLM) treats prompts as part of the environment. It loads the input prompt as a variable inside a REPL environment.",
                "figure_type": "figure",
                "is_figure_chunk": True,
                "text": "Figure 2: A Recursive Language Model treats prompts as part of the environment.",
                "page": 2,
            },
            {
                "chunk_id": "chunk_sec4_obs3",
                "section_title": "4. Results and Discussion",
                "section": "Observation 3: LM performance degrades as a function of input length",
                "chunk_type": "result",
                "text": "Observation 3: For context lengths beyond 2^14, the RLM consistently outperforms GPT-5. In the small input context regime, the base LM outperforms RLM.",
                "page": 6,
            },
            {
                "chunk_id": "fig_01_scaling",
                "label": "Figure 1",
                "caption": "A comparison of GPT-5 and a corresponding RLM using GPT-5 on three long-context tasks of increasing complexity. For context lengths beyond 2^14, RLM maintains strong performance while GPT-5 degrades.",
                "figure_type": "figure",
                "is_figure_chunk": True,
                "text": "Figure 1: Comparison of GPT-5 and RLM across input lengths from 2^13 to 2^18.",
                "page": 1,
            },
        ]

        query = "what input context at which RLM outperforms the normal model?"
        ranked = retrieve_chunks(query, chunks, limit=4)

        # The top retrieved chunk should be the empirical result (Observation 3 or Figure 1)
        top_chunk_ids = [c["chunk_id"] for c in ranked[:2]]
        self.assertTrue("chunk_sec4_obs3" in top_chunk_ids or "fig_01_scaling" in top_chunk_ids)

    def test_visual_prompts_keep_image_and_source_provenance_separate(self):
        figures = [
            {
                "_vision_evidence_id": "V1",
                "_vision_ref_id": 1,
                "source_paper_id": "paper-a",
                "_source_title": "Anchor Study",
                "label": "Figure 1",
                "caption": "Alpha trend",
            },
            {
                "_vision_evidence_id": "V2",
                "_vision_ref_id": 2,
                "source_paper_id": "paper-b",
                "_source_title": "Reference Study",
                "label": "Figure 1",
                "caption": "Beta trend",
            },
        ]
        observation_prompt = _build_visual_observation_prompt(
            "Compare the trends",
            figures,
            {
                "paper-a": {"title": "Anchor Study"},
                "paper-b": {"title": "Reference Study"},
            },
        )
        self.assertIn("V1 (citation [1], source_id='paper-a'", observation_prompt)
        self.assertIn("V2 (citation [2], source_id='paper-b'", observation_prompt)

        observations = _parse_visual_observations(
            '{"observations": ['
            '{"evidence_id":"V1","observation":"alpha only"},'
            '{"evidence_id":"V2","observation":"beta only"}] }',
            {"V1", "V2"},
        )
        self.assertEqual(observations, {"V1": "alpha only", "V2": "beta only"})
        synthesis = _build_multi_vision_prompt(
            question="Compare the trends",
            figures=figures,
            text_context="[T1 -> citation [3] | source_id=paper-b] supporting text",
            paper_title="Anchor Study",
            visual_observations=observations,
            source_metadata={"paper-b": {"title": "Reference Study"}},
        )
        self.assertIn("V1 -> citation [1] | source_id=paper-a", synthesis)
        self.assertIn("V2 -> citation [2] | source_id=paper-b", synthesis)
        self.assertIn("alpha only", synthesis)
        self.assertIn("beta only", synthesis)

    def test_multimodal_answer_attaches_only_matching_source_observation(self):
        figures = [
            {
                "chunk_id": "fig-a",
                "figure_id": "1",
                "source_paper_id": "paper-a",
                "label": "Figure 1",
                "caption": "Alpha caption",
                "figure_type": "figure",
                "image_file": "same.png",
                "page": 2,
            },
            {
                "chunk_id": "fig-b",
                "figure_id": "1",
                "source_paper_id": "paper-b",
                "label": "Figure 1",
                "caption": "Beta caption",
                "figure_type": "figure",
                "image_file": "same.png",
                "page": 4,
            },
        ]
        generations = [
            LocalGenerationResult(
                response=(
                    '{"observations":['
                    '{"evidence_id":"V1","observation":"alpha pixels"},'
                    '{"evidence_id":"V2","observation":"beta pixels"}]}'
                ),
                requested_model="vision-test",
                resolved_model="vision-test",
            ),
            LocalGenerationResult(
                response="**Answer**\nAlpha [1] and beta [2] are distinct.",
                requested_model="vision-test",
                resolved_model="vision-test",
            ),
        ]
        with (
            patch(
                "backend.services.vision_service.ollama_available",
                AsyncMock(return_value=True),
            ),
            patch(
                "backend.services.vision_service._load_figure_png",
                side_effect=lambda source_id, _name: source_id.encode("utf-8"),
            ) as load_image,
            patch(
                "backend.services.vision_service.generate_result",
                AsyncMock(side_effect=generations),
            ),
        ):
            result = asyncio.run(answer_with_multimodal_evidence(
                question="Compare alpha and beta",
                figure_chunks=figures,
                context_chunks=[],
                paper_id="paper-a",
                paper_metadata={"title": "Anchor Study"},
                source_metadata={
                    "paper-a": {"title": "Anchor Study"},
                    "paper-b": {"title": "Reference Study"},
                },
                model="vision-test",
            ))

        self.assertEqual(
            [call.args[0] for call in load_image.call_args_list],
            ["paper-a", "paper-b"],
        )
        image_citations = result["citations"][:2]
        self.assertEqual(image_citations[0]["visual_observation"], "alpha pixels")
        self.assertEqual(image_citations[1]["visual_observation"], "beta pixels")
        self.assertNotIn("beta pixels", image_citations[0]["quote"])
        self.assertEqual(image_citations[1]["source_paper_id"], "paper-b")
        self.assertEqual(image_citations[1]["source_title"], "Reference Study")

    def test_build_multi_vision_prompt_synthesis_instructions(self):
        """Verify _build_multi_vision_prompt instructs joint text-vision synthesis."""
        figures = [{
            "label": "Figure 1",
            "caption": "Comparison of GPT-5 and RLM",
            "figure_type": "figure",
        }]
        prompt = _build_multi_vision_prompt(
            question="what input context at which RLM outperforms the normal model?",
            figures=figures,
            text_context="Observation 3: For context lengths beyond 2^14, RLM outperforms GPT-5.",
            paper_title="Recursive Language Models",
        )

        self.assertIn("CRITICAL INSTRUCTIONS FOR MULTIMODAL SYNTHESIS", prompt)
        self.assertIn("Synthesize findings from BOTH the visual figures and the supporting text passages", prompt)
        self.assertIn("Quantitative Findings / Thresholds", prompt)


if __name__ == "__main__":
    unittest.main()

