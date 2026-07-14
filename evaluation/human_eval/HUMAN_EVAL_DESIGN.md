# ScholAR Human Evaluation: Design Specification

This document specifies the human evaluation pipeline for ScholAR. It is grounded in the published human-evaluation methodology of the leading scholarly-RAG systems (OpenScholar, SciRAG, PaperQA2), the general LLM-answer evaluation conventions (MT-Bench, Chatbot Arena), and the foundational citation-evaluation framework (ALCE). It defines the exact questions asked of the evaluator, the 4-model comparative structure, the 100-case set, the annotation protocol, and the scoring plan. A separate `BUILD_PROMPT.md` turns this specification into runnable artifacts.

## 1. Purpose

The ScholAR paper currently reports only automated metrics: retrieval Recall@K/MRR, the NLI-CFS faithfulness score (SummaC-ZS + SCR + KFP), and keyword-overlap answer-quality proxies. Every comparable system in this space includes a human evaluation, so its absence is a reviewer-facing gap. This pipeline has two goals:

1. **Demonstrate model-agnostic grounding.** Each question is answered by 4 different local models plugged into the same ScholAR retrieval and citation-grounding pipeline. Only the generation LLM changes. If all 4 models score similarly on Faithfulness and citation support, that shows ScholAR's grounding is robust to the choice of local model, which is a core claim for a fully local system where the user might swap models.

2. **Validate the automated NLI-CFS metric.** By collecting human Faithfulness judgments on the same answers the automated metric scores, we can report the correlation between the two. This upgrades the paper's central claim from "we used automated proxies" to "we validated those proxies against expert human judgment."

## 2. How comparable systems run human evaluation

| System | Annotators | Items | Answer-level questions | Scale | Citation grading | Multi-system format |
|---|---|---|---|---|---|---|
| SciRAG (arXiv:2511.14362) | 3 experts (MSc+ CS) | 30 queries | Relevance, Coverage, Organization, Overall Usefulness | 1-5, anchored 1/5 | Citation F1 | Absolute per-system score, averaged over raters |
| OpenScholar (arXiv:2411.14199) | 16 (PhD/postdoc, >=3 yrs) | 108 questions, 1-3 raters each | Organization, Coverage, Relevance, Usefulness | 1-5 | Citation precision/recall/F1 | Absolute per-system, plus pairwise Win/Tie/Lose vs a reference (kappa 0.68) |
| PaperQA2 (arXiv:2409.13740) | PhD domain experts | LitQA2 (~200) + summarization sample | LitQA2 accuracy/precision | MCQ | Per-statement: cited-and-supported / missing / cited-and-unsupported | Per-system accuracy |
| MT-Bench (arXiv:2306.05685) | crowd + expert | 80 multi-turn | Single overall quality score | 1-10 | none | Single Answer Grading (1-10) or pairwise (Chatbot Arena) |
| YESciEval (arXiv:2505.14279) | LLM judge + human | scientific QA | 9 dimensions (correctness, completeness, coherence, relevance, integration, informativeness, readability, conciseness, cohesion) | Likert | none | per-answer scoring |
| ALCE (arXiv:2305.14627) | defines metrics | - | fluency, correctness | - | Citation recall (cited passages jointly entail the statement); precision (a citation is irrelevant if it alone does not support the claim AND removing it does not change support) | - |

## 3. What exactly the evaluator is asked (the format question, answered)

A common question is whether the evaluator gives a single overall score (for example 0-10) or something else. The evidence is clear:

- **The scholarly-RAG papers do not use a single overall score.** They ask multiple questions per answer, one per quality dimension, each on a 1-5 Likert scale with anchored definitions, phrased as instructions rather than open questions. SciRAG's exact wording: Relevance "Evaluate if the response stays on topic and maintains a clear focus"; Coverage "Evaluate if the output provides sufficient coverage and amount of information"; Organization "Evaluate if the output is well-organized and logically structured"; Overall Usefulness "Evaluate if the output contains useful information to fulfill the information needs." OpenScholar uses the same four dimensions on 1-5.
- **Citation quality is asked separately, at the level of individual statements**, not as a whole-answer score. PaperQA2 has the evaluator label each statement cited-and-supported, missing-citation, or cited-and-unsupported.
- **The single 0-10 overall score is the MT-Bench convention** for general chatbots ("Single Answer Grading"). It is considered coarser and is not used in the scholarly-QA literature, which has moved to multi-dimensional 1-5 scoring (YESciEval uses 9 dimensions).
- **For comparing multiple systems on the same question, the standard is absolute per-system scoring**, so mean scores can be compared across systems (SciRAG). OpenScholar adds a pairwise preference on top.

ScholAR therefore uses multiple 1-5 dimension questions plus per-citation grading, with absolute per-model scoring across the 4 models, plus a ranking. The exact question set is in Section 6.

## 4. The 4-model comparative structure

Each question is answered by 4 local models. The retrieval, chunking, page-preserving segmentation, and indirect citation grounding are identical across all four; only the generation LLM that writes the final answer changes. This isolates the effect of model choice on answer quality and citation faithfulness.

- **Models**: qwen3.5:9b, gemma4:12b, llama3.1:8b, mistral:7b. Four vendors (Alibaba, Google, Meta, Mistral) spanning 7-12B, all run one at a time on the 18GB target hardware. Confirm the exact set with the professor before building.
- **Visual cases**: only qwen3.5:9b and gemma4:12b are natively multimodal, so the 4-model comparison runs on the 75 text and mathematical cases. The 25 visual cases are human-evaluated with the 2 vision-capable models using the same instrument, which also upgrades the paper's existing 2-model visual comparison from an automated proxy to human judgment.
- **Model-agnostic claim**: supported if the 4 models show low variance on Faithfulness and citation-support even if fluency or usefulness varies. Any real gap is reported honestly rather than hidden.

## 5. Why ScholAR's citation grounding fits the claim-level scheme exactly

ScholAR's chat endpoint returns every answer with a `citations` array in which each numbered citation `[n]` carries an exact `page`, a `quote` (the supporting sentence), a `chunk_id`, a `section_title`, and a `source_paper_id`. This is the ideal substrate for PaperQA2-style per-statement grading: the claim, the cited page, and the exact quoted evidence are presented together, so the highest-signal judgment (does the cited evidence support the claim) is fast and reliable.

## 6. The ScholAR instrument: the exact questions

The evaluator answers Q1 to Q6 for each of the 4 model answers to a question, then Q7 once across the 4 answers. Answers are shown blind (no model identity) in randomized order.

### Part A: answer-level rubric (four 1-5 Likert questions, anchored 1/3/5)

ScholAR follows SciRAG and OpenScholar, with one deliberate substitution: it replaces "Organization" with "Faithfulness". Organization matters less for ScholAR's short grounded answers, and Faithfulness is the core contribution and the dimension that validates NLI-CFS.

**Q1 Relevance**: Does the answer stay on topic and directly address the question?
- 1: Does not address the question, or is off-topic.
- 3: Partially addresses the question but drifts or omits the core ask.
- 5: Stays tightly focused on exactly what was asked, with appropriate depth.

**Q2 Coverage**: Does the answer capture the key facts the question requires? (Use the case `gold_answer` and `must_include` list as the reference.)
- 1: Misses most of the key facts the question requires.
- 3: Captures some key facts but omits at least one important item.
- 5: Captures all the key facts the question requires.

**Q3 Faithfulness**: Is every statement in the answer supported by the paper, with no fabrication or overstatement?
- 1: Contains clear fabrication, or a claim that contradicts the paper.
- 3: Mostly grounded, but includes at least one unsupported or overstated claim.
- 5: Every statement is directly supported by the paper, with no fabrication or overstatement.

**Q4 Usefulness**: Overall, how useful is this answer for understanding the paper?
- 1: Confusing or unhelpful; would not help a researcher.
- 3: Somewhat helpful but incomplete or hard to follow.
- 5: Clear, complete, and genuinely helpful for understanding the paper.

### Part B: per-citation grading (Q5 and Q6)

**Q5** for each numbered citation `[n]`: open its `page`, read its `quote`, and label whether the cited evidence supports the specific claim it is attached to:
- **Supported**: the cited evidence directly and fully supports the claim.
- **Partial**: related and supports the claim in part, but does not fully establish it (supports one part of a multi-part claim, or is topically correct but missing the specific value or detail claimed).
- **Unsupported**: does not support the claim (wrong page or figure, topically unrelated, or contradicts the claim). This is a citation hallucination or misattribution.

**Q6** (free text): list any checkable, paper-specific factual claim in the answer that has no citation but should have one. This count feeds citation recall.

### Part C: comparative ranking (Q7)

**Q7**: rank the 4 answers to this question from best to worst overall. Ties are allowed. This is a secondary preference signal on top of the absolute scores.

### Derived citation metrics (per answer, then averaged)

Following ALCE and OpenScholar:
- Citation precision (strict) = Supported / (Supported + Partial + Unsupported)
- Citation precision (lenient) = (Supported + 0.5 * Partial) / (Supported + Partial + Unsupported)
- Citation recall = (cited citation-worthy claims) / (cited citation-worthy claims + missing-citation claims)
- Citation F1 = harmonic mean of citation precision and citation recall

The raw label distribution (percent Supported / Partial / Unsupported) is also reported, mirroring PaperQA2's headline three-category result.

## 7. The 100 curated cases

The set spans 25 papers and three query types, mined from the prepared corpus by a local model and source-verified (every gold fact was checked to appear verbatim in its source passage) by `evaluation/mine_cases.py`. The earlier hand-curated 3-paper set is preserved as `cases_curated_3paper.json`.

| Capability | Count | Models that answer | Source pattern |
|---|---|---|---|
| Single-document text QA | 50 | all 4 | Body-level factual questions mined and source-verified across the 25-paper corpus |
| Mathematical | 25 | all 4 | Questions over equations, derivations, and numerical results |
| Visual grounding | 25 | 2 multimodal (qwen, gemma) | Figure and table questions, in the style of `visual_benchmark.json` |

### Case schema (JSON)

```json
{
  "case_id": "he_001",
  "capability": "single_doc_text | math | visual",
  "paper_id": "1706.03762",
  "secondary_paper_ids": [],
  "question": "The user-facing question, phrased as a real researcher would ask it.",
  "gold_answer": "A concise expert reference answer, used to ground Coverage and Faithfulness scoring.",
  "must_include": ["key fact 1", "key fact 2"],
  "answer_locus": "page 4 / Figure 1 / reference [Sennrich 2016]",
  "difficulty": "easy | medium | hard",
  "notes": "Why this case exists and what it tests."
}
```

The `gold_answer` and `must_include` fields are the evaluator's ground truth. They are what make Coverage and Faithfulness judgments consistent across annotators rather than subjective.

## 8. Volume and annotation protocol

- **Full volume**: 75 text and mathematical cases x 4 models = 300 answers, plus 25 visual cases x 2 models = 50 visual answers, for **350 human-scored answers** carrying 829 citations in total (mean 2.4 per answer). Each answer takes Q1-Q6; each question adds one Q7 ranking.
- **Reduced protocol** if an evaluator is time-limited: a stratified sample (for example 20 text + 10 math cases x 4 models, plus 10 visual x 2 = 140 answers) still exceeds the scale of SciRAG (90) and OpenScholar (108 questions, ~25 instances per annotator).
- **Blinding and order**: the evaluator sees only the question and the answers with their citations, never the model identity, and the 4 answers appear in randomized order per question.
- **Inter-annotator agreement**: a 20-answer overlap subset is scored independently by at least two evaluators. Report Cohen's kappa on the three-way citation labels and Krippendorff's alpha (or Pearson) on the Likert dimensions, matching how OpenScholar (kappa 0.68) and SciRAG (~0.87) report agreement.
- **Worked examples**: the evaluator instruction sheet includes one worked Supported, one Partial, and one Unsupported citation.

## 9. Scoring and analysis

The pipeline computes and reports:

1. **Per-model mean Likert** (Relevance, Coverage, Faithfulness, Usefulness), overall and per capability, with cross-model variance. Low variance on Faithfulness and citation-support is the model-agnostic evidence.
2. **Per-model citation precision, recall, F1**, plus the raw Supported/Partial/Unsupported distribution.
3. **Ranking summary**: how often each model ranks first, and mean rank per model.
4. **Significance test**: a Friedman test across the 4 models on the same questions for the Faithfulness dimension and the citation-support rate. A non-significant difference supports model-agnostic grounding; any significant fluency or usefulness gap is reported honestly.
5. **Validation result**: Pearson and Spearman correlation between the per-answer human Faithfulness score (1-5) and ScholAR's automated NLI-CFS (0-1), pooled across models and per model, plus the correlation between the per-answer citation-support rate and the SCHR / KFP components of NLI-CFS. A strong positive correlation demonstrates the automated metric tracks expert judgment, which is the result that most strengthens the paper.

## 10. References

- Asai et al. OpenScholar: Synthesizing Scientific Literature with Retrieval-augmented LMs. arXiv:2411.14199.
- Ding et al. SciRAG: Adaptive, Citation-Aware, and Outline-Guided Retrieval and Synthesis for Scientific Literature. arXiv:2511.14362.
- Skarlinski et al. Language Agents Achieve Superhuman Synthesis of Scientific Knowledge (PaperQA2). arXiv:2409.13740.
- Zheng et al. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. arXiv:2306.05685.
- YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering. arXiv:2505.14279.
- Gao et al. Enabling Large Language Models to Generate Text with Citations (ALCE). EMNLP 2023, arXiv:2305.14627.
