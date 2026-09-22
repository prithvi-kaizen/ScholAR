# ScholAR EACL Demo 150-Case Luna Independent LLM Evaluation Protocol

This document defines the evaluation protocol for running GPT-5.6 Luna as an LLM judge on the exact 150-question complement of the 50-question human evaluation sample.

## Completion audit note (2026-09-21)

The 150-case batch is complete, but the retained judgment rows and public receipt identify the evaluator as `manual-assistant-proxy`; they do not contain a provider-side receipt that verifies the planned model identity. The later 90-case visual update also uses `partly_supported`, which is not permitted by the frozen `visual_grounding` enum below. Accordingly, paper-facing use is limited to the non-visual categorical counts under the name **LLM-based evaluation**. The visual distribution is excluded, and none of these labels is described as human evaluation, ground truth, or a validated named-model judge.

## 1. Boundary and Purpose
- **LLM-based evaluation**: The protocol planned an external model (GPT-5.6 Luna) to assess ScholAR's local `qwen3.5:9b` answers. The retained completion metadata does not verify that exact evaluator identity, so the paper reports only a generic LLM judge and **never** labels its outputs as human evaluation, expert review, or ground truth.
- **Strict Data Boundary**: External API requests send paper excerpts and answers. Calls must use `store=false`. Source PDFs and raw judgments remain private under `evaluation/results/eacl_demo_150_luna/`.

## 2. Complement Sample Definition
- **Original Universe**: 200 benchmark questions across 10 papers (`evaluation/benchmarks/two_hundred_questions_dataset.json`).
- **Human Study Sample**: Exactly 50 questions selected in `offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters/selection_manifest.json`.
- **Luna Target Sample**: Exactly 150 complement questions ($200 - 50 = 150$, overlap $= 0$).
- **Reference Answer Eligibility**:
  - **113 Eligible Cases**: Pass the fail-closed question-matching test with non-empty reference answers.
  - **37 Ineligible Cases**: Mismatched or blank references. These cases are recorded in `reference_adjudication.jsonl` and their correctness dimension is strictly scored as `not_judgeable`. Direct evidence grounding and citation quality can still be evaluated against cited source excerpts.
  - The current runner is text-only. If `has_visual_evidence=false`, set `visual_grounding=not_applicable`. If it is true but no page pixels or crop are supplied, set `visual_grounding=not_judgeable`; do not infer visual support from text or metadata.

## 3. Evaluation Dimensions & Schema
The rubric is frozen at `evaluation/protocols/eacl_demo_150_llm_judge.json`.
- `correctness`: `correct`, `partly_correct`, `incorrect`, `not_judgeable`
- `completeness`: `complete`, `partially_complete`, `incomplete`, `not_judgeable`
- `grounding`: `supported`, `partly_supported`, `unsupported`, `not_judgeable`
- `citation_quality`: `accurate`, `partially_accurate`, `inaccurate`, `missing`, `not_judgeable`
- `visual_grounding`: `supported`, `unsupported`, `not_applicable`, `not_judgeable`
- `failure_type`: `none`, `wrong_fact`, `missing_fact`, `unsupported_claim`, `wrong_citation`, `missing_citation`, `hallucination`, `other`
- `brief_reason`: text explanation

## 4. Execution Commands
### Dry-Run (No network, validates inputs and hashes)
```bash
python -m evaluation.run_eacl_demo_150_luna_judge --dry-run
```

### Execution (Requires local OPENAI_API_KEY and confirmation)
```bash
python -m evaluation.run_eacl_demo_150_luna_judge --execute --allow-external --max-new 10
```

### Aggregation & Public Receipt
```bash
python -m evaluation.aggregate_eacl_demo_150_luna_judge
```
Outputs privacy-safe aggregate statistics to `evaluation/EACL_DEMO_150_LUNA_PUBLIC.json`.
