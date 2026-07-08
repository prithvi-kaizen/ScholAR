# Build Prompt: ScholAR Human Evaluation Pipeline (4-Model Comparative)

Paste the prompt below into Claude Code (from the ScholAR repo root) to build the human evaluation pipeline. It implements the design in `evaluation/human_eval/HUMAN_EVAL_DESIGN.md`. Read that design file first; it defines the 4 models, the exact Q1-Q7 evaluator questions, the case distribution, and every metric this prompt refers to.

Prerequisites before running: the backend must be running (`make backend`) with the target papers already prepared under `backend/data/papers/`, and all 4 models pulled in Ollama (`ollama pull qwen3.5:9b gemma4:12b llama3.1:8b mistral:7b`), so the answer-generation step can call the live `/api/papers/{id}/chat` endpoint with each model.

---

## Prompt

You are building the human evaluation pipeline for ScholAR, a local-first RAG system for scientific papers. The full design is in `evaluation/human_eval/HUMAN_EVAL_DESIGN.md`. Read it in full before writing anything. Follow it exactly: the 4-model comparative structure, the four 1-5 Likert dimensions, the three-way citation labels, the ranking question, the case distribution, and the derived metrics are all specified there and must not be redesigned. Do not add em dashes to any prose you write (project convention in `CLAUDE.md`).

Work in plan mode first per the project workflow: write your plan to `.claude/tasks/`, get it reviewed, then implement. Produce these five artifacts in `evaluation/human_eval/`.

### 1. `cases.json` (100 curated cases)

Build 100 cases following the schema in Section 7 of the design. Distribution: 40 single-document text QA, 20 visual grounding, 20 multi-document, 20 hard-retrieval.

- For the 40 single-document cases, start from the existing 51 cases in `evaluation/faithfulness_cases.json`. Convert each into the new schema, preserving the claim-type spread. Map `query` to `question`, `expected_claim` to the basis for `gold_answer` and `must_include`, and record the supporting chunk's page as `answer_locus`.
- For the 20 visual cases, adapt from `evaluation/visual_benchmark.json` (fields `question`, `expected_figure_label`, `anchor_paper`).
- For the 20 multi-document cases, adapt from `evaluation/multidoc_benchmark.json`, keeping only cases whose expected answer is in a resolvable cited paper. Set `secondary_paper_ids` accordingly.
- For the 20 hard-retrieval cases, write new questions whose answer is in the body of one of the prepared papers and is not stated in the abstract. Each must be single-source and specific. This follows LitQA2's design principle; cite it as methodology, do not copy its data.
- Every case needs a real `gold_answer` and a `must_include` list. These are the annotator's ground truth. Do not leave placeholders. Verify each `gold_answer` against the actual paper text in `backend/data/papers/{paper_id}/pages.json` before writing it.
- Only use `paper_id` values that actually exist under `backend/data/papers/`. Check first.

### 2. `generate_answers.py` (4-model answer generation)

For each case, generate an answer from each applicable model by swapping the generation model per request against the same ScholAR pipeline:

- The 4 text models are qwen3.5:9b, gemma4:12b, llama3.1:8b, mistral:7b. Text cases (single_doc_text, multi_doc, hard_retrieval) are answered by all 4. Visual cases are answered only by the 2 multimodal models (qwen3.5:9b, gemma4:12b).
- Swap the model per request. Check whether the backend accepts a per-request model override on `/api/papers/{paper_id}/chat`; if it does not, add a minimal optional `model` field to the chat request handler in `backend/main.py` that overrides `OLLAMA_MODEL` for that request only, defaulting to the env value when absent. Keep retrieval, chunking, and citation grounding identical across models; only the generation model changes.
- Capture the full response per (case, model): `answer`, `citations` (each with `ref_id`, `page`, `quote`, `chunk_id`, `section_title`, `source_paper_id`), `vision`, `figure_image_url`, and the `model` used.
- Reuse the backend-call pattern from `evaluation/run_multidoc_eval.py`. Write `answers.json` as a list of `{case_id, model, question, answer, citations, ...}`. Handle an unreachable backend or a missing model with a clear error; do not silently write empty answers.

### 3. `rubric.md` (the PhD evaluator instruction sheet)

A standalone document the evaluator reads before scoring. It must:

- Explain the task: for each question they see 4 answers (2 for visual cases) in randomized blind order, each with numbered citations, and they score every answer.
- Reproduce Q1-Q4 (the anchored 1-5 Likert dimensions Relevance, Coverage, Faithfulness, Usefulness) verbatim from Section 6, including the 1/3/5 anchors.
- Reproduce Q5 (the three citation labels Supported/Partial/Unsupported) and Q6 (missing-citation free text), with one worked example of each label using a realistic ScholAR answer and citation.
- Reproduce Q7 (rank the answers best to worst).
- Explain that the `gold_answer` and `must_include` list are the reference for Coverage and Faithfulness, and state the blinding and overlap-subset instructions from Section 8.

### 4. `score_sheet.html` (self-contained scoring interface)

A single self-contained local HTML file (inline CSS/JS, no external CDN, per the project artifact constraints) that:

- Loads `answers.json` and `cases.json` and renders each question in turn: the question, the `gold_answer` and `must_include` for reference, then the 2-4 answers for that question in randomized blind order (label them Answer A, B, C, D; keep the hidden mapping to the real model for export).
- Shows each answer with its numbered citations, each citation displaying its page and quoted evidence inline.
- Provides per answer: four 1-5 dropdowns (Q1-Q4), a Supported/Partial/Unsupported dropdown per citation (Q5), and a free-text field for missing citations (Q6). Provides per question a ranking control for Q7.
- Exports all entered scores to a downloadable JSON (and CSV) keyed by `case_id` and the real `model` (un-blinded on export only), when the evaluator clicks a button.
- Persists progress to `localStorage` so a long session is not lost on refresh.

### 5. `compute_scores.py`

Reads the exported score file, `answers.json`, and the automated NLI-CFS results in `evaluation/results/faithfulness_eval_results_v3.json`, and computes per Section 9 of the design:

- Per-model mean Likert (overall and per capability) with cross-model variance.
- Per-model citation precision (strict and lenient), recall, and F1, plus the raw Supported/Partial/Unsupported distribution.
- Ranking summary: first-place rate and mean rank per model.
- Friedman test across the 4 models on the Faithfulness dimension and citation-support rate (same questions, related samples).
- Inter-annotator agreement on the overlap subset: Cohen's kappa on citation labels, Krippendorff's alpha or Pearson on the Likert dimensions.
- The validation result: Pearson and Spearman correlation between per-answer human Faithfulness (1-5) and automated NLI-CFS (0-1), pooled and per model, and between citation-support rate and the SCHR/KFP components, for the cases that overlap the 51-case faithfulness benchmark.
- Write `human_eval_results.json` and a short `human_eval_report.md` summarizing the numbers in a form that can be dropped into the paper (with a per-model table and the model-agnostic conclusion).

### Verification

- `cases.json` validates against the schema, has exactly 100 cases in the specified distribution, and every `paper_id` exists under `backend/data/papers/`.
- `generate_answers.py` runs end to end against the live backend and produces the right number of answers per case (4 for text, 2 for visual), each tagged with its model.
- `score_sheet.html` opens in a browser, renders every question with its blinded answers and citations, persists progress, and its export produces a well-formed file with the model mapping restored.
- `compute_scores.py` runs on a small hand-filled sample without error and produces both output files, including the per-model table and the human-vs-NLI-CFS correlation.
- No em dashes in any generated prose.
