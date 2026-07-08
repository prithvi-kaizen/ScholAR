# ScholAR Human Evaluation Pipeline

A model-agnostic, citation-grounded human evaluation for ScholAR. Each of 100 curated
questions is answered by several local models running the same ScholAR pipeline (only
the generation model changes), and expert evaluators score every answer on four 1-5
dimensions plus per-citation grading. Design and grounding are in `HUMAN_EVAL_DESIGN.md`;
the evaluator-facing instructions are in `rubric.md`.

## Files

| File | Role |
|---|---|
| `HUMAN_EVAL_DESIGN.md` | The design: instrument, 4-model structure, metrics, grounded in SciRAG/OpenScholar/PaperQA2 |
| `rubric.md` | The guideline handed to each human evaluator (Q1-Q7, anchors, worked examples) |
| `cases.json` | The 100 curated cases (40 single-doc, 20 visual, 20 multi-doc, 20 hard-retrieval) |
| `_build_cases.py` | Regenerates `cases.json` from the source benchmarks plus curated additions |
| `generate_answers.py` | Runs every case through each model via the live backend, writes `answers.json` |
| `_build_score_sheet.py` | Builds the self-contained `score_sheet.html` from cases + answers (blinded, randomized) |
| `compute_scores.py` | Aggregates evaluator exports into `human_eval_results.json` + `human_eval_report.md` |
| `BUILD_PROMPT.md` | The original build specification (kept for provenance) |

## Models

Text cases are answered by 4 models; visual cases by the 2 multimodal ones:

```
TEXT_MODELS   = qwen3.5:9b, gemma4:12b, llama3.1:8b, mistral:7b
VISION_MODELS = qwen3.5:9b, gemma4:12b
```

`qwen3.5:9b` and `gemma4:12b` are already pulled. Before a full run:

```
ollama pull llama3.1:8b mistral:7b
```

Edit the model lists at the top of `generate_answers.py` (and `VISION_MODELS`) to change the set.

## How to run

```bash
# 0. from the repo root, backend running with all papers prepared
make backend

# 1. (cases.json is already committed; regenerate only if you edit the curation)
python3 evaluation/human_eval/_build_cases.py

# 2. generate answers from every model (long: each answer is one local-model call)
#    use --only-installed to run a partial set before pulling all 4 models
python3 evaluation/human_eval/generate_answers.py            # -> answers.json

# 3. build the offline scoring interface
python3 evaluation/human_eval/_build_score_sheet.py          # -> score_sheet.html

# 4. hand rubric.md + score_sheet.html to each evaluator.
#    They open score_sheet.html in a browser, score every answer, click "Export scores".
#    Collect the exported *.json files into evaluation/human_eval/exports/

# 5. compute the results
python3 evaluation/human_eval/compute_scores.py              # -> human_eval_results.json + report.md
```

`score_sheet.html` is fully self-contained (data embedded, no network), autosaves to the
browser's localStorage, and shows the 4 answers per question in randomized blind order
(Answer A/B/C/D). The real model identity is restored only on export.

## What the results show

`compute_scores.py` reports, per model: mean Relevance/Coverage/Faithfulness/Usefulness,
citation precision/recall/F1, and the Supported/Partial/Unsupported distribution. It runs
a Friedman test across the models on faithfulness and citation-support (a non-significant
difference is the model-agnostic evidence), reports inter-annotator agreement when two or
more evaluator files are present, and correlates human faithfulness against the automated
NLI-CFS metric on the single-document cases that overlap the faithfulness benchmark.

Generated artifacts (`answers.json`, `score_sheet.html`, `exports/`, `human_eval_results.json`,
`human_eval_report.md`) are produced by the steps above and are not committed until a real
run is done.
