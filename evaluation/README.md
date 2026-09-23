# Evaluation and release tooling

This directory contains current benchmark inputs, runners, human-study templates, and
the fail-closed release pipeline. Generated outputs go to ignored
`evaluation/results/`; old outputs should not be restored as evidence.

## Entry point

```bash
./run_experiments.sh                 # strict-local deterministic smoke
./run_experiments.sh artifact-only   # plan only
./run_experiments.sh measured-retrieval
./run_experiments.sh model-backed --model <installed-tag>
./run_experiments.sh full --model <installed-tag>
```

Non-smoke profiles print their plan until `--execute` is supplied. Model-backed profiles
require a strict-local backend and loopback Ollama. Measured retrieval requires prepared
benchmark papers and cached text/visual encoders; required conditions fail closed instead
of silently degrading.

## Implicit visual retrieval

`run_visual_retrieval_eval.py` compares text-only, legacy CLIP page, ColQwen2 page,
and full hybrid retrieval on paired explicit/implicit questions. Its versioned input
records paper-disjoint splits, gold pages, optional normalized gold regions, visual type,
and whether the item is visual-only, visual-dominant, or mixed.

The runner rejects a non-disjoint test set by default, refuses to score a nominal
ColQwen2 condition when that model did not actually load, and reports paper-clustered
bootstrap intervals:

```bash
python evaluation/run_visual_retrieval_eval.py \
  --benchmark evaluation/releases/eacl_visual_v1/data_cards/implicit_visual_cases.json \
  --split test
```

## Current runners

| Script | Purpose |
|---|---|
| `run_retrieval_eval.py` | Retrieval R@1/3/5, MRR, and NDCG |
| `run_visual_retrieval_eval.py` | Explicit/implicit page and region retrieval with clustered intervals |
| `run_generation_faithfulness_eval.py` | Scores actual answers against their recorded context |
| `run_abstention_eval.py` | Answerable/unanswerable behavior |
| `run_efficiency_eval.py` | Latency, throughput, memory, and generation metadata |
| `run_comparison_eval.py` | Resource-matched local comparison |
| `run_multidoc_eval.py` | Cross-document localization |
| `run_multidoc_bounds_eval.py` | Oracle and random localization bounds |
| `m3sciqa/build_m3sciqa.py` | Converts an explicitly acquired M3SciQA checkout |
| `m3sciqa/run_m3sciqa_eval.py` | M3SciQA localization conditions |
| `spiqa/build_spiqa.py` | Validates and builds SPIQA multimodal scientific QA benchmark |
| `spiqa/run_spiqa_eval.py` | Multimodal visual retrieval and scientific answering benchmark (NeurIPS 2024) |
| `mine_cases.py` | Mines and source-checks candidate questions |
| `build_scaled_benchmark.py` | Builds scaled retrieval/faithfulness inputs |
| `build_abstention_benchmark.py` | Builds unanswerable controls |
| `benchmark_parsers.py` | Compares parser configurations |

`add_confidence_intervals.py` and `build_capability_breakdown.py` post-process fresh
results. `faithfulness_negative_control.py` probes scorer behavior on corrupted claims.

## Inputs

- `benchmark_cases.json` and `benchmark_cases_scaled.json`: retrieval cases.
- `faithfulness_cases.json` and `faithfulness_cases_scaled.json`: generation/scoring cases.
- `abstention_cases.json`: answerable and unanswerable controls.
- `multidoc_benchmark.json` and `multihop_rag_cases.json`: multi-document inputs.
- `spiqa/spiqa_cases_sample.json`: offline scientific multimodal QA cases (NeurIPS 2024).
- `human_eval/cases.json`: current human-study cases.

Downloaded dataset repositories are ignored.

## Result lifecycle

Normal runners write disposable artifacts to `evaluation/results/`. For release-quality
evidence:

1. `run_release_suite.py` records raw production `AnswerTrace` rows.
2. `score_release.py` creates scored rows without changing raw traces.
3. `score_human_gate.py` computes the predeclared support/coverage decision from adjudicated labels.
4. `aggregate_release.py` creates case-balanced aggregates and schema-v2 tables only after that gate passes.
5. `validate_release.py` checks schemas, checksums, trace-executed channels, source bundles, identities, provenance, and gates.

`reproduce_release_fixture.py` rebuilds the model-free toy fixture under
`fixtures/releases/release_v1_minimal/`. The real `releases/eacl_industry_v1/` skeleton
stays blocked until held-out data, human, ethics, model, hardware, and venue gates pass.

## Human study

`human_eval/` contains cases, rubric, answer/scoring utilities, and versioned templates
for claim annotation, judge validation, ethics review, and the researcher pilot. Raw
evaluator exports are local-only. Validate templates with:

```bash
.venv/bin/python evaluation/validate_human_templates.py
```

No toy fixture or empty template is empirical submission evidence.

## EACL protocol and held-out benchmark

The preregistration candidate is `protocols/eacl_industry_v1_protocol.json`. It
predeclares conditions S0--S5, primary metrics, paper-clustered paired bootstrap
inference, human-label requirements, and immutable failure accounting. It remains
`DRAFT` while dataset, model, prompt, hardware, and annotation identities are open.

The claim-bearing release config now uses schema v2 and encodes S0--S5 as typed retrieval controls. Measured
conditions reject automatic backend selection, a missing required page retriever,
feature-hashed dense retrieval, and heuristic reranker fallback. Each row binds the
corpus manifest, ingestion/chunking policies, retrieval controls, component and
generator identities, calibration hashes, exact device, prompt hashes, verifier,
and fallback policy. Validation reads channel execution from `AnswerTrace`; it does
not infer execution from environment settings. Scoring reads gold pages and regions only from the
frozen case file; citation relevance is distinct from citation-page provenance.
Aggregation treats papers as independent clusters, emits per-paper effects, and
applies Holm--Bonferroni correction to the declared paired family.

Visual verification is origin-aware. Source pixels remain the cited evidence;
the vision model's transcription is a linked derived artifact and is never added
to lexical or semantic support text. Pixel-only claims are labeled
`UNVERIFIED_VISUAL` for independent human review. `support_calibration.py` can fit
a strict-local semantic diagnostic only from a schema-validated development split;
raw calibration labels stay under the ignored `calibration/private/` directory.

```bash
make eacl-protocol-check
make eacl-heldout-audit
make eacl-heldout-ready  # intentionally fails until both artifacts are final
```

The legacy 200-question file is development-only. Its generated quality audit records
paper leakage, corpus coverage, missing adjudication fields, modality balance, and
answerability coverage. Only `audit_heldout_candidate.py --freeze-to ...` may create
the release case file, and it refuses to write when any readiness rule fails.
