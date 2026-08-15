# ScholAR Experiment Ledger

This document is the research source of truth for experiments already run in ScholAR. It separates completed evidence from planned evaluation and records the caveats that must travel with every result. Stored numbers come from committed files under `evaluation/results/`, with the newer entailment-judge rescoring taking precedence over older cosine-proxy summaries.

## How to read this ledger

The repository contains two different ideas that were both called faithfulness during development:

1. **Retrieval-support CFS:** compares a known oracle claim with retrieved chunks using semantic similarity, whole-claim cosine, and key-fact overlap. This answers whether retrieval surfaced text similar to a known claim.
2. **Generation faithfulness:** decomposes an actual generated answer into claims and checks whether supplied evidence entails them. The latest version uses a local LLM judge and records contradictions.

They are not interchangeable. Retrieval-support CFS can remain high even when a generated answer makes unsupported claims.

## Evidence hierarchy

Use results in this order when claims conflict:

1. Independent expert human ratings, once completed
2. Entailment-judge rescoring with the negative-control calibration
3. Hand-labeled retrieval benchmarks
4. Auto-labeled or mined benchmarks
5. Cosine-based proxies and lexical answer proxies

The current project has no completed item at level 1.

## Experiment inventory

| Area | Dataset or sample | Status | Primary source |
|---|---|---|---|
| Retrieval anchor | 14 hand-labeled cases, 3 papers | Complete | `retrieval_eval_results.json` |
| Retrieval scaled | 100 auto-labeled cases, 25 papers | Complete | `retrieval_eval_results_scaled.json` |
| Retrieval-support CFS | 51 hand-labeled and 100 auto-labeled claims | Complete | `faithfulness_eval_results_v3.json`, `faithfulness_eval_results_scaled.json` |
| Generated-answer faithfulness | 350 answers across 4 models | Complete, judge not human-validated | `faithfulness_judged.json` |
| Faithfulness negative control | 20 true and deliberately corrupted claims | Complete | `faithfulness_negative_control.json` |
| Local baseline comparison | 91 shared cases, qwen3.5:9b | Complete | `comparison_faithfulness_judged.json`, `comparison_results.json` |
| Page support | 378 ScholAR and 283 PDF-chat cited pages | Complete, local judge | `page_correctness_results.json` |
| Bootstrap confidence intervals | 10,000 resamples | Complete | `confidence_intervals.json` |
| Visual routing pilot | 18 cases | Complete, limited proxy | `visual_eval_results.json` |
| Caption versus image ablation | 18 visual cases | Complete | `visual_caption_ablation_results.json` |
| Multi-document anchor | 18 cases, 10 arXiv-resolvable | Complete, small | `multidoc_eval_results.json` |
| Multi-document bounds | Same anchor | Complete | `multidoc_bounds_results.json` |
| M3SciQA localization | 297 labeled locality cases | Complete | `m3sciqa_results.json` |
| Abstention | 20 synthetic unanswerable cases per model | Complete, small | `abstention_results.json` |
| Efficiency | 20 questions per model | Complete on one laptop | `efficiency_results.json` |
| Human evaluation | 100 questions, 350 prepared answers | Instrument ready, annotation pending | `human_eval/` |

## 1. Retrieval

### Question

Which retriever most reliably returns the supporting chunk near the top of the ranking?

### Systems

- keyword overlap
- plain BM25
- BM25-primary with heuristic reranking
- BM25-primary with page hints
- dense all-MiniLM-L6-v2 retrieval
- hybrid BM25, dense, and reciprocal rank fusion in supporting experiments

### Scaled result: 100 cases across 25 papers

| System | R@1 | R@3 | R@5 | MRR | NDCG@5 |
|---|---:|---:|---:|---:|---:|
| Keyword overlap | 0.670 | 0.860 | 0.950 | 0.779 | 0.762 |
| BM25-only | 0.810 | 0.910 | 0.940 | 0.863 | 0.828 |
| BM25-primary | 0.810 | 0.910 | 0.930 | 0.861 | 0.824 |
| Dense-only | 0.470 | 0.650 | 0.740 | 0.572 | 0.523 |

Bootstrap 95% intervals for BM25-only are R@5 `[0.89, 0.98]` and MRR `[0.801, 0.919]`.

### Hand-labeled anchor: 14 cases across 3 papers

Dense-only performed best on the small anchor with R@5 `1.000` and MRR `0.881`. That direction reversed on the larger 25-paper set.

### Interpretation

Production remains BM25-primary because lexical retrieval is dependable, simple, and fast. The current heuristic layer does not improve aggregate retrieval over plain BM25. Dense-only performance on the tiny anchor was a small-sample artifact.

### Caveat

The 100 questions were mined from source passages and retain lexical overlap with them. This design likely favors lexical retrieval. Keep the 14-case hand-labeled set as a higher-precision anchor and do not claim universal BM25 superiority.

## 2. Retrieval-support CFS

### Question

Given a known claim, do the retrieved chunks contain semantically similar support and required key facts?

### Scaled result

| System | Combined CFS | Supporting-chunk hit at 5 | Labeled faithful |
|---|---:|---:|---:|
| BM25-primary | 0.785 | 0.860 | 93 / 100 |
| Hybrid BM25 + dense + RRF | 0.782 | 0.750 | 92 / 100 |

### Interpretation

Hybrid retrieval did not improve scaled retrieval support. This helped justify the simpler BM25-primary production path.

### Critical caveat

This metric uses cosine-based semantic components and known oracle claims. It is not evidence that generated answers are 78.5% faithful.

## 3. Faithfulness scorer negative control

### Question

Can the scorer distinguish real claims from deliberately corrupted versions that perturb a number, negate a relation, or reverse a result?

### Result

The current local entailment judge, `qwen3.5:9b`, was run on 20 true and corrupted claims:

| Metric | Result |
|---|---:|
| Mean score on true claims | 0.725 |
| Mean score on corrupted claims | 0.100 |
| Separation | 0.625 |
| Corrupted claims caught | 0.900 |
| Corrupted claims marked contradiction | 0.750 |
| True claims falsely marked unfaithful | 0.250 |
| True claims falsely marked contradiction | 0.100 |

The earlier cosine scorer caught 50% of corruptions and marked no contradiction. The judge is materially better for this purpose, but its 25% false-unfaithful rate also shows that it is not a gold standard.

### Interpretation

Use the judge for the current audit, not as a substitute for human validation. Report both sensitivity and false-positive behavior.

## 4. Generated answers across local models

### Question

If retrieval and citation mechanics remain fixed, how much do answer grounding and citation support vary by local generation model?

### Dataset

- 100 text, math, and visual questions across 25 papers
- visual questions run only on multimodal models
- 350 total answers
- 1,279 citation occurrences checked under the entailment judge

### Entailment-judge result

| Model | Answers | Mean faithfulness | Mean contradiction rate | Answers with a contradiction | Citation support |
|---|---:|---:|---:|---:|---:|
| gemma4:12b | 100 | 0.645 | 0.096 | 21 | 0.856 |
| llama3.1:8b | 75 | 0.607 | 0.140 | 15 | 0.835 |
| qwen3.5:9b | 100 | 0.594 | 0.094 | 30 | 0.643 |
| mistral:7b | 75 | 0.602 | 0.169 | 27 | 0.718 |

The macro mean generation faithfulness is approximately `0.61`. Ninety-three of 350 answers contain at least one contradicted atom.

### Superseded proxy result

The earlier cosine-based matrix reported faithfulness from `0.719` to `0.951`, with a macro mean around `0.85`. Keep the file for provenance, but do not use it as the current headline result.

### Interpretation

Faithfulness is modest and fairly flat across models. Citation support varies more, partly because models cite at very different rates. This does not establish strong model-agnostic grounding. It establishes that swapping the model did not repair the underlying support problem.

## 5. Resource-matched local comparison

### Question

How does ScholAR compare with local alternatives when the model and cases are held constant?

### Systems

- long-context PDF-chat
- vanilla RAG
- a local PaperQA2-style rerank, contextualize, and summarize path
- ScholAR

All systems use `qwen3.5:9b` on 91 shared cases.

### Latest judge result

| System | Judge generation faithfulness | Judge contradiction rate |
|---|---:|---:|
| PDF-chat | 0.439 | 0.029 |
| Vanilla RAG | 0.735 | 0.060 |
| PaperQA2-style | 0.779 | 0.111 |
| ScholAR | 0.453 | 0.179 |

### Other measured axes

| System | Must-include recall | Citation F1 |
|---|---:|---:|
| PDF-chat | 0.267 | 0.742 |
| Vanilla RAG | 0.340 | 0.802 |
| PaperQA2-style | 0.550 | 0.782 |
| ScholAR | 0.563 | 0.760 |

ScholAR versus PaperQA2-style paired bootstrap differences were not significant:

- must-include recall delta `+0.013`, 95% CI `[-0.104, 0.133]`;
- older cosine generation-faithfulness delta `+0.020`, 95% CI `[-0.020, 0.059]`;
- citation-F1 delta `-0.020`, 95% CI `[-0.068, 0.028]`.

### Interpretation

ScholAR retrieved enough information to match the best must-include answer recall, but it was less faithful under the stronger judge. Correctness recall and evidence grounding came apart. The original claim that ScholAR led local baselines on faithfulness was an artifact of the weaker scorer.

## 6. Page correctness

### Question

When a system cites a page, does that page support the attached claim?

| System | Cited pages checked | Supported | Page support rate |
|---|---:|---:|---:|
| PDF-chat | 283 | 202 | 0.714 |
| ScholAR | 378 | 249 | 0.659 |

### Interpretation

ScholAR's application-generated page mapping guarantees that the page exists and came from retrieved evidence. It does not guarantee that the page supports the claim. A free-form baseline scored slightly higher on support in this audit.

This result defines the next engineering target: claim-level support verification and repair.

## 7. Visual grounding

### Pilot

The 18-case pilot retrieved the expected figure or table in the top five for all cases and did not use caption fallback. This validates routing on the pilot, not visual answer correctness.

Earlier keyword-overlap answer-quality proxies should not be treated as visual reasoning metrics. They reward surface overlap and cannot prove the image was understood.

### Stronger evidence

The M3SciQA localization experiment below provides a cleaner image-value ablation because resolving the figure changes cross-document retrieval substantially.

## 8. Multi-document localization on M3SciQA

### Task

Given an anchor-paper question that refers to a figure, rank the anchor's bibliography to find the cited paper that contains the answer. The evaluation uses 297 labeled locality cases with a mean candidate pool of 47.9 references.

### Result

| System | MRR | R@5 |
|---|---:|---:|
| Random floor in our pools | 0.121 | not reported |
| Dense MiniLM | 0.125 | 0.158 |
| Hybrid RRF | 0.158 | 0.215 |
| BM25 text-only | 0.180 | 0.242 |
| Vision then BM25, gemma4:12b | 0.455 | 0.572 |
| Vision then BM25, qwen3.5:9b | 0.474 | 0.606 |

Published M3SciQA reference points include BM25 `0.127`, best reported open-source LMM `0.144`, text-embedding-3-large `0.297`, GPT-4V `0.400`, GPT-4o `0.500`, and human experts `0.796`.

### Interpretation

The question text alone often omits the entity needed to locate the cited paper because the entity appears only inside the figure. Resolving that figure locally before retrieval changes the query and lifts MRR from `0.180` to `0.474`.

This is the strongest positive systems result in the repository.

### Caveats

- ScholAR uses 297 labeled locality cases, while published baselines may use a different test split.
- Candidate-pool reconstruction differs slightly, which is why random floors are reported separately.
- The pipeline decomposes the task into vision resolution and lexical retrieval.
- Expert performance remains much higher. The task is not solved.

## 9. Abstention

Twenty paper-specific questions were paired with papers in which required facts were verified absent.

| Model | Abstention rate | Fabrication rate |
|---|---:|---:|
| qwen3.5:9b | 1.00 | 0.00 |
| llama3.1:8b | 1.00 | 0.00 |
| gemma4:12b | 0.95 | 0.05 |
| mistral:7b | 0.90 | 0.10 |

Bootstrap intervals are wide because `n=20`. Rare apparent failures included a benign fact overlap rather than unrestricted invention. Treat this as a tendency on a synthetic set.

## 10. Efficiency

Measured on one Apple Silicon laptop with 18 GB unified memory:

| Model | Loaded memory, GB | Tokens/s | Mean latency, s | P95 latency, s |
|---|---:|---:|---:|---:|
| mistral:7b | 6.5 | 24.9 | 6.1 | 8.4 |
| llama3.1:8b | 6.9 | 26.9 | 6.3 | 17.8 |
| qwen3.5:9b | 6.1 | 21.9 | 11.4 | 14.6 |
| gemma4:12b | 8.1 | 16.3 | 8.8 | 14.2 |

BM25 retrieval was approximately 40 ms, so end-to-end latency was generation-bound. These are single-machine measurements, not general hardware benchmarks.

## 11. Human evaluation

### What is ready

- 100 cases across 25 papers
- 350 generated answers
- blinded randomized HTML score sheet
- anchored rubric for relevance, coverage, faithfulness, and usefulness
- per-citation Supported, Partial, and Unsupported labels
- missing-citation capture
- aggregation, Friedman testing, inter-annotator agreement, and correlation code

### What is missing

- independent expert evaluators
- exported evaluator ratings
- agreement analysis
- human-versus-judge correlation
- adjudication of disagreements

### Completion standard

Use at least two evaluators, preferably three. Freeze the artifact versions before distribution. Report agreement and rater counts. De-identify committed outputs. Do not claim validation until actual ratings have been analyzed.

## 12. Research conclusions that are currently supportable

### Supported with caveats

- ScholAR is a working local-first scientific PDF study system with inspectable citation provenance.
- BM25 is a strong and efficient production retriever on the current benchmarks.
- The current reranking heuristics do not improve aggregate retrieval.
- Embedding cosine can substantially overstate generation faithfulness and miss contradictions.
- Evidence-ID grounding guarantees a real retrieved page, but not claim support.
- Local vision resolution substantially improves M3SciQA cross-document localization.

### Not supported

- evidence-ID grounding improves answer faithfulness;
- page-preserving chunking is itself a novel faithfulness method;
- ScholAR is more faithful than local vanilla RAG or PaperQA2-style baselines;
- visual answers are correct because visual routing succeeded;
- the automated entailment judge is human-validated;
- the results generalize to broad scholarly literature or production use.

## 13. Reproducing and extending results

Run from the repository root. See `evaluation/README.md` for detailed flags.

```bash
python3 evaluation/run_retrieval_eval.py \
  --cases evaluation/benchmark_cases_scaled.json --tag scaled
python3 evaluation/run_faithfulness_eval.py \
  --cases evaluation/faithfulness_cases_scaled.json --tag scaled
python3 evaluation/faithfulness_negative_control.py
python3 evaluation/m3sciqa/run_m3sciqa_eval.py --tier text
python3 evaluation/run_abstention_eval.py --model qwen3.5:9b
python3 evaluation/run_efficiency_eval.py --model qwen3.5:9b
```

Generation, judge, visual, and some multi-document runs require Ollama. Paper acquisition and dataset assembly may require network access.

When adding an experiment, record:

- research question;
- hypothesis before seeing results;
- exact command and Git revision;
- dataset and exclusions;
- model name and digest;
- prompt or scorer version;
- generation options and seeds;
- hardware and software environment;
- raw output path;
- aggregation method and uncertainty;
- failures, caveats, and whether the result changed the project decision.
