# ScholAR Dataset, Baseline, and Evaluation Plan

## 1. Executive decision

The paper should make one central claim:

> ScholAR is an auditable, locally deployable scientific question-answering system that improves grounded multi-paper reasoning under practical latency, memory, and reliability constraints.

The dataset is the measurement instrument for that claim, not the entire contribution. A benchmark-only story is less aligned with an Industry Track than a system-and-deployment story. The most recent concrete EACL Industry Track call emphasizes real-world implementation, application-relevant datasets, human-in-the-loop development, robustness, offline/online evaluation, latency, efficiency, scalability, and reproducibility [EACL 2026 Industry Track CFP](https://2026.eacl.org/calls/industry/).

As of 31 August 2026, EACL 2026 has already occurred. Its submission deadline was 17 November 2025 and the conference ran 24–29 March 2026. This plan therefore targets the next suitable EACL/ACL Industry Track; its final call and dates must be checked when announced.

No experimental design guarantees acceptance. The best acceptance strategy is to demonstrate:

1. **A real user need:** researchers need answers that can be verified against exact paper evidence, not merely fluent literature summaries.
2. **Technical novelty:** explicit multi-level, multi-document, multimodal evidence paths; source-scoped traceability; calibrated abstention; and evidence-aware repair.
3. **Realistic operation:** local/private inference, bounded RAM/VRAM, corpus-scale retrieval, p95 latency, throughput, and cost-per-grounded-answer.
4. **Credible evaluation:** independent gold labels, strong public and system baselines, human-calibrated LLM judging, statistical uncertainty, and predeclared primary outcomes.
5. **Reproducibility:** immutable corpus and dataset hashes, model digests, prompt hashes, raw traces, scoring-only reruns, and a complete data/limitations statement.

## 2. Research questions and predeclared claims

Define these before collecting final test results.

- **RQ1 — End-to-end quality:** Does ScholAR improve grounded answer quality over the strongest open scientific-QA baseline and a controlled hybrid-RAG baseline?
- **RQ2 — Multi-level reasoning:** How does performance change from direct lookup through multi-paper synthesis, and which stages fail at each level?
- **RQ3 — Grounding:** Does ScholAR retrieve all necessary evidence and attach correct, complete citations at paper, page, and evidence-span levels?
- **RQ4 — Selective reliability:** Does abstention reduce answer risk on insufficient, contradictory, or missing evidence without collapsing useful coverage?
- **RQ5 — Component value:** What do query decomposition, iterative retrieval, multimodal evidence, numeric execution, and verification/repair contribute independently?
- **RQ6 — Generalization:** Do gains hold outside CS/AI under domain-macro evaluation?
- **RQ7 — Industry utility:** What are the latency, memory, throughput, and cost trade-offs, and do users verify answers faster than with a conventional paper-search workflow?

Recommended primary hypotheses:

- **H1:** Full ScholAR has a higher domain-macro **Grounded Answer Rate** than the strongest open baseline.
- **H2:** Full ScholAR has higher **all-hop evidence recall** on L4–L5 questions than non-iterative hybrid RAG.
- **H3:** Selective verification reduces independently judged unsupported-claim rate while preserving answer correctness and reporting the associated coverage loss.
- **H4:** The local deployment meets a predeclared p95 latency and hardware envelope while improving cost per grounded answer.

Do not claim that the current evidence graph or numeric plan improves reasoning until it affects retrieval or generation. In the present implementation, the graph is primarily an audit artifact and the numeric result is not reliably fed into generation.

## 3. Benchmark scope

### 3.1 Release sizes

Use staged releases so quality is established before scale.

| Release | Purpose | Questions | Paper clusters | Approx. papers | Human status |
|---|---|---:|---:|---:|---|
| Pilot | Validate schema, workflow, and difficulty | 300 | 30–50 | 150–250 | All audited; 20% double-annotated |
| Development v0.5 | Build baselines and judge calibration | 1,500 | 100–150 | 500–800 | Dev/test audited |
| Publication v1.0 | Final benchmark | 6,000 | 300+ | 1,500+ | Entire test adjudicated; sampled train/dev audit |

Recommended v1 split:

- train: 3,600 cases (60%)
- development: 600 cases (10%)
- test: 1,200 cases (20%)
- private challenge test: 600 cases (10%), retained for contamination-resistant follow-up evaluation

Splits must be **citation-cluster-disjoint**, not merely question-disjoint. Near-duplicate papers, versions of the same preprint, shared benchmark reports, and citation families must remain in one split. Add a temporal slice where feasible, but do not use recency as the only contamination defense.

### 3.2 Domain allocation

It is not defensible to claim coverage of “every domain.” Claim broad cross-domain coverage and publish the exact taxonomy.

| Broad domain | Target share |
|---|---:|
| Computer science and AI | 55% |
| Biology and biomedicine | 12% |
| Finance and economics | 8% |
| Physics and mathematics | 7% |
| Engineering and materials | 6% |
| Climate and earth science | 5% |
| Social and behavioral science | 4% |
| Other audited scientific fields | 3% |

Report both micro averages and an equal-weight **domain macro average**. The non-CS test slice must be large enough for confidence intervals per broad domain; otherwise it only demonstrates anecdotal transfer.

### 3.3 Reasoning levels

Use the repository’s existing L1–L5 vocabulary, but define it operationally through required evidence rather than question wording.

| Level | Operational definition | Target share | Example requirement |
|---|---|---:|---|
| L1 | One explicit evidence unit | 10% | Find a reported dataset size |
| L2 | Two facts within one paper or one local inference | 20% | Compare two ablations in a table |
| L3 | Multiple sections/modalities or an executable calculation in one paper | 25% | Connect method detail to a figure result |
| L4 | Two papers and at least two necessary evidence hops | 30% | Compare a method with the cited baseline using aligned metrics |
| L5 | Three or more papers, contradiction resolution, or constrained synthesis | 15% | Explain why results disagree after accounting for data and evaluation setup |

A case is multi-hop only when removing any required hop makes the gold answer unsupported or materially incomplete. Query decomposition alone is not evidence of multi-hop reasoning.

### 3.4 Question and evidence dimensions

Stratify and tag every case along independent axes:

- **Answer form:** short fact, set/list, numeric, comparison, explanation, causal/mechanistic, methodological transfer, contradiction resolution, synthesis.
- **Evidence modality:** text, table, figure/chart, equation, mixed.
- **Corpus scope:** one paper, two papers, three or more papers.
- **Answerability:** answerable, insufficient evidence, conflicting evidence requiring qualified answer, missing source.
- **Reasoning operation:** lookup, join, compare, normalize units, calculate, temporal ordering, causal qualification, reconcile definitions, aggregate evidence.
- **Difficulty:** calibrated from baseline performance and human completion time, not assigned only by an LLM.

Recommended answerability distribution: 80–85% answerable and 15–20% unanswerable/qualified. Include natural negatives as well as minimally edited adversarial negatives. Keep the construction source in metadata so results can be broken down by negative type.

### 3.5 Corpus policy

1. Prefer openly redistributable papers: arXiv, PubMed Central open-access content, ACL Anthology, and other sources with clear licenses.
2. Record DOI/arXiv ID, title, authors, venue, year, domain, license, acquisition URL, retrieval date, and PDF SHA-256.
3. Preserve the exact PDF and ScholAR-derived bundle hash used for annotation.
4. Deduplicate preprint/published versions and retractions; mark corrections and withdrawals.
5. Build each cluster from a focal paper, cited/citing papers, and hard topical distractors.
6. Never distribute full text where the license does not allow it. For restricted sources, distribute metadata, IDs, annotations permitted by license, and a reconstruction script.
7. Keep train/dev/test corpus manifests immutable after freeze.

## 4. Typed benchmark record

Extend `evaluation/release/schemas.py`; do not hide gold labels in free-form metadata.

```json
{
  "case_id": "scholar_v1_bio_L4_000123",
  "dataset": "scholarbench_v1",
  "split": "test",
  "cluster_id": "cluster_0042",
  "domain": "biology_biomedicine",
  "reasoning_level": "L4",
  "question_type": ["comparison", "numeric_normalization"],
  "answerable": true,
  "question": "...",
  "paper_id": "primary_id",
  "secondary_paper_ids": ["source_2", "source_3"],
  "distractor_paper_ids": ["distractor_1"],
  "gold_answers": [
    {
      "answer_type": "structured",
      "canonical_text": "...",
      "aliases": ["..."],
      "key_points": ["kp1", "kp2"],
      "numeric_value": null,
      "unit": null,
      "tolerance": null
    }
  ],
  "gold_evidence": [
    {
      "hop_id": "H1",
      "source_paper_id": "source_2",
      "evidence_id": "E_042",
      "page": 6,
      "section": "Results",
      "bbox_norm": [0.08, 0.15, 0.92, 0.36],
      "char_start": 0,
      "char_end": 214,
      "quote_sha256": "...",
      "supports_key_points": ["kp1"]
    }
  ],
  "reasoning_graph": {
    "nodes": ["H1", "H2"],
    "edges": [{"from": "H1", "to": "H2", "relation": "requires"}],
    "ordered": true
  },
  "numeric_program": null,
  "annotation": {
    "construction": "evidence_first_llm_assisted",
    "annotator_ids": ["anon_a", "anon_b"],
    "adjudicator_id": "anon_c",
    "adjudication_status": "accepted",
    "difficulty": "hard"
  },
  "provenance": {
    "dataset_version": "1.0.0",
    "corpus_sha256": "...",
    "case_sha256": "..."
  }
}
```

Required typed models:

- `GoldAnswer`
- `GoldKeyPoint`
- `GoldEvidence`
- `ReasoningHop` and `ReasoningDependency`
- `NumericGold` with operation, operands, units, tolerance, and expected value
- `AnnotationProvenance`
- `CaseRecord` fields for domain, split, level, answerability, gold labels, distractors, and cluster ID

Stable joins must use `source_paper_id::local_id_kind::local_id`, corpus hash, page, and quote hash. Temporary prompt IDs such as `E1` must never be gold identifiers.

## 5. Dataset creation workflow

### Stage A — Paper acquisition and cluster construction

1. Sample domains and venues against the allocation table.
2. Select focal papers with accessible PDFs and sufficient citation context.
3. Create clusters of 3–10 relevant papers plus hard distractors.
4. Ingest every PDF through `PaperFinalizeService` and store parser engine, degraded mode, page count, evidence AST hash, chunk hash, and visual-unit hash.
5. Reject clusters with missing pages, unusable extraction, uncertain licensing, duplicated paper versions, or unresolved identity.

### Stage B — Evidence-first candidate generation

Generate questions from verified evidence bundles, not by asking an LLM to invent a question from paper titles.

1. Sample an intended level, operation, answer form, and modality.
2. Select the exact evidence units and construct the minimal evidence dependency graph.
3. Derive a canonical answer and key points from those units.
4. Ask a generator model to draft a natural question that requires the selected evidence.
5. Ask an independent critic model to attempt the question from each strict subset of the evidence.
6. Retain a multi-hop case only if the full evidence supports the answer and the required leave-one-hop-out subsets do not.
7. Generate topical distractors and verify that they do not independently answer the question.
8. Store generator/critic model IDs, prompt hashes, seeds, and timestamps for provenance; do not expose these fields to evaluated systems.

Use at least two independent candidate-generation models or prompt families. Otherwise the benchmark may favor the linguistic and reasoning patterns of one model family.

### Stage C — Automatic quality gates

A candidate must pass all applicable gates:

- schema and referential integrity
- PDF/bundle/license integrity
- answer-evidence entailment
- evidence completeness for every key point
- multi-hop necessity through leave-one-hop-out tests
- no single-paper shortcut for L4–L5
- numeric recomputation and unit validation
- cited table/figure existence and region validity
- ambiguity and multiple-valid-answer detection
- near-duplicate question and evidence-path detection
- title/abstract-only shortcut detection
- train/test citation-family leakage detection
- answer string leakage from question text
- distractor non-support
- prompt-injection and malformed-content scan

Automatic checks nominate cases; they do not make the final test set gold.

### Stage D — Human audit and adjudication

For the final test set:

1. Annotator A answers from the supplied papers and marks evidence without seeing the generated answer.
2. Annotator B independently validates the question, answer, key points, evidence path, answerability, and level.
3. A domain-capable adjudicator resolves disagreements.
4. Reject rather than repair deeply ambiguous cases; preserve rejected-case reasons for the data statement.
5. Double-annotate at least all L4/L5, all non-CS cases used for domain claims, all unanswerable cases, and every human-evaluation item.

For train/dev, audit a stratified sample from every domain × level × modality cell. Expand auditing if any cell fails the target quality threshold.

### Stage E — Difficulty calibration and freeze

Run baseline models only after candidate labels are fixed. Use baseline accuracy, all-hop recall, and human completion time to estimate difficulty. Do not remove cases solely because ScholAR fails them. Freeze:

- `dataset.jsonl` and SHA-256
- corpus manifest and SHA-256
- split/cluster manifest
- schema version
- annotation guideline version
- rejected-case statistics
- model and prompt identities used during construction

## 6. Integration into the ScholAR pipeline

### 6.1 Preserve one measured answer path

Continue using:

`CaseRecord → run_release_suite → run_scholar_http → AnswerPipelineService → AnswerTrace → raw rows → score-only rows`

Do not create a separate benchmark answer implementation. Raw traces must remain immutable and scoring must be rerunnable without generation.

### 6.2 Required repository changes

#### `evaluation/release/schemas.py`

- Add the typed gold models above.
- Add dataset, split, domain, level, cluster, answerability, distractor, and gold fields to `CaseRecord`.
- Add retrieval condition, corpus-search condition, context budget, parser condition, and visual backend to typed `SystemOptions`.
- Add an independently versioned judge/scorer identity.

#### `evaluation/run_release_suite.py`

- Pass all frozen retrieval and visual options into `run_scholar_http`.
- Fail closed if any required primary/secondary paper bundle is missing; the current pipeline silently skips missing secondary bundles.
- Record cold/warm-cache state and corpus-index identity.
- Support three evaluation modes:
  1. **oracle evidence** — gold evidence is supplied to estimate the reader upper bound;
  2. **oracle paper set** — correct papers are supplied, but evidence retrieval is measured;
  3. **corpus retrieval** — the system must retrieve papers and then evidence.

#### `evaluation/release/scoring.py`

Make scoring case-aware. Add deterministic answer, document/evidence, citation, path, abstention, robustness, judge, and efficiency metrics. The current trace-derived lexical support rate may remain a diagnostic, but it cannot serve as independent faithfulness evaluation because the same verifier repairs the output.

#### `backend/services/answer_pipeline.py`

Add a strict evaluation mode that:

- disables or separately labels the current first-two-anchor-chunk `CONTEXT_PRELUDE` behavior;
- fails when required sources are absent;
- records document retrieval separately from within-document evidence retrieval;
- executes hop-conditioned retrieval rather than only fusing independently generated subqueries;
- passes validated evidence paths and numeric execution results into generation when those components are claimed;
- records confidence or abstention score before thresholding.

#### `backend/schemas/answer_trace.py`

Add, if used in the paper:

- document-retrieval traces
- hop-by-hop query/state transitions
- calibrated confidence
- explicit cache state
- monetary cost identity and amount, or local energy/runtime accounting
- corpus index/version hashes

Do not estimate monetary cost from latency. For hosted models, freeze provider, model version, request usage, tariff date, and currency. For local systems, report energy if measured and amortized hardware assumptions separately.

### 6.3 Corpus retrieval is a necessary end-to-end condition

The current request names the primary and secondary paper IDs. This evaluates evidence retrieval and synthesis inside an oracle paper set, not realistic literature search. The paper should clearly separate:

- **reader quality:** oracle evidence;
- **evidence retriever quality:** oracle paper set;
- **end-to-end application quality:** corpus-scale paper retrieval followed by evidence retrieval and answering.

Implement a first-stage paper index over title, abstract, metadata, citation links, and optionally full-text representations. Evaluate document Recall@k before passage/evidence metrics. If corpus retrieval cannot be completed, narrow the paper’s claim to “QA over a user-selected paper collection.”

## 7. Baselines

All controlled retrieval baselines should use the same generator, prompt budget, paper corpus, and decoding settings. Otherwise model quality is confounded with retrieval quality.

| ID | Baseline | Purpose |
|---|---|---|
| B0 | Closed-book generator | Measures memorization and question leakage |
| B1 | BM25 top-k + shared generator | Lexical retrieval floor |
| B2 | Dense top-k + shared generator | Semantic retrieval baseline |
| B3 | Hybrid BM25+dense+RRF/reranker + shared generator | Strong controlled RAG baseline |
| B4 | Long-context oracle paper set + shared generator | Tests retrieval versus context stuffing |
| B5 | Oracle gold evidence + shared generator | Reader/generation upper bound |
| B6 | Strong open scientific-QA agent, preferably PaperQA2 or OpenScholar where runnable | External system baseline |
| B7 | Frontier hosted RAG/agent reference, if policy and budget permit | Non-local quality ceiling; report version and cost |
| S0 | Current ScholAR without iterative hop control | Honest repository baseline |
| S1 | Full ScholAR without verification/repair | Isolates verifier effect |
| S2 | Full ScholAR | Proposed system |

Relevant scientific-QA comparisons include [PaperQA2/LitQA2](https://arxiv.org/abs/2409.13740), [OpenScholar/ScholarQABench](https://arxiv.org/abs/2411.14199), [M3SciQA](https://arxiv.org/abs/2411.04075), [PaperArena](https://arxiv.org/abs/2510.10909), and recent multi-stage grounding work such as [LitTraceQA](https://arxiv.org/abs/2608.07370). Use a baseline only if its released artifacts and license permit a faithful run; otherwise explain the omission rather than reimplementing an approximation under the same name.

Run external validation on official public datasets where licenses permit. At minimum, include one single-paper benchmark and one multi-document/multimodal benchmark. The repository’s current hard-coded QASPER, PeerQA, and SciVQA examples are smoke tests, not valid official benchmark results.

## 8. Ablation matrix

Use a compact, additive ablation that maps directly to claims:

1. lexical retrieval only
2. + dense retrieval and reranking
3. + query decomposition
4. + hop-conditioned iterative retrieval
5. + evidence-path-aware generation
6. + table/figure/equation handling
7. + deterministic numeric execution
8. + verification and selective repair
9. full system + calibrated abstention

Additional controlled tests:

- parser P0–P4 or Docling versus degraded PyMuPDF mode
- text-only versus multimodal
- small versus large context budget
- one, two, and three or more papers
- index sizes such as 100, 1,000, and full corpus
- local model sizes/quantizations under identical retrieval
- no distractors versus hard distractors
- no corruption versus counterfactual number/entity corruption

Do not include an “evidence graph ablation” until removing the graph changes executed behavior.

## 9. Metrics

### 9.1 Primary outcomes

Avoid an opaque weighted composite. Use a transparent pass criterion and report its components.

**Grounded Answer Rate (GAR):** percentage of all expected cases for which:

- the answer is correct or substantially correct;
- every required key point is present;
- citation precision and citation recall each exceed a predeclared threshold;
- no major claim is contradicted by the cited evidence;
- an answerable case is not abstained.

Report GAR macro-averaged across domain and reasoning level, with a 95% cluster-bootstrap confidence interval. Also publish continuous component metrics.

### 9.2 Retrieval and evidence

- document Recall@1/5/10 and MRR
- evidence Recall@1/3/5/10 and NDCG@10
- **all-hop recall@k:** every required hop represented
- per-hop recall and source coverage
- ordered/unordered evidence-path precision, recall, and F1
- gold evidence selected for context
- gold evidence actually shown to the generator
- page localization accuracy
- figure/table region IoU where gold boxes exist
- retrieval performance as corpus size grows

Filter `CONTEXT_PRELUDE` entries from ranked retrieval metrics and report them separately.

### 9.3 Answer correctness

Use type-aware deterministic metrics first:

- normalized exact match and token F1 for short answers
- set precision/recall/F1 for list answers
- numeric accuracy with unit normalization and case-specific tolerance
- key-point precision/recall/F1 for explanatory answers
- contradiction-resolution correctness for qualified answers
- LLM/human rubric score only where deterministic scoring is inadequate

Replace the current substring “EM” and token-set F1 in the smoke multi-hop adapter with benchmark-standard normalization and multiset-aware counts.

### 9.4 Citation and faithfulness

- citation precision/recall/F1 at paper, page, and evidence-unit levels
- citation correctness: cited evidence entails the associated claim
- citation completeness: supported factual claims have citations
- claim precision and unsupported-claim rate
- contradiction rate
- citation-source diversity where multiple papers are required
- application-imputed citation rate, reported separately and excluded from model citation-generation quality

Score `raw_answer` and `final_answer` separately. The difference quantifies repair benefit and any correctness loss caused by repair.

### 9.5 Abstention and calibration

- answerable accuracy with abstentions counted as incorrect
- unanswerable detection precision, recall, F1, and balanced accuracy
- selective risk at fixed coverage points
- risk–coverage curve and area under that curve
- expected calibration error and Brier score if a confidence value is emitted
- abstention reason by pipeline stage

Never report accuracy only on answered cases; that rewards excessive abstention.

### 9.6 Robustness

- domain and reasoning-level macro results
- paraphrased questions
- hard distractor injection
- missing-hop and conflicting-evidence conditions
- number, entity, and relation counterfactuals
- parser degradation/OCR failures
- figure/table-only evidence
- citation-cluster and temporal transfer
- small-to-large corpus scaling

### 9.7 Efficiency and deployment

- cold and warm p50/p95 end-to-end latency
- p50/p95 per-stage latency
- throughput at concurrency 1 and a realistic concurrent load
- prompt/output tokens and tokens per second
- peak RAM, VRAM, model disk size, and index size
- preprocessing/index-build time
- energy per query if measured
- hosted API cost per query if used
- **cost per grounded answer:** total measured cost divided by GAR successes
- failure and timeout rate

Freeze hardware, OS, runtime, model digest, quantization, thread count, cache state, and corpus size.

## 10. LLM-as-a-judge protocol

LLM judging should scale evaluation, not define truth. Deterministic metrics and human-adjudicated gold evidence remain primary wherever possible.

### 10.1 Judge inputs

For each case, provide:

- question and answerability label
- canonical answer and key points
- exact gold evidence snippets/pages
- candidate final answer and its cited evidence
- type-specific rubric

Hide the system name, model name, retrieval scores, and internal verifier labels. Do not provide hidden chain-of-thought. Treat papers and candidate text as untrusted quoted data so embedded instructions cannot control the judge.

### 10.2 Decomposed rubric

Use separate structured judgments rather than one holistic score:

- correctness: 0–4
- key-point completeness: 0–3
- citation correctness: 0–4
- citation completeness: 0–3
- faithfulness/unsupported claims: categorical plus claim count
- answerability/abstention appropriateness: 0–2
- practical usefulness: 0–3, secondary only

Require strict JSON and a short evidence-linked rationale. Validate outputs against a schema and fail invalid judge responses rather than silently coercing them.

### 10.3 Pointwise and pairwise use

- Use **pointwise grading** against gold for correctness, completeness, and faithfulness.
- Use **pairwise preference** only for overall usefulness/readability or comparisons that lack an objective gold score.
- For pairwise judging, evaluate A/B and B/A, randomize labels, allow ties, and aggregate only position-consistent results.
- Use multiple trials or multiple judges for high-impact comparisons and report judge variance.

LLM judges can show position and other systematic biases; position swapping and judge validation are therefore mandatory. Foundational evidence and known limitations are discussed in [MT-Bench/Chatbot Arena judge evaluation](https://arxiv.org/abs/2306.05685) and [position-bias analysis](https://arxiv.org/abs/2406.07791).

### 10.4 Judge independence and freezing

- The primary judge must not be the same model/version used to generate the candidate answer or create its gold labels.
- Use one strong primary judge and one different-family audit judge.
- Freeze provider/model version or model digest, prompt, rubric, seed, temperature, decoding, and execution date.
- Run a minimum of three balanced permutations/trials for the human-calibration subset.
- Preserve all judge responses and failures for scoring-only reanalysis.

### 10.5 Human calibration gate

Before using judge scores at scale, compare them with human labels on the stratified human subset.

Report:

- quadratic-weighted kappa for ordinal scores
- Krippendorff’s alpha across human and judge labels
- Spearman correlation for ranking
- pairwise agreement and chance-corrected kappa
- confusion matrices and error rates by domain, level, answer length, and candidate model
- bootstrap confidence intervals

Do not validate with raw percent agreement alone. A reasonable predeclared starting gate is weighted kappa at least 0.65 overall, with no major domain or reasoning stratum below 0.55 and a lower confidence bound above 0.50. Final thresholds should be set after a pilot and before final system evaluation. If the judge fails, use it only as a diagnostic or retrain/rewrite the rubric and revalidate on a new held-out calibration subset.

### 10.6 Bias and sensitivity probes

Measure:

- position bias
- verbosity/length bias
- self-family preference
- style and citation-count preference
- sensitivity to intentionally corrupted answers
- sensitivity to missing or irrelevant citations
- robustness to prompt injection in candidate text
- stability across trials

## 11. Human evaluation

### 11.1 Output-evaluation sample

A defensible initial target is:

- 300 test questions, stratified by domain × L1–L5 × modality × answerability
- four selected systems: strongest controlled baseline, strongest external baseline, ScholAR without repair, and full ScholAR
- 1,200 outputs total
- two independent ratings per output
- third-rater adjudication for disagreements and at least 20% overlap across the full rater pool

This is small relative to the full judged experiment but large enough to calibrate the judge and estimate system differences. Conduct a pilot-based power analysis and increase the sample if the minimum important effect is not detectable.

### 11.2 Raters and procedure

- Use domain-capable graduate researchers or professionals for specialized non-CS questions.
- Train raters with written guidelines and 20–30 shared calibration cases.
- Blind system identity and randomize output order.
- Show source PDF pages/evidence, not only system-selected snippets.
- Ask for correctness, key-point completeness, citation correctness/completeness, unsupported claims, abstention appropriateness, and usefulness.
- Record completion time and uncertainty.
- Include attention checks that test the rubric rather than obscure trivia.
- Report expertise, recruitment, compensation, time, exclusions, and adjudication policy.
- Seek institutional ethics/IRB guidance where required, particularly for logging real user interactions.

Analyze ordinal scores using a mixed-effects ordinal model with system/domain/level fixed effects and question/rater random effects. Analyze pairwise preferences with a Bradley–Terry-style model or paired cluster bootstrap.

### 11.3 Real-world user pilot

For a strong Industry Track fit, add a small deployment study rather than only offline output grading:

- 10–20 researchers across at least CS/AI and two non-CS domains
- 2–4 weeks or a controlled crossover task
- compare ScholAR with the existing literature-search workflow
- measure task completion, time to verified answer, number of opened papers, citation correction rate, trust calibration, useful abstentions, and critical failures
- collect qualitative failure and maintainability feedback

The strongest practical metric is **time to a verified answer**, not subjective satisfaction alone.

## 12. Statistical analysis

1. Treat the citation cluster, not each generated question, as the independent resampling unit.
2. Use paired cluster bootstrap confidence intervals with at least 10,000 resamples for primary system comparisons.
3. Report absolute differences, relative differences where meaningful, confidence intervals, and effect sizes.
4. Use paired tests because systems answer the same cases; use permutation tests or Wilcoxon signed-rank tests as appropriate.
5. Correct families of secondary comparisons with Holm’s procedure.
6. Macro-average domains and reasoning levels; also show sample counts and micro results.
7. Run at least three generation seeds for the principal systems, or demonstrate deterministic stability. Separate model variance from case variance.
8. Predeclare the primary metric, strongest baseline, exclusion rules, judge gate, and stopping rule before the final test run.
9. Never tune retrieval thresholds, prompts, judge rubrics, or abstention thresholds on test labels.
10. Publish negative and null results for components that do not help.

## 13. Baseline and evaluation phases

### Phase 0 — Instrumentation gate

**Goal:** prove one case can be frozen, run, traced, and rescored.

Deliverables:

- typed case schema and validators
- one immutable raw row per case/system/model/seed
- case-aware scoring
- strict missing-source behavior
- separate retrieved/selected/shown evidence metrics
- model, corpus, prompt, and Git identity

Exit gate: 100 pilot cases replay deterministically with no identity mismatch and no unreported fallback.

### Phase 1 — Dataset pilot

**Goal:** validate question construction and annotation.

- 300 cases
- full L1–L5 and domain matrix
- double annotation sample
- leave-one-hop-out tests
- baseline-neutral difficulty analysis

Exit gate: at least 90% of audited cases accepted without substantive gold correction, strong annotation agreement, and no empty domain/level cells.

### Phase 2 — Baseline metric phase

**Goal:** establish honest floors, upper bounds, and bottlenecks before changing ScholAR.

Run B0–B5 and S0 on the pilot/development set. Produce:

- document retrieval table
- evidence/all-hop retrieval table
- answer/citation table
- abstention/risk-coverage plot
- latency/memory/cost table
- domain × level heatmap
- failure taxonomy

Exit gate: metrics discriminate closed-book, retrieval, and oracle conditions; oracle evidence substantially exceeds retrieved evidence; scorer negative controls behave as expected.

### Phase 3 — System implementation and ablation

Implement and test components in claim order:

1. strict benchmark context behavior
2. corpus paper retrieval
3. iterative hop-conditioned evidence retrieval
4. evidence-path-aware generation
5. numeric execution integration
6. multimodal integration
7. independent verification and calibration

Exit gate: every claimed component changes executed behavior, has a trace field, and has an ablation.

### Phase 4 — Judge calibration and human study

- freeze rubric
- collect human ratings
- measure inter-rater agreement
- validate judge and run bias probes
- freeze accepted judge configuration

Exit gate: judge passes predeclared agreement and subgroup gates; otherwise human/deterministic results remain primary.

### Phase 5 — Frozen final evaluation

- freeze dataset and corpus
- freeze systems, models, prompts, seeds, hardware, and thresholds
- run all canonical keys exactly once per seed
- retain success, abstention, timeout, and error rows
- score from immutable raw traces
- run statistical analysis and robustness subsets

Exit gate: no missing expected keys, no dirty/unidentified release, and all tables regenerate from committed configuration plus permitted artifacts.

### Phase 6 — Deployment evidence and paper freeze

- complete user pilot
- finish data statement, model/system card, ethics, and Limitations
- anonymize code/artifact links
- reproduce headline tables on a clean machine
- lock claims to measured evidence

## 14. Suggested implementation sequence

| Weeks | Workstream | Output |
|---|---|---|
| 1–2 | Schema, corpus manifest, annotation guide | Typed records and pilot protocol |
| 3–4 | Candidate generator and automatic gates | 300-case pilot |
| 5 | Human pilot audit and schema revision | Frozen pilot v0.1 |
| 6–7 | Gold-aware scoring and strict release runner | End-to-end baseline harness |
| 8–9 | Controlled baselines and negative controls | Baseline report |
| 10–11 | Iterative retrieval and evidence-path execution | Main system ablations |
| 12 | Multimodal/numeric and efficiency instrumentation | Reliability/efficiency runs |
| 13 | Judge calibration and human output study | Validated judge configuration |
| 14 | Dataset scale-up/final adjudication | Frozen v1 test |
| 15 | Final runs and statistics | Headline tables/figures |
| 16 | User-study analysis, artifact audit, paper | Submission package |

Scale-up and system work can overlap only after the pilot schema and guidelines are frozen.

## 15. EACL Industry Track paper structure

The last concrete EACL Industry Track format allowed six content pages and required a dedicated Limitations section [official CFP](https://2026.eacl.org/calls/industry/). Verify the future venue’s requirements.

Recommended six-page narrative:

1. **Introduction and deployment problem** — verifiable literature QA under privacy/compute constraints.
2. **ScholAR system** — document/evidence retrieval, hop controller, multimodal reasoning, provenance, verification, abstention.
3. **ScholARBench** — evidence-first construction, domain/level matrix, human quality control, frozen splits.
4. **Experimental setup** — strongest baselines, primary metrics, human-calibrated judge, hardware.
5. **Results** — one main quality table, one level/domain plot, one ablation/efficiency figure.
6. **Deployment findings and conclusion** — time-to-verified-answer, failure modes, cost/reliability trade-offs.

Place annotation details, full prompts/rubrics, extra tables, schemas, examples, and robustness results in the appendix/supplement, but keep the central method and primary evidence self-contained.

### What will make the paper distinctive

- evidence dependency graphs are gold annotations, not post-hoc prose
- all-hop necessity is tested, not inferred from question length
- multimodal and numeric evidence is evaluated at region/program level
- retrieval, shown context, answer, citation, repair, abstention, and efficiency are measured separately
- local/private deployment is compared on quality–latency–cost, not quality alone
- LLM judging is independently human-calibrated and bias-tested
- non-CS results are macro-averaged with real sample sizes
- a field pilot measures time to verified evidence

### Likely reviewer objections to preempt

- **“Synthetic questions favor the generating model.”** Use multiple generators, human-adjudicated test data, evidence-first construction, and generator-family breakdowns.
- **“Multi-hop is not actually necessary.”** Publish evidence graphs and leave-one-hop-out validation.
- **“The system is given the papers.”** Include corpus retrieval or narrow the claim explicitly.
- **“The verifier scores itself.”** Use independent gold/human/judge faithfulness scores; keep internal verifier scores diagnostic.
- **“LLM-as-judge is unreliable.”** Apply human calibration, order swaps, multiple trials, subgroup gates, and chance-corrected agreement.
- **“Cross-domain is superficial.”** Use quotas, domain experts, macro metrics, and confidence intervals.
- **“Industry impact is hypothetical.”** Report a deployment pilot, practical constraints, p95 latency, failure rate, and time to verified answer.
- **“The benchmark is contaminated.”** Use cluster-disjoint and private/temporal test slices, hashes, and closed-corpus evidence requirements.
- **“The comparison is unfair.”** Share the generator/context budget for controlled baselines and separately label external systems.

## 16. Reproducibility, ethics, and release package

Release where licensing permits:

- dataset cards and domain/level statistics
- train/dev annotations and either test labels or a scoring server/private test protocol
- corpus manifest, acquisition scripts, and hashes
- annotation guideline and adjudication form
- candidate-generation and judge prompts with hashes
- anonymized human-evaluation forms and aggregate labels
- exact release configs, model digests, seeds, and hardware profile
- raw traces, scored rows, table-generation scripts, and negative controls
- failure taxonomy and rejected-case counts
- data statement, model/system card, ethics discussion, and Limitations

Follow the current [ARR Responsible NLP Research guidance](https://aclrollingreview.org/responsibleNLPresearch/) and the target venue’s final checklist. Discuss copyright/licensing, annotator treatment, domain risks, medical/financial non-advice, automation bias, citation misuse, paper-author representation, model/provider environmental cost, and the limits of treating publication as truth.

## 17. Minimum viable submission versus strong submission

### Minimum viable

- 1,500 high-quality cases
- CS/AI plus at least three non-CS domains
- cluster-disjoint test split
- BM25/dense/hybrid/oracle/current-ScholAR baselines
- deterministic answer and citation metrics
- 200–300-case human calibration subset
- latency/memory report
- complete ablations and reproducible release

### Strong Industry Track submission

- 6,000-case release with 1,200 fully adjudicated test cases
- corpus-scale first-stage paper retrieval
- genuine iterative multi-hop execution
- strong external scientific-agent baseline
- validated multimodal and numeric subsets
- calibrated abstention and risk–coverage analysis
- human-calibrated, bias-tested LLM judging
- researcher deployment pilot with time-to-verified-answer
- quality–latency–cost Pareto analysis

The final go/no-go decision should be based on whether ScholAR beats the strongest controlled and open baseline on grounded answer quality with statistically supported gains, while offering a defensible practical advantage in privacy, traceability, latency, cost, or user verification time. If it does not, publish the benchmark and failure analysis only after reframing the contribution and choosing a venue whose scope matches that result.

---

Content informed by web sources was paraphrased for compliance with licensing restrictions.