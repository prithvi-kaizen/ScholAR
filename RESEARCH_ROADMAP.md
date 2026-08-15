# ScholAR Research Roadmap

This roadmap is venue-neutral. It records what is complete, what is only partially validated, and what must be finished before the next submission. Priorities are ordered by research value and dependency, not by feature novelty.

Status labels:

- **Done:** implemented, run, and documented with traceable output
- **Partial:** implementation or evidence exists, but validation is insufficient
- **Next:** highest-priority active work
- **Later:** valuable after the central evidence gap is closed
- **Parked:** do not invest until the research direction requires it

## Current research position

The original hypothesis was that page-preserving chunks and indirect evidence-ID citations would improve faithfulness. Stronger evaluation did not support that claim. The mechanism guarantees that a citation maps to a real retrieved page, but the page supports its attached claim only 65.9% of the time. In a resource-matched local comparison, ScholAR was less faithful under the entailment judge than vanilla RAG and a PaperQA2-style pipeline.

The project still has two credible contributions:

1. a measurement result showing that cosine proxies materially inflate generation faithfulness and miss contradictions;
2. a positive local vision result on M3SciQA cross-document localization, where MRR rises from 0.180 to 0.474.

The next milestone is therefore evidence repair and human validation, not submission formatting.

## Milestone 1: foundation and product loop

**Status: Done**

- [x] Search arXiv and cache results locally
- [x] Upload and validate PDFs
- [x] Extract text page by page with PyMuPDF
- [x] Store inspectable paper artifacts as local JSON
- [x] Render pages in a split-screen reading workspace
- [x] Generate paper-specific study goals
- [x] Answer questions through a local Ollama model
- [x] Return page-linked citations and quotes
- [x] Provide deterministic fallbacks when the model is unavailable

## Milestone 2: retrieval and citation infrastructure

**Status: Done, with a known research limitation**

- [x] Page-preserving chunks with page and section metadata
- [x] BM25-primary retrieval
- [x] Query expansion, page hints, section hints, and explicit figure pinning
- [x] Evidence-ID citation generation and backend normalization
- [x] Multi-paper chunk provenance through `source_paper_id`
- [x] 14-case hand-labeled retrieval benchmark
- [x] 100-case, 25-paper scaled benchmark
- [x] Dense and hybrid retrieval comparisons
- [x] Bootstrap confidence intervals

Decision: keep BM25 primary. Current reranking heuristics provide no aggregate gain and dense-only retrieval does not scale on the mined benchmark.

## Milestone 3: multimodal and multi-document support

**Status: Partial**

- [x] Extract figure and table regions
- [x] Add figure chunks to the retrieval pool
- [x] Route rank-one visual evidence to a local multimodal model
- [x] Resolve bibliographies with Semantic Scholar and arXiv
- [x] Ingest selected open-access references
- [x] Merge anchor and secondary paper chunks safely
- [x] Evaluate M3SciQA localization on 297 cases
- [ ] Replace implicit rank-one vision routing with an explicit routing decision
- [ ] Evaluate visual answer correctness with human or task-specific labels
- [ ] Evaluate cross-paper answer synthesis after localization
- [ ] Measure reference-resolution precision for uploaded PDFs

Decision: retain the local vision plus BM25 decomposition. It produces the strongest current positive result. Do not describe the broader multi-document reasoning problem as solved.

## Milestone 4: honest faithfulness measurement

**Status: Done for automated audit, partial for validation**

- [x] Generate answers across four local models
- [x] Compare ScholAR with local PDF-chat, vanilla RAG, and PaperQA2-style baselines
- [x] Replace cosine generation scoring with a local entailment judge
- [x] Build a negative control with deliberately corrupted claims
- [x] Re-score the baseline comparison and model matrix
- [x] Audit cited-page support
- [x] Record scorer false positives and contradictions
- [ ] Validate the judge against independent expert ratings
- [ ] Repeat stochastic generation runs and report variance across runs

Decision: older cosine scores remain provenance artifacts, not headline results. The current automated estimate is provisional until human validation.

## Milestone 5: claim-level evidence repair

**Status: Next**

### 5.1 Build the verifier

- [ ] Parse a generated answer into atomic claims without losing citation attachment
- [ ] Gather the exact evidence attached to each claim
- [ ] Classify each pair as Supported, Partial, Unsupported, or Contradicted
- [ ] Record judge confidence and raw rationale in a machine-readable trace
- [ ] Add unit cases for negation, number changes, multi-part claims, and citation reuse

### 5.2 Build the repair policy

- [ ] Keep supported claims unchanged
- [ ] Qualify partial claims to the supported scope
- [ ] Remove or regenerate unsupported claims
- [ ] Force abstention when no supplied evidence can support the answer
- [ ] Prevent the repair model from introducing uncited new claims
- [ ] Return a visible repair status for research logging, without cluttering the reader UI

### 5.3 Evaluate the intervention

- [ ] Freeze the current pipeline as the control
- [ ] Run paired evaluation on the same cases and model outputs
- [ ] Measure judge faithfulness, contradiction rate, citation support, answer coverage, and latency
- [ ] Check whether repair improves grounding without destroying useful answer content
- [ ] Validate a sample with humans before treating the automated delta as real

Exit criterion: a paired improvement in claim support with no material loss in coverage, confirmed by expert review.

## Milestone 6: human evaluation

**Status: Next, instrument ready**

- [x] Create 100 cases across 25 papers
- [x] Generate 350 answers across four local models
- [x] Create a blinded randomized score sheet
- [x] Write anchored evaluator guidance
- [x] Implement aggregation and statistical analysis
- [ ] Freeze answer and case versions
- [ ] Recruit at least two expert evaluators, preferably three
- [ ] Run a calibration round and clarify ambiguous rubric language
- [ ] Collect independent scores
- [ ] Compute inter-annotator agreement
- [ ] Compute citation precision, recall, and F1
- [ ] Correlate human faithfulness with the automated judge
- [ ] Audit high-disagreement and judge-error cases
- [ ] Publish de-identified aggregate results and protocol

Exit criterion: complete ratings with reported agreement, rater counts, adjudication policy, and a clear conclusion about whether the automated judge is usable.

## Milestone 7: reproducibility and code quality

**Status: Next**

- [ ] Split evidence construction, chat prompting, and citation normalization out of `backend/main.py`
- [ ] Add unit tests for safe paper IDs, URL validation, chunk page boundaries, BM25 ranking, figure pinning, and citation normalization
- [ ] Add API integration tests with Ollama mocked
- [ ] Capture model digests and prompt versions in generated result files
- [ ] Record environment, generation settings, and Git revision with every experiment
- [ ] Make evaluation output writes atomic and resumable where missing
- [ ] Add a small fixture PDF that can legally be committed for tests
- [ ] Add continuous integration for Python checks and frontend type/build checks

Exit criterion: a new contributor can reproduce deterministic results and run all non-model checks from a clean clone.

## Milestone 8: document understanding improvements

**Status: Later**

### Structured extraction

- [ ] Compare PyMuPDF with GROBID and one equation-aware parser
- [ ] Evaluate table fidelity, reading order, captions, and equations on a fixed corpus
- [ ] Preserve the existing chunk schema so retrievers remain comparable

### Mathematical evidence

- [ ] Add a math-heavy hand-labeled retrieval set
- [ ] Preserve symbols and equations during extraction
- [ ] Test math-aware tokens or structured formula fields
- [ ] Add equation-region citation highlighting

### Visual evidence

- [ ] Improve figure and table region boundaries
- [ ] Separate caption retrieval from image-answer routing
- [ ] Build a visual correctness benchmark with task-specific labels
- [ ] Test whether a visual page retriever adds value after controlling for caption overlap

Exit criterion: improvements must be demonstrated on labeled content-specific benchmarks, not screenshots or anecdotes.

## Milestone 9: product hardening

**Status: Later**

- [ ] Stream model responses
- [ ] Add explicit job progress for long paper preparation
- [ ] Improve PDF text-to-glyph highlighting
- [ ] Add paper-data deletion from the UI
- [ ] Add accessible keyboard navigation and focus management
- [ ] Add clearer offline, fallback, and network-status indicators
- [ ] Add exportable study notes with citation provenance

Production deployment remains out of scope until authentication, rate limiting, user isolation, storage quotas, and a threat model exist.

## Parked ideas

These ideas may be useful later, but they do not address the current research bottleneck:

- citation graph visualization
- recommendation systems and similar-paper discovery
- a full knowledge graph
- fine-tuning a custom model
- cloud-hosted inference modes
- multi-user collaboration
- slide generation
- venue-specific submission packaging

Do not prioritize these ahead of claim repair, human evaluation, and reproducibility.

## Manuscript and future submission

**Status: venue-neutral draft retained**

- [x] Preserve the current manuscript, bibliography, and figures under neutral filenames
- [x] Remove canceled venue deadlines, style files, and submission checklists
- [x] Rewrite the abstract and discussion around provenance versus faithfulness
- [ ] Update the manuscript after human evaluation
- [ ] Add the claim-repair experiment if successful
- [ ] Choose a venue whose scope fits the evidence
- [ ] Apply that venue's template only after selection
- [ ] Complete venue-specific ethics, reproducibility, and disclosure requirements at that time

## Suggested order of work

1. Freeze and run the human-evaluation calibration round.
2. Refactor citation normalization enough to test it independently.
3. Build the claim verifier and repair policy.
4. Run paired automated and human evaluation.
5. Improve reproducibility metadata and tests.
6. Rewrite the manuscript around the resulting evidence.
7. Select the venue and only then add submission formatting.
