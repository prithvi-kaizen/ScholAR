# EACL 2027 Demo paper implementation plan

This plan is written for an implementer and writer who has no prior context.
Start with `paper/eacl_demo/WRITER_CONTEXT.md`. The goal is a truthful,
six-page EACL Demo paper, a short video, and a working package link. This plan
does not promise acceptance; it is designed to remove avoidable desk-rejection
and reviewer risks.

Repository root:

`/Users/prithvirajsangramsinhpatil/Downloads/ScholAR`

## Definition of done

The submission is ready only when all of these are true:

- the paper compiles in the current official ACL style and has no more than six
  content pages;
- author names and affiliations are present because the demo track is
  single-blind;
- every system claim resolves to current code/tests and every number resolves
  to an eligible aggregate or a newly frozen evaluation;
- the architecture and UI figures are readable at final column/page size;
- human results are either inserted from five complete, validated evaluator
  files or are described only as an in-progress protocol with no invented
  values;
- the new Luna evaluation covers the exact 150 cases outside the human sample,
  with honest per-dimension denominators;
- the PDF contains a working video link and live-demo or installable-package
  link;
- the video is at most 2.5 minutes and shows the same system described in the
  paper;
- the final public repository contains code, documentation, protocols, tests,
  and safe aggregates, but no private PDFs, raw answers, raw judge traffic, or
  evaluator exports.

## Target manuscript layout and page budget

Create a new demo manuscript. Do not overwrite the industry manuscript.

| Target file | Section | Approximate content budget | Source material |
|---|---|---:|---|
| `paper/eacl_demo/main.tex` | Title, author block, inputs, link box | 0.25 page | `paper/eacl_industry/main.tex`, official call |
| `paper/eacl_demo/sections/abstract.tex` | Problem, working system, key distinction, validation, availability | 150–180 words | `README.md`, `docs/CODEBASE.md`, eligible evaluation sources |
| `paper/eacl_demo/sections/introduction.tex` | User problem, gap, demo contribution, intended users | 0.75 page | `paper/eacl_industry/sections/introduction.tex`, `docs/CODEBASE.md` |
| `paper/eacl_demo/sections/related_work.tex` | Scientific-document QA, RAG, multimodal document systems, auditable/local tools | 0.45 page | `paper/eacl_industry/sections/related_work.tex`, `paper/eacl_industry/references.bib` |
| `paper/eacl_demo/sections/system.tex` | Architecture, data boundary, ingestion, retrieval, answering, verification | 1.45 pages | `docs/PIPELINE.md`, backend sources listed in `WRITER_CONTEXT.md` |
| `paper/eacl_demo/sections/demo.tex` | User workflow, UI, evidence inspection, supported use cases | 1.05 pages | frontend files listed in `WRITER_CONTEXT.md`, final screen capture |
| `paper/eacl_demo/sections/evaluation.tex` | Compact retrospective evidence, human placeholder/results, Luna evaluation | 1.25 pages | `evaluation/EVIDENCE_LEDGER.md`, new aggregates |
| `paper/eacl_demo/sections/availability.tex` | Installation, data boundary, license, video/package links | 0.35 page | `README.md`, `docs/SETUP.md`, `README_REPRODUCE.md`, `LICENSE` |
| `paper/eacl_demo/sections/conclusion.tex` | What is demonstrated and what remains limited | 0.25 page | final manuscript |
| `paper/eacl_demo/limitations.tex` | Limits outside content pages where permitted | not counted by current call | existing limitations plus new evaluation caveats |
| `paper/eacl_demo/ethics.tex` | Privacy, external judging, evaluator data, source licenses | not counted by current call | human agreement/protocol and Luna boundary |
| `paper/eacl_demo/appendix.tex` | Protocol detail, complete tables, setup detail | after references | validated artifacts only |

If the official style or page-count rule changes, follow the call rather than
this budget. The main text should describe concepts, not internal filenames.

## Phase 0 — Freeze authority and scaffold the demo paper

### 0.1 Record the venue requirements

Create `paper/eacl_demo/venue_requirements.json` with the current values from
`https://2027.eacl.org/calls/demos/`: deadline, six-page limit, single-blind
status, 2.5-minute video cap, mandatory software/live-demo link, and link
requirements. Record the check date and official URL.

Do not copy `paper/eacl_industry/venue_requirements.json`; it is for the
industry track and contains a different deadline and anonymity policy.

**Gate:** a second person opens the official page and confirms each value.

### 0.2 Create the manuscript scaffold

Create:

- `paper/eacl_demo/main.tex`
- `paper/eacl_demo/sections/abstract.tex`
- `paper/eacl_demo/sections/introduction.tex`
- `paper/eacl_demo/sections/related_work.tex`
- `paper/eacl_demo/sections/system.tex`
- `paper/eacl_demo/sections/demo.tex`
- `paper/eacl_demo/sections/evaluation.tex`
- `paper/eacl_demo/sections/availability.tex`
- `paper/eacl_demo/sections/conclusion.tex`
- `paper/eacl_demo/limitations.tex`
- `paper/eacl_demo/ethics.tex`
- `paper/eacl_demo/appendix.tex`
- `paper/eacl_demo/references.bib`
- `paper/eacl_demo/Makefile`
- `paper/eacl_demo/README.md`

Copy the unmodified official style files from
`paper/eacl_industry/style/acl.sty` and
`paper/eacl_industry/style/acl_natbib.bst` into
`paper/eacl_demo/style/`. Confirm against the current ACL style repository
before submission.

Copy bibliography entries only as needed from
`paper/eacl_industry/references.bib`; do not copy unused entries.

**Gate:** a placeholder manuscript compiles, uses A4, contains authors, and
produces a clean PDF with no style warnings caused by local modifications.

### 0.3 Create the demo claim map

Create `paper/eacl_demo/claim_map.json`, modeled on
`paper/eacl_industry/claim_map.json`. Give each claim an ID, exact draft text,
type, status, and source files. Initially allow only:

- software claims supported by current implementation and tests;
- retrospective empirical claims marked eligible in
  `evaluation/EVIDENCE_LEDGER.md`;
- human and Luna claims with status `PENDING`, excluded from compiled prose by
  default.

Create `paper/eacl_demo/build/evaluation_gates.tex` with false-by-default flags
such as `\humanresultsfalse` and `\llmresultsfalse`. Never compile fake rows or
zero-filled placeholders. The draft may show an editor note in a clearly
marked internal build, but the submission build must contain finished prose or
omit the unfinished result.

**Gate:** every numerical statement in the compiled PDF has a `SUPPORTED`
claim-map entry.

## Phase 1 — Write the demo-centered narrative

### 1.1 Abstract

Write `paper/eacl_demo/sections/abstract.tex` from:

- product scope: `README.md`;
- runtime boundaries: `docs/CODEBASE.md`;
- exact pipeline: `docs/PIPELINE.md`;
- eligible validation only: `evaluation/EVIDENCE_LEDGER.md` and
  `evaluation/PHASE2_AGGREGATE_PUBLIC.json`.

The abstract should answer: who uses the tool, what they can do in the live
demo, what is technically distinctive, what validation exists, and where the
tool is available. Do not lead with a list of model components.

### 1.2 Introduction and contributions

Write `paper/eacl_demo/sections/introduction.tex`. Reuse only verified concepts
from `paper/eacl_industry/sections/introduction.tex`. End with three compact
contributions matching the three claims in `WRITER_CONTEXT.md`.

Present ScholAR as an inspectable scientific-paper assistant, not as a new
foundation model or a state-of-the-art benchmark system. State intended users:
researchers, students, and technical readers inspecting papers locally.

### 1.3 Related work

Write `paper/eacl_demo/sections/related_work.tex` using
`paper/eacl_industry/sections/related_work.tex` and verified entries from
`paper/eacl_industry/references.bib`.

Organize by comparison dimension rather than a paper-by-paper list:

1. scientific-document question answering;
2. retrieval-augmented and citation-grounded assistants;
3. multimodal document understanding;
4. local/auditable research tools.

For each group, explain the demo gap ScholAR addresses. Verify every citation
against the cited paper before submission.

### 1.4 System description

Write `paper/eacl_demo/sections/system.tex` from `docs/PIPELINE.md` and the
implementation files below:

- ingestion: `backend/services/ingestion_service.py`,
  `backend/services/paper_finalize_service.py`;
- retrieval: `backend/services/retrieval_service.py`,
  `backend/services/dense_embedding_service.py`,
  `backend/services/document_visual_retrieval_service.py`;
- generation: `backend/services/answer_pipeline.py`,
  `backend/services/ollama_service.py`, `backend/services/vision_service.py`;
- verification: `backend/services/verifier_service.py`;
- trace: `backend/schemas/answer_trace.py`,
  `backend/services/evidence_graph_service.py`;
- network boundary: `backend/services/network_policy_service.py`.

Explain the path in user order: prepare PDF, retrieve evidence, inspect pixels
when needed, generate cited answer, verify/remap/repair, save trace. State all
fallback behavior briefly. Say explicitly that the verifier is lexical and
number-consistency based, not an entailment model.

### 1.5 Demo experience

Write `paper/eacl_demo/sections/demo.tex` from:

- `frontend/app/page.tsx`;
- `frontend/app/paper/[id]/page.tsx`;
- `frontend/components/StudyWorkspace.tsx`;
- `frontend/components/ChatBox.tsx`;
- `frontend/components/PdfViewer.tsx`;
- `frontend/components/EvidenceGraphModal.tsx`;
- `frontend/components/ReferencesPanel.tsx`;
- `frontend/app/telemetry/page.tsx`.

Describe one continuous scenario:

1. load or prepare a paper;
2. ask a question that needs both prose and a figure/table;
3. inspect the cited page and saved evidence region;
4. inspect verification/trace information;
5. export or continue the study workflow.

List limitations in the flow: model/capability fallbacks, visual crop coverage,
latency on local hardware, and the difference between evidence provenance and
evidence support.

### 1.6 Availability, conclusion, limitations, and ethics

Write `paper/eacl_demo/sections/availability.tex` from `README.md`,
`docs/SETUP.md`, `README_REPRODUCE.md`, and `LICENSE`. Include visible
placeholders for the final video URL and package/live-demo URL until the owner
confirms them. The final PDF cannot retain broken placeholder links.

Write `paper/eacl_demo/limitations.tex` using
`paper/eacl_industry/limitations.tex`, `evaluation/EVIDENCE_LEDGER.md`, and the
new evaluation protocols. Include development overlap, limited reference
verification, local-hardware timing, automatic verifier limits, and external
judge limits.

Write `paper/eacl_demo/ethics.tex` using
`evaluation/human_eval/EVALUATOR_AGREEMENT.md`,
`evaluation/human_eval/EVALUATOR_README.md`,
`evaluation/LUNA_JUDGE_PROTOCOL.md`, and `README_REPRODUCE.md`. State what
stays local, what is transmitted to an external judge, and how evaluator data
is protected.

**Phase 1 gate:** the paper reads as a demo paper: at least as much space is
given to the working interaction, architecture, access, and intended users as
to retrospective metrics.

## Phase 2 — Produce and verify all paper visuals

Use vector output for diagrams and authentic screenshots for the UI. Inspect
every figure at its final printed size.

### 2.1 Figure 1: system architecture

Create:

- `paper/eacl_demo/figs/architecture.tex`
- `paper/eacl_demo/figs/architecture.pdf`
- `paper/eacl_demo/figs/architecture_preview.png`

Start from `paper/eacl_industry/figs/architecture.tex`. Cross-check every box
against `docs/diagrams/system-overview.svg`,
`docs/diagrams/ingestion-pipeline.svg`, `docs/diagrams/answer-pipeline.svg`, and
`docs/PIPELINE.md`.

The final diagram should show:

`PDF -> source-scoped text/visual artifacts -> hybrid retrieval -> local answer
generation -> citation mapping and lexical verification/repair -> answer,
evidence view, and audit trace`.

Also show the local data boundary. Do not use a shield/checkmark graphic that
implies correctness. Use a one-sentence caption that explains the main claim,
not merely the box labels.

### 2.2 Figure 2: authentic UI and evidence inspection

Preserve `paper/eacl_demo/figs/ui_demo_raw.png`. Treat
`paper/eacl_demo/figs/ui_demo.png` as provisional.

Run the real application using `docs/SETUP.md`, load a paper that may legally
appear in the submission, and create a clean landscape capture. The image
should simultaneously show:

- the user question;
- the generated answer with citation markers;
- the cited page or exact saved evidence crop/highlight;
- enough interface context to understand how the user moves between them.

Save the final crop as `paper/eacl_demo/figs/ui_demo.png` and its uncropped
source as `paper/eacl_demo/figs/ui_demo_raw.png`. Remove personal paths, API
keys, unrelated browser chrome, and private paper content. Do not reconstruct
evidence regions by searching the answer text; use coordinates from the saved
model trace, consistent with `evaluation/human_eval/offline_50_builder.py`.

### 2.3 Optional Figure 3: compact demo flow

Only if the UI screenshot is too dense, create
`paper/eacl_demo/figs/demo_flow.tex` and `.pdf`: four numbered panels for
question, cited answer, evidence inspection, and trace/export. Derive the
panels from the real workflow in `frontend/components/StudyWorkspace.tsx` and
`frontend/components/ChatBox.tsx`. Do not duplicate Figure 1.

### 2.4 Visual quality gate

For each figure:

- render the complete manuscript PDF;
- inspect at 100% and as a printed A4 page;
- ensure labels are readable, colors work in grayscale, and the caption states
  what the reader should learn;
- keep diagrams as vector PDF and screenshots at sufficient resolution;
- add accessible text descriptions to the paper source comments or artifact
  documentation.

**Phase 2 gate:** Figure 1 explains the pipeline without the surrounding text,
and Figure 2 proves that the inspectable interaction is real.

## Phase 3 — Stabilize the 50-question human evaluation

The human result remains a placeholder until five real, independently returned
JSON files pass validation.

### 3.1 Fix the package/import schema mismatch

Reconcile:

- `evaluation/human_eval/offline_50_builder.py` (`offline_50_v2`);
- `evaluation/human_eval/offline_50_import.py` (currently expects v1 metadata);
- `tests/test_offline_50_human_study.py`;
- `tests/test_notion_ground_truth.py`;
- `tests/test_human_study.py`.

Choose one versioned schema, update validation and tests, and refuse mixed
packages. Update stale v4/v1 paths and examples in
`evaluation/human_eval/STUDY_50_FIVE_RATERS_README.md` to the final candidate
or rebuild into a newly named immutable directory. Never silently mutate a
package after evaluators start.

### 3.2 Run a full dry-return test

Use the package:

`evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters.zip`

Test in a clean browser profile: progress save, restore, final export, schema
validation, import, duplicate rejection, and analysis. The exact source files
are:

- UI: `evaluation/human_eval/offline_50_template.html`;
- builder: `evaluation/human_eval/offline_50_builder.py`;
- importer/analyzer: `evaluation/human_eval/offline_50_import.py`;
- instructions: `evaluation/human_eval/EVALUATOR_README.md`;
- agreement: `evaluation/human_eval/EVALUATOR_AGREEMENT.md`;
- PDF builder: `evaluation/human_eval/build_evaluator_pdfs.py`.

Confirm the selection against:

`evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters/selection_manifest.json`.

### 3.3 Collect and analyze

Assign evaluator codes R01–R05. All five evaluate the same 50 questions. Keep
the five final JSON files private. Validate each file before import. Do not
share one evaluator's labels with another before completion.

The final analysis must report:

- 250 completed question ratings;
- correctness/completeness/usefulness distributions with question-level
  confidence intervals;
- citation/evidence support distributions with their own denominators;
- reference-answer verification outcomes;
- pairwise agreement and an appropriate multi-rater agreement statistic;
- category and operational-difficulty breakdowns;
- missing/unclear counts and adjudication rules.

Write the privacy-safe aggregate to a new file such as
`evaluation/HUMAN_50_PUBLIC_AGGREGATE.json`. Keep returned JSON files, comments,
contacts, and row-level data under `evaluation/human_eval/private/`.

### 3.4 Paper integration

Until completion, `paper/eacl_demo/sections/evaluation.tex` should contain an
editor-only placeholder stating: five evaluators independently rate the same
50 stratified questions; analysis is pending. The submission build must not
show a fake result row.

After completion, add one compact table sourced only from
`evaluation/HUMAN_50_PUBLIC_AGGREGATE.json`, update
`paper/eacl_demo/claim_map.json`, and switch the human-results build gate only
after validation.

**Phase 3 gate:** five valid final exports, exactly 250 ratings, agreement and
uncertainty reported, and no private data in the public aggregate.

## Phase 4 — Implement GPT-5.6 Luna evaluation for the remaining 150

This is a new evaluation. Do not stretch the current 20-case harness to make a
150-case claim.

### 4.1 Freeze the exact complement of the human sample

Create `evaluation/build_eacl_demo_150_judge_manifest.py`. It must read:

- `evaluation/benchmarks/two_hundred_questions_dataset.json`;
- `evaluation/results/200_questions_benchmark_results.json`;
- `evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters/selection_manifest.json`;
- `evaluation/human_eval/notion_ground_truth.py`;
- `evaluation/human_eval/private/offline_50_model_trace_v7/annotation_source.zip`.

Write the frozen private manifest to
`evaluation/results/eacl_demo_150_luna/manifest.json`. Assert:

- 200 unique original case IDs;
- 50 unique human-study IDs;
- no overlap between the 50 and complement;
- exactly 150 complement IDs;
- one answer record per complement ID;
- source-paper, category, page/citation, and input hashes are retained;
- reference eligibility is explicit rather than inferred from a blank value.

Create tests in `tests/test_eacl_demo_150_luna_judge.py` for all assertions.

### 4.2 Repair or separate the 37 reference-ineligible cases

The current Notion pairing yields 163 eligible references; the human 50 use
50 of them. Therefore 113 of the remaining 150 initially have eligible paired
references and 37 do not.

For each of the 37, either:

1. independently pair and verify a corrected reference against the source
   paper, recording reviewer identity and provenance; or
2. mark reference-based correctness `not_judgeable` and evaluate only dimensions
   supported by direct source evidence.

Store the private repair/adjudication record under
`evaluation/results/eacl_demo_150_luna/reference_adjudication.jsonl`. Add only
counts and hashes to the public aggregate. Never score an answer against a
mismatched or blank reference.

### 4.3 Write the protocol before running the judge

Create `evaluation/EACL_DEMO_150_LLM_JUDGE_PROTOCOL.md` and a frozen
machine-readable rubric at
`evaluation/protocols/eacl_demo_150_llm_judge.json`.

Build from `evaluation/LUNA_JUDGE_PROTOCOL.md` and the calibration guidance in
`docs/DATASET_EVALUATION_PLAN.md`. Define:

- primary judge: GPT-5.6 Luna, separate from the `qwen3.5:9b` generator;
- exact model identifier and returned model snapshot/digest;
- blinded input fields and stable case hashes;
- a strict JSON schema;
- dimensions: answer correctness, completeness, source grounding, citation
  relevance, and visual grounding where pixels are supplied;
- labels and numeric mappings;
- `not_judgeable` rules and per-dimension denominators;
- no chain-of-thought collection or publication;
- retry, timeout, failure, resume, and duplicate rules;
- fixed decoding settings;
- privacy/licensing authorization and `store=false` caveat;
- raw/private and aggregate/public output boundaries.

The prompt must not reveal local automatic scores, human labels, difficulty
labels, or desired outcomes.

### 4.4 Implement the resumable runner

Create `evaluation/run_eacl_demo_150_luna_judge.py`. Use
`evaluation/run_luna_judge.py` only as a transport/error-handling reference.
Do not inherit its 20-case condition logic or visual exclusion.

For each case, provide only the material needed to judge it:

- question;
- frozen ScholAR answer;
- eligible reference answer, when available;
- model-cited evidence text;
- cited page/region image for visual/table cases when authorized;
- neutral instructions and response schema.

The runner must support dry-run, `--max-new`, resume, deterministic case order,
request/input hashes, explicit `--allow-external`, and fail-closed output
validation. Write one append-only private JSONL under
`evaluation/results/eacl_demo_150_luna/`. Never put API keys or full requests
in Git.

Before the full run, execute a two-case text/visual capability check. Confirm
that the API model identifier is available and that image input works. If Luna
cannot accept images through the available API, mark visual grounding
`not_judgeable`; do not replace pixels with an OCR summary and call it visual
evaluation.

### 4.5 Aggregate and audit

Create `evaluation/aggregate_eacl_demo_150_luna_judge.py`. It must reject:

- missing or duplicate IDs;
- changed manifest, rubric, source, answer, or model hashes;
- malformed judge output;
- incomplete primary runs;
- correctness scoring for reference-ineligible cases;
- visual-grounding scores without an image input receipt.

Report:

- all 150 attempted cases and failure count;
- label distributions and means by dimension;
- explicit denominators for reference correctness and visual grounding;
- question-level bootstrap confidence intervals;
- breakdown by paper, category, and operational difficulty;
- `not_judgeable` and API-failure counts;
- common failure categories using fixed labels;
- no unsupported significance claims.

Write a privacy-safe, hash-bound aggregate to
`evaluation/EACL_DEMO_150_LUNA_PUBLIC.json`. Keep raw judgments in the ignored
private result directory.

### 4.6 Calibrate Luna against the human 50

After the five human returns are frozen, run the same Luna rubric on the same
50 cases as a separate calibration set. Do not tune the rubric after looking at
those outcomes. Compare Luna and human consensus using exact agreement,
confusion matrices, weighted kappa or another predeclared ordinal statistic,
and dimension-specific coverage.

Store the public calibration receipt in
`evaluation/EACL_DEMO_LUNA_HUMAN_CALIBRATION_PUBLIC.json`. This does not turn
Luna judgments into human evaluation. In the paper, name it an
independent-model proxy and report its agreement with the human sample.

**Phase 4 gate:** exactly 150 complement cases audited; model and input hashes
frozen; all denominators explicit; visual cases receive real pixels or are
marked unjudgeable; raw data stays private.

## Phase 5 — Build the evaluation section and tables

Write `paper/eacl_demo/sections/evaluation.tex` in four short parts.

### 5.1 Retrospective retrieval baseline

Use only `evaluation/PHASE2_AGGREGATE_PUBLIC.json`, cross-checked against
`evaluation/PHASE2_RESULTS.md`. Include a compact table comparing:

- keyword;
- BM25;
- dense;
- BM25+dense RRF;
- RRF+modality.

Report Hit@1, Hit@5, MRR, and optionally nDCG if space permits. State that this
is a retrospective, development-overlapping page-retrieval study. Highlight
the negative modality result and avoid claiming a meaningful RRF improvement
over BM25 when the interval includes zero.

Target source:

`paper/eacl_demo/tables/retrieval_baselines.tex`

Generate it from a script or a checked JSON source; do not hand-copy values.

### 5.2 Compact repair/operation evidence

If space permits, include either a small repair table or a one-sentence result
from `evaluation/PHASE2_AGGREGATE_PUBLIC.json`. Name token/reference-unit overlap
as automatic overlap, not correctness. Report the cited-page-alignment decline.

Latency may be a sentence, not a table. Use the exact hardware and uncontrolled
cache caveat from `evaluation/PHASE2_RESULTS.md`.

### 5.3 Human evaluation

Before completion, keep an internal placeholder only. After completion,
generate `paper/eacl_demo/tables/human_50.tex` from
`evaluation/HUMAN_50_PUBLIC_AGGREGATE.json`. Include rating distributions,
confidence intervals, and agreement; do not squeeze all category breakdowns
into the main paper. Put detailed breakdowns in `paper/eacl_demo/appendix.tex`.

### 5.4 Luna evaluation

Generate `paper/eacl_demo/tables/luna_150.tex` from
`evaluation/EACL_DEMO_150_LUNA_PUBLIC.json`. The caption must say
"independent-LLM proxy evaluation," name the judge model, and state the
reference-correctness and visual-grounding denominators. If calibration is
available, report the human/Luna agreement next to the proxy results or in one
sentence.

### 5.5 Interpretation

The interpretation should distinguish:

- retrieval effectiveness;
- automatic answer diagnostics;
- human judgment;
- independent-LLM proxy judgment;
- operational usability/latency.

Do not merge these into a single score. Discuss adverse findings as design
lessons, not as missing successes.

**Phase 5 gate:** every table is generated from a frozen JSON source, captions
state scope and denominator, and the numbers match the claim map.

## Phase 6 — Package, video, and reproducibility

### 6.1 Public package decision

Confirm the final repository URL. The local remote and the previously provided
`https://github.com/prithvi-kaizen/SCHOLAR-EACL.git` must be reconciled by the
owner. Do not put both in the paper.

Verify the package using:

- `README.md`;
- `docs/SETUP.md`;
- `README_REPRODUCE.md`;
- `LICENSE`;
- dependency locks under `requirements/locks/`;
- `frontend/package-lock.json`.

The public repository should contain no material listed as private in
`paper/eacl_demo/WRITER_CONTEXT.md`.

### 6.2 Reproducibility smoke test

On a clean environment, follow `docs/SETUP.md` exactly. Run the relevant test
set, then at minimum:

- ingest one permitted PDF;
- ask one text question and one visual/table question;
- click citations and evidence regions;
- inspect the trace;
- restart and confirm persisted paper data remains usable.

Record the tested commit, OS, RAM, model tag/digest, and commands in
`paper/eacl_demo/ARTIFACT_SMOKE_TEST.md`.

### 6.3 Record the 2.5-minute video

Create `paper/eacl_demo/video_script.md` with this approximate timing:

- 0:00–0:20: the problem and target user;
- 0:20–0:40: local paper preparation and data boundary;
- 0:40–1:30: ask a question and inspect the cited answer/evidence;
- 1:30–2:00: visual/table evidence and verification trace;
- 2:00–2:25: export, setup, and availability;
- 2:25–2:30: concise close.

Use the same interaction described in `paper/eacl_demo/sections/demo.tex` and
shown in `paper/eacl_demo/figs/ui_demo.png`. Add readable captions. Verify the
duration and open the final URL in a private/incognito browser.

### 6.4 OpenReview metadata

Prepare `paper/eacl_demo/SUBMISSION_METADATA.md` with the final title,
abstract, authors, affiliations, keywords, conflicts, reciprocal reviewer,
video URL, software/live-demo URL, and checklist confirmations. Copy values
from the final paper rather than maintaining a second divergent description.

**Phase 6 gate:** the PDF, video, and software link all open from a clean
machine without author-only credentials.

## Phase 7 — Final paper audit

### 7.1 Mechanical validation

Create or adapt validation for the demo paper rather than silently reusing
industry assumptions. Relevant starting files are:

- `evaluation/validate_paper.py`;
- `evaluation/validate_submission_pdf.py`;
- `tests/test_submission_integrity.py`;
- `tests/test_final_submission_audit.py`;
- `tests/test_anonymous_artifact.py` (adapt because this track is single-blind).

Check page count, embedded fonts, A4 size, broken links, missing citations,
unresolved references, placeholder tokens, overfull boxes, figure resolution,
and author presence.

### 7.2 Claim audit

Compare the PDF sentence by sentence with:

- `paper/eacl_demo/claim_map.json`;
- `evaluation/EVIDENCE_LEDGER.md`;
- `evaluation/PHASE2_AGGREGATE_PUBLIC.json`;
- `evaluation/HUMAN_50_PUBLIC_AGGREGATE.json` if complete;
- `evaluation/EACL_DEMO_150_LUNA_PUBLIC.json` if complete;
- current code/tests for system claims.

Search the manuscript for risky words such as `expert`, `human`, `correct`,
`ground truth`, `entailment`, `private`, `secure`, `state of the art`, and
`production`. Keep them only when the exact scope is supported.

### 7.3 Visual and prose review

Render the final PDF pages to images and inspect each page. Confirm:

- no tiny figure labels or table text;
- no clipped content or empty regions;
- each section transitions naturally;
- the first page makes the live demo and links obvious;
- filenames and internal directory paths do not appear in main-paper prose;
- the conclusion does not introduce a new claim;
- limitations match the evaluation denominators.

### 7.4 Cold-reader review

Give the PDF, video, and package link to someone unfamiliar with ScholAR. Ask
them to answer four questions:

1. What problem does the system solve?
2. What will be shown in the demo?
3. Why is it different from ordinary PDF chat?
4. Can they install or access it from the provided link?

Revise any point they cannot answer from the submission itself.

**Final gate:** no `TODO`, `TBD`, placeholder URL, pending result presented as
complete, unsupported claim, or private artifact remains in the submission.

## Final deliverable inventory

The handoff is complete when these artifacts exist:

- `paper/eacl_demo/main.pdf`;
- all manuscript sources under `paper/eacl_demo/`;
- `paper/eacl_demo/claim_map.json`;
- `paper/eacl_demo/venue_requirements.json`;
- `paper/eacl_demo/figs/architecture.pdf`;
- `paper/eacl_demo/figs/ui_demo.png`;
- `paper/eacl_demo/ARTIFACT_SMOKE_TEST.md`;
- `paper/eacl_demo/video_script.md`;
- `paper/eacl_demo/SUBMISSION_METADATA.md`;
- working public package/live-demo URL;
- working video URL of at most 2.5 minutes;
- fixed and tested 50-question human-study import path;
- five private evaluator returns and
  `evaluation/HUMAN_50_PUBLIC_AGGREGATE.json`, if completed before freeze;
- frozen 150-case Luna manifest, private raw judgments, and
  `evaluation/EACL_DEMO_150_LUNA_PUBLIC.json`;
- optional `evaluation/EACL_DEMO_LUNA_HUMAN_CALIBRATION_PUBLIC.json` after human
  completion;
- a final audit report with page, link, claim, and private-data checks.

## Recommended execution order under deadline pressure

1. Freeze venue rules and scaffold the paper.
2. Write the demo narrative and finish the two mandatory visuals.
3. Fix and dry-test the human-study importer before collecting returns.
4. Build and dry-run the 150-case Luna manifest/runner; obtain permission before
   any external transmission.
5. Run Luna in resumable batches while writing and compiling the paper.
6. Generate tables from audited aggregates.
7. Record the video only after the UI and paper scenario are stable.
8. Run the final claim, PDF, link, and privacy audits.

Do not delay a complete, truthful demo submission to chase a larger metric
table. The architecture, working interaction, evidence inspection, access
path, and honest validation are the core of this track.
