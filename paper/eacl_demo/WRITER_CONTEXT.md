# ScholAR EACL Demo writer context

This is the starting document for anyone writing the EACL Demo paper without
prior knowledge of the repository. Read it before editing the manuscript.

Repository root:

`/Users/prithvirajsangramsinhpatil/Downloads/ScholAR`

All paths below are relative to that root. File paths belong in this handoff
document, the reproducibility notes, and the claim map. They should **not** be
written into the main paper as prose.

## 1. What ScholAR is

ScholAR is a local-first assistant for asking questions about scientific PDFs.
It ingests a paper, preserves page and source identity, retrieves text and
visual evidence, produces an answer with citations, checks the answer with a
deterministic verifier, and stores an audit trace. The demo value is not simply
"chat with a PDF." The useful distinction is that users can inspect the page,
text region, figure/table crop, citation, verification state, and execution
trace behind an answer while the main analysis path can stay local.

The default answer generator is local Ollama `qwen3.5:9b`. The generator does
not decide page numbers or evidence identity; those come from stored artifacts
and remain application-owned.

## 2. What the demo paper should claim

The paper should make three restrained contributions:

1. A working, local-first scientific-paper assistant with inspectable text and
   visual evidence.
2. A source-identity-preserving pipeline from PDF ingestion to answer trace,
   including explicit fallbacks and a lexical/number-consistency verifier.
3. A usable interface for moving from a question to a cited answer, inspecting
   its evidence, and exporting or auditing the reasoning trace.

Do not call the verifier semantic entailment, factual correctness, or an expert
judge. Do not describe automatic lexical scores as human evaluation. Do not
claim that local execution makes the system correct, private under every
configuration, or generally better than all competing systems.

## 3. Paper sections at a glance

The paper is planned in this order:

1. **Abstract** — the problem, what ScholAR does, the live demo, and the main
   validation evidence.
2. **Introduction** — why researchers need inspectable answers from scientific
   papers and what ScholAR contributes.
3. **Related Work** — how ScholAR relates to scientific QA, retrieval systems,
   multimodal document tools, and local research software.
4. **System** — how papers are processed, evidence is retrieved, answers are
   generated, and citations and traces are checked.
5. **Demo** — the user workflow and what evaluators will see in the interface.
6. **Evaluation** — eligible retrieval results, the human study (placeholder
   until complete), and the separate Luna evaluation of the remaining 150
   questions.
7. **Availability** — how to access or install ScholAR, its license, and links
   to the demo video and software.
8. **Conclusion** — what the system offers and the main limits of the current
   evidence.
9. **Limitations, Ethics, References, and Appendix** — constraints, data and
   evaluator protections, cited work, and supporting detail.

## 4. Canonical reading order

Read these files in order:

| Order | File | Why it matters |
|---:|---|---|
| 1 | `README.md` | Product scope, setup, and user-facing features. |
| 2 | `docs/CODEBASE.md` | Current runtime, repository map, service ownership, and invariants. |
| 3 | `docs/PIPELINE.md` | Exact ingestion, retrieval, generation, verification, and fallback behavior. |
| 4 | `docs/SETUP.md` | Reproducible local setup and model requirements. |
| 5 | `paper/eacl_industry/main.tex` | Existing paper structure and language to reuse selectively. |
| 6 | `paper/eacl_industry/sections/system.tex` | Existing system description. Verify every sentence against current code. |
| 7 | `paper/eacl_industry/sections/method.tex` | Existing retrieval, generation, and verification description. |
| 8 | `paper/eacl_industry/sections/evaluation.tex` | Existing retrospective evaluation protocol and caveats. |
| 9 | `paper/eacl_industry/sections/results.tex` | Existing eligible result tables and restrained interpretation. |
| 10 | `evaluation/EVIDENCE_LEDGER.md` | Authority on which numerical artifacts may enter a paper. |
| 11 | `evaluation/PHASE2_RESULTS.md` | Human-readable record of the eligible retrospective results. |
| 12 | `evaluation/PHASE2_AGGREGATE_PUBLIC.json` | Machine-readable source for those numbers. |
| 13 | `paper/eacl_industry/claim_map.json` | Existing map from system/empirical claims to code or result sources. |
| 14 | `paper/eacl_industry/REVIEWER_RISK_REGISTER.md` | Known reviewer objections and claim risks. |
| 15 | `evaluation/human_eval/STUDY_50_FIVE_RATERS_README.md` | Human-study protocol and current limitations; note the stale paths described below. |
| 16 | `evaluation/LUNA_JUDGE_PROTOCOL.md` | Existing independent-LLM protocol; useful as a template, not the 150-case solution. |
| 17 | `docs/DATASET_EVALUATION_PLAN.md` | Broader evaluation, calibration, agreement, and reporting rules. |

## 5. System map and claim sources

### Ingestion and local data

| Topic | Primary implementation | Verification |
|---|---|---|
| API and chat endpoints | `backend/main.py` | `tests/test_api_endpoints.py`, `tests/test_answer_pipeline.py` |
| PDF/AST ingestion | `backend/services/ingestion_service.py` | `tests/test_ingestion_routes.py`, `tests/test_ingestion_dual_engine.py` |
| Atomic paper bundle publication | `backend/services/paper_finalize_service.py` | `tests/test_paper_finalize_service.py` |
| Network boundary and strict-local mode | `backend/services/network_policy_service.py` | `tests/test_network_policy.py`, `tests/test_offline_strict.py` |
| Stored evidence types | `backend/schemas/evidence.py`, `backend/schemas/visual_document.py` | `tests/test_schemas_and_capabilities.py` |

Prepared PDFs, indexes, model caches, traces, and evaluator returns live under
ignored/private paths such as `backend/data/`, `evaluation/results/`, and
`evaluation/human_eval/private/`. They are not repository source and must not
be uploaded merely to make the paper reproducible.

### Retrieval, answering, and audit

| Topic | Primary implementation | Verification |
|---|---|---|
| BM25, dense, modality, visual, and fusion retrieval | `backend/services/retrieval_service.py` | `tests/test_retrieval_hybrid.py`, `tests/test_retrieval_identity.py` |
| Dense retrieval and cache binding | `backend/services/dense_embedding_service.py` | `tests/test_dense_embedding_cache.py` |
| Full-page/crop visual retrieval | `backend/services/document_visual_retrieval_service.py`, `backend/services/visual_page_retrieval_service.py`, `backend/services/colqwen_visual_retrieval_service.py` | `tests/test_visual_page_retrieval.py`, `tests/test_colqwen_visual_retrieval.py` |
| Question analysis and routing | `backend/services/question_analyzer.py`, `backend/services/routing_service.py` | `tests/test_question_analyzer.py`, `tests/test_routing_and_grounding.py` |
| Evidence budgeting by RAM/model capability | `backend/services/budgeting_service.py` | `tests/test_budgeting_service.py` |
| End-to-end answer path | `backend/services/answer_pipeline.py` | `tests/test_answer_pipeline.py`, `tests/test_chat_provenance.py` |
| Visual answering | `backend/services/vision_service.py` | `tests/test_vision_formatting.py`, `tests/test_visual_verification_origin.py` |
| Verification, citation remapping, and repair | `backend/services/verifier_service.py` | `tests/test_verifier_and_repair.py`, `tests/test_atomic_verifier.py` |
| Evidence graph and trace | `backend/services/evidence_graph_service.py`, `backend/schemas/answer_trace.py` | `tests/test_evidence_graph.py`, `tests/test_phase4_streaming_telemetry.py` |
| Export | `backend/services/export_service.py` | `tests/test_export_service.py` |

### Interface and demo sequence

| Demo action | Interface source |
|---|---|
| Open/upload a paper | `frontend/app/page.tsx` |
| Main paper workspace | `frontend/app/paper/[id]/page.tsx`, `frontend/components/StudyWorkspace.tsx` |
| Ask a question and inspect citations | `frontend/components/ChatBox.tsx` |
| Read the source PDF/highlighted region | `frontend/components/PdfViewer.tsx` |
| Inspect evidence graph | `frontend/components/EvidenceGraphModal.tsx` |
| Inspect references | `frontend/components/ReferencesPanel.tsx` |
| Compare papers | `frontend/app/compare/page.tsx` |
| Inspect traces/telemetry | `frontend/app/telemetry/page.tsx` |

The paper should follow one real user story: prepare a scientific PDF, ask a
question, read the answer, click a citation, inspect the source evidence, and
open or export the trace. Mention only features that work in the recorded demo.

## 6. Current diagrams and screenshots

| Asset | Status | Use |
|---|---|---|
| `paper/eacl_industry/figs/architecture.tex` | Usable source | Best starting point for the demo-paper architecture. Copy and simplify rather than editing the industry version. |
| `paper/eacl_industry/figs/architecture.pdf` | Existing vector output | Reference rendering only; regenerate from the copied demo source. |
| `paper/eacl_industry/figs/architecture_preview.png` | Existing preview | Visual inspection only. |
| `docs/diagrams/system-overview.svg` | Accurate but dense | Technical reference; do not drop it directly into the paper without simplifying. |
| `docs/diagrams/ingestion-pipeline.svg` | Accurate but dense | Reference for ingestion details. |
| `docs/diagrams/answer-pipeline.svg` | Accurate but dense | Reference for answer-stage details. |
| `paper/eacl_demo/figs/ui_demo_raw.png` | Authentic raw capture | Preserve as source material. |
| `paper/eacl_demo/figs/ui_demo.png` | Provisional crop | May be used for drafting, but replace with a clean landscape capture if the final UI can be run. |

The final paper needs at least two readable visuals: a vector system architecture
and an authentic UI/evidence-inspection screenshot. A compact demo-flow or
evidence-inspection inset is optional if space permits. Never fabricate a UI.

## 7. Current evaluation evidence

### Eligible retrospective machine evidence

The authority is `evaluation/EVIDENCE_LEDGER.md`. The current eligible sources
are `evaluation/PHASE2_RESULTS.md` and
`evaluation/PHASE2_AGGREGATE_PUBLIC.json`.

- Retrieval uses 100 questions from 25 papers. BM25+dense RRF has Hit@1 0.85
  versus BM25 0.84, but the difference is uncertain. The modality boost harms
  Hit@1 and MRR in this run. Report the negative finding.
- The 30-case repair comparison is an automatic diagnostic. It raises lexical
  overlap but lowers cited-page alignment. It is not a correctness study.
- The 20-case cross-paper result measures pipeline status only. It does not
  establish semantic refusal accuracy.
- Latency is a single-machine, uncontrolled-cache measurement with no peak-RSS
  profile. Report the hardware and caveat.

Use `paper/eacl_industry/sections/evaluation.tex` and
`paper/eacl_industry/sections/results.tex` as prose/table starting points, but
reframe them for a demo paper and keep only the most useful compact table.

### Human evaluation: in progress

The candidate five-rater package is:

`evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters.zip`

The package contains the same 50 questions for five evaluators. Sampling is ten
papers by five questions, with 10 easy, 20 medium, and 20 hard operational
labels. The category counts are 10 direct-text, 23 text-synthesis, 8 visual, 3
math/mechanism, and 6 table/quantitative. The selection source is:

`evaluation/human_eval/private/offline_50_model_trace_v7/ScholAR_Offline_50_Five_Raters/selection_manifest.json`

The build receipt and private database are:

- `evaluation/human_eval/private/offline_50_model_trace_v7/build_manifest.json`
- `evaluation/human_eval/private/offline_50_model_trace_v7/study.sqlite3`

The supplied reference-answer source is frozen at:

`evaluation/human_eval/private/offline_50_model_trace_v7/annotation_source.zip`

Do not call those references independently verified ground truth until the
raters inspect them. The package uses saved model evidence regions; it does not
reconstruct highlights by post-hoc phrase search.

There is one blocking inconsistency before real returns are imported:

- `evaluation/human_eval/offline_50_builder.py` writes `offline_50_v2`.
- `evaluation/human_eval/offline_50_import.py` currently requires
  `offline_50_v1` metadata.
- `evaluation/human_eval/STUDY_50_FIVE_RATERS_README.md` still names older v4/v1
  paths instead of the v7 candidate.

Until this is fixed and tested, the manuscript must contain only a clearly
marked human-evaluation placeholder, never invented values or an empty result
table that looks complete.

### Independent LLM evaluation: not yet implemented for 150 cases

The requested evaluator is GPT-5.6 Luna, which differs from the local
`qwen3.5:9b` answer generator. The existing files
`evaluation/run_luna_judge.py`, `evaluation/aggregate_luna_judge.py`, and
`evaluation/LUNA_JUDGE_PROTOCOL.md` cover only 20 text/math cases across an
older condition comparison. They exclude visual cases and cannot be reported
as the requested remaining-150 evaluation.

The new 150-case run must subtract the exact 50 human-study case IDs from:

- `evaluation/benchmarks/two_hundred_questions_dataset.json`
- `evaluation/results/200_questions_benchmark_results.json`

There are 163 reference records that pass the current fail-closed question
matching rule. Because the selected human 50 are among them, only 113 of the
remaining 150 initially have eligible paired references. The other 37 require
reference repair/verification before a correctness score can use all 150.
Grounding/citation judgments may still be possible if their cited evidence is
complete, but denominators must be reported by dimension.

## 8. Artifacts that must not support paper claims

Do not take claims or headline numbers from these files:

- `evaluation/reports/ScholAR_200_Question_Evaluation_Report.tex`
- `evaluation/reports/ScholAR_200_Question_Evaluation_Report.pdf`
- `evaluation/reports/ScholAR_Team_Briefing_Metrics_Ablations.tex`
- any file that calls lexical heuristics "expert review," "human correctness,"
  or "atomic correctness"
- superseded directories under `evaluation/human_eval/private/`
- the 50-question legacy reports listed as excluded in
  `evaluation/EVIDENCE_LEDGER.md`

The 200-question run may supply frozen questions and saved model outputs for a
new evaluation. Its existing automatic labels are retrospective diagnostics,
not human or LLM-judge evidence.

## 9. Public/private boundary

Keep in the public repository:

- source code, tests, configuration schemas, setup instructions, the paper
  source, figures made by the authors, protocols, empty templates, and
  privacy-safe aggregate receipts;
- `LICENSE` (MIT), `README.md`, and `README_REPRODUCE.md`;
- a small redistributable example only if its source license permits it.

Keep private/local:

- source PDFs without clear redistribution permission;
- `backend/data/`, model caches, embeddings, raw traces, prompts/completions,
  raw evaluation rows, evaluator exports, contact details, consent records,
  and `evaluation/human_eval/private/`;
- API keys and `.env` files.

The external LLM judge is an additional data boundary. `store=false` is not a
zero-retention guarantee. Do not transmit paper text, crops, or answers until
the project owner confirms that the relevant papers and study data may be sent
to the chosen API.

## 10. Venue requirements to enforce

The authoritative call is:

`https://2027.eacl.org/calls/demos/`

As checked on 18 September 2026, the EACL 2027 Demo submission deadline is
22 September 2026 at 23:59 AoE. A submission requires all three components:

1. a paper of at most six content pages in the official ACL style;
2. a demonstration video no longer than 2.5 minutes;
3. a live demo website or an installable software-package link.

The paper is single-blind, so names and affiliations must appear. It should
cover motivation/novelty, related work, architecture and functionality,
intended users and interface, validation, access/license, and presentation
details. Zero validation evidence may be desk-rejected even though a
comprehensive benchmark is not required for a demo paper. Put both the video
and demo/package links in the PDF and OpenReview form.

Recheck the official call immediately before submission. Do not reuse
`paper/eacl_industry/venue_requirements.json`; it describes a different track
and deadline.

## 11. Decisions the writer cannot invent

The project owner must supply or confirm:

- final paper title, author order, affiliations, and contact email;
- the final public repository/package URL (the local Git remote and the
  previously supplied `SCHOLAR-EACL` URL do not currently match);
- final video URL;
- whether a hosted demo is offered or only an installable local package;
- permission for external Luna judging and private PDF sharing;
- ethics/consent/compensation wording and any institutional approval status;
- final human-evaluation results after all five valid JSON returns arrive.

Everything else in the implementation plan can be completed from the
repository and validated sources.
