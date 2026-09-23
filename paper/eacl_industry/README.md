# SCHOLAR Industry Track review manuscript

The anonymous manuscript reports audited **retrospective diagnostics** only. Its retrieval, repair, status, and latency results are supported by `evaluation/PHASE2_AGGREGATE_PUBLIC.json` and the local trace auditor. Its separate 150-case table is a post-hoc LLM-based diagnostic supported by `evaluation/EACL_DEMO_150_LUNA_PUBLIC.json` and the retained judgment rows; it is not a human, held-out, or named-model release. The visual-judgment distribution is excluded because its completed labels do not match the frozen rubric. As recorded in `evaluation/HUMAN_EVAL_50_STATUS_PUBLIC.json`, the prepared 50-question, five-evaluator study had 250 planned and zero submitted ratings at the September 21, 2026 manuscript freeze, so the separate `eacl_industry_v1` human-evidence gate remains closed.

The [official EACL 2027 Industry call](https://2027.eacl.org/calls/industry/) requires at most six review content pages, unmodified ACL style, double-blind review, a dedicated Limitations section before references, and any appendix after the bibliography. The deadline was **11 September 2026, 23:59 AoE** (= 12 September 2026, 11:59 UTC / 17:29 IST), so it had passed when the rules were rechecked on 21 September 2026. This source remains useful for a chair-approved late revision, rebuttal material, or a later venue, but it must not be uploaded as a new on-time EACL Industry submission without venue authorization. It also remains scientifically NO-GO under the project's strict held-out/human-evidence gate.

ACL policy requires disclosure when generative AI contributes paper content. Review submissions cannot contain acknowledgments under the Industry Track call, so the authors must follow the venue's submission-time disclosure mechanism and add a specific acknowledgment to the final version if the paper is accepted.

Build and check the manuscript:

```bash
make -C paper/eacl_industry compile
.venv/bin/python evaluation/validate_submission_pdf.py --paper-dir paper/eacl_industry
.venv/bin/python evaluation/validate_paper.py --paper-dir paper/eacl_industry
```

The strict `make submission-ready` gate additionally requires the planned paper-disjoint and independent human evidence, governance reviews, and valid measured release; it should remain NO-GO until those are genuinely complete. Do not turn off the gate to describe the retrospective paper as a validated deployment study.
