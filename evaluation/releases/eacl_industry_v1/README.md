# EACL Industry release v1 — blocked evidence directory

This is the designated output directory for the eventual claim-bearing EACL Industry release. It intentionally contains **no `manifest.json`, expected-key universe, raw rows, scores, aggregates, or empirical tables**. Creating plausible-looking release evidence before the studies run would be a research-integrity failure.

Generation is controlled by `evaluation/configs/eacl_industry_v1.json`, now using **release schema v2** while the separate `release_v1_minimal` toy fixture remains unchanged. Its `study_status` remains `NOT_READY`. The runner fails closed until the held-out paper-disjoint cases, corpus manifest, component identities, calibration artifacts, prompts, model, and exact hardware are frozen and the required gates are cleared with real evidence.

When ready, run the four explicit stages from the repository root:

```bash
.venv/bin/python evaluation/run_release_suite.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/score_release.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/score_human_gate.py --release-dir evaluation/releases/eacl_industry_v1 --annotations <adjudicated-human-bundle.json> --spec <paired-gate-spec.json>
.venv/bin/python evaluation/aggregate_release.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/validate_release.py --release-dir evaluation/releases/eacl_industry_v1
```

`SUCCESS`, `ABSTAINED`, and `ERROR` rows are immutable and all remain in expected-count accounting. A missing model or returned condition mismatch becomes an immutable `ERROR` row. Schema-v2 tables cannot be rendered before the independent-human primary gate passes. Their provenance binds the aggregate, stable release-manifest identity, and human-gate hash.
