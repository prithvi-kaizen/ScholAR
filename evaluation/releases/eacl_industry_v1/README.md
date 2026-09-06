# EACL Industry release v1 — blocked evidence directory

This is the designated output directory for the eventual claim-bearing EACL Industry release. It intentionally contains **no `manifest.json`, expected-key universe, raw rows, scores, aggregates, or empirical tables**. Creating plausible-looking release evidence before the studies run would be a research-integrity failure.

Generation is controlled by `evaluation/configs/eacl_industry_v1.json`, whose `study_status` remains `NOT_READY`. The runner fails closed until the held-out paper-disjoint cases are frozen and hashed, immutable model identity is recorded, human calibration is complete, and the required study gates in `gates.json` are cleared with real evidence.

When ready, run the four explicit stages from the repository root:

```bash
.venv/bin/python evaluation/run_release_suite.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/score_release.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/aggregate_release.py --config evaluation/configs/eacl_industry_v1.json
.venv/bin/python evaluation/validate_release.py --release-dir evaluation/releases/eacl_industry_v1
```

`SUCCESS`, `ABSTAINED`, and `ERROR` rows are immutable and all remain in expected-count accounting. Scoring never invokes generation, tables read only aggregate JSON, and the validator checks schemas, keys, failures, checksums, provenance, anonymity, and forbidden legacy evidence.
