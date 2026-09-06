# Release-v1 minimal fixture

**NON-RELEASE TOY ARTIFACT.** `evidence_class="toy"`; `claim_status="non_release"`. No value in this directory is empirical evidence about ScholAR.

This committed fixture exercises the artifact-only path without a backend server, Ollama, model weights, paper cache, or network access. Its four expected keys span two seeds and two cases, and its raw JSONL deliberately stores rows out of key order. It includes two successes, one explicit abstention, and one immutable error. The scorer, case-balanced aggregator, table renderer, checksum validator, and manifest validator must reproduce deterministic golden outputs from these rows.

Rebuild only the fixture:

```bash
SCHOLAR_NETWORK_MODE=strict-local .venv/bin/python evaluation/fixtures/releases/release_v1_minimal/build_fixture.py
```
