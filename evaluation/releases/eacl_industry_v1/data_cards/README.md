# Held-out dataset card — pending

No held-out cases have been created or implied. Before generation, this directory must contain a paper-disjoint case file, licensing/provenance notes, annotation definitions, exclusions, exact supporting spans or regions, answerability labels, and a frozen SHA-256 recorded in the release config.

The canonical case contract is `evaluation.release.schemas.CaseRecord`. Every measured
case must identify the reserved `test` split, evidence modality, answerability, gold
pages, visual regions where applicable, required key points, acceptable answers,
license/redistribution status, and a two-annotator adjudication hash.

Use:

```bash
.venv/bin/python evaluation/audit_heldout_candidate.py \
  --candidate path/to/adjudicated_cases.json \
  --require-ready \
  --freeze-to evaluation/releases/eacl_industry_v1/data_cards/heldout_cases.json
```

The command refuses to write the release file if it finds development-paper leakage,
unknown papers, duplicates, missing annotations, missing license records, insufficient
paper/case counts, modality imbalance, or absent explicit/implicit pairs.
