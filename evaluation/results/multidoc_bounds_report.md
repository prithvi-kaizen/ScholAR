# Multi-Document Locality: Oracle and Random-Floor Bounds

Computed over the 10 locality cases with a resolvable arXiv secondary paper (same subset as `locality_arxiv` in `multidoc_eval_report.md`).

| Bound | R@1 | R@5 | MRR |
|---|---:|---:|---:|
| Random floor (uniform guess, no retrieval signal) | 0.125 | 0.625 | 0.34 |
| ScholAR (flat retrieval across all loaded papers) | 0.0 | 0.5 | 0.183 |
| Oracle (retrieval restricted to only the correct paper) | 0.0 | 0.7 | 0.289 |

Average candidate secondary papers per case: 8.0.
