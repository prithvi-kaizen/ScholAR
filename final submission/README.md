# ScholAR Final Submission

This folder collects the main files for the final GenAI project submission.

## Contents

- `reports/FINAL_WRITTEN_REPORT.md`: 4 to 6 page written report covering the real problem, system design, what worked, what failed, and why.
- `evaluation/retrieval_eval_report.md`: quantitative evaluation report with comparison and ablation.
- `evaluation/retrieval_eval_results.json`: raw evaluation output.
- `evaluation/benchmark_cases.json`: manually checked retrieval benchmark cases.
- `architecture/ScholAR_architecture_flow.png`: architecture and flow diagram for the system.
- `architecture/ScholAR_architecture_flow.svg`: editable vector version of the diagram.

Project setup instructions are in the root `README.md`. The domain note is in the root `domain_note.md`.

## Current evaluation summary

The final retrieval design uses BM25 as the primary retrieval signal, with lightweight reranking for page hints, section hints, phrase hints, and semantic overlap.

| System | Recall@1 | Recall@3 | Recall@5 | MRR |
|---|---:|---:|---:|---:|
| keyword overlap | 0.571 | 0.786 | 0.929 | 0.687 |
| BM25 only | 0.714 | 0.929 | 1.000 | 0.812 |
| BM25-primary without page hints | 0.714 | 0.929 | 1.000 | 0.812 |
| BM25-primary with page hints | 0.714 | 0.929 | 1.000 | 0.812 |

The honest conclusion is that BM25 was the strongest tested grounding method, so ScholAR was updated to use BM25 as the backbone instead of forcing a more complicated hybrid-primary retriever.

## Reproduce evaluation

From the project root:

```bash
python3 evaluation/run_retrieval_eval.py
```
