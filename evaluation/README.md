# ScholAR Quantitative Evaluation

This folder contains a small, real evaluation for the ScholAR retrieval layer.

The goal is not to claim conference-level results yet. The goal is to create a clear first benchmark that can be extended into the final project report.

## What is evaluated

ScholAR answers paper questions by retrieving chunks from the selected PDF, then asking the model to answer from those chunks. If retrieval is wrong, the model can cite the wrong evidence. So this evaluation measures whether retrieval finds the right evidence chunks.

## Compared systems

The script compares four retrieval settings:

1. `keyword_overlap`: simple token overlap baseline.
2. `bm25_only`: BM25-style lexical baseline.
3. `bm25_primary_no_page_hints`: current BM25-primary retrieval without page hints.
4. `bm25_primary_with_page_hints`: current BM25-primary retrieval with page hints when a query mentions pages.

This gives one comparison and one ablation:

- Comparison: BM25-primary retrieval versus keyword and BM25 baselines.
- Ablation: BM25-primary retrieval with page hints versus BM25-primary retrieval without page hints.

The project originally used a more aggressive hybrid score. The first evaluation showed BM25 was more reliable, so the app was updated to use BM25 as the main signal and use semantic, section, phrase, and page signals only as light reranking boosts.

## Metrics

- `Recall@1`: Did the top retrieved chunk include a known relevant chunk?
- `Recall@3`: Did the top 3 include a known relevant chunk?
- `Recall@5`: Did the top 5 include a known relevant chunk?
- `MRR`: Mean reciprocal rank of the first relevant chunk.

These metrics are common for retrieval evaluation and are simple enough to explain.

## Run

From the project root:

```bash
python3 evaluation/run_retrieval_eval.py
```

The script writes:

- `evaluation/results/retrieval_eval_results.json`
- `evaluation/results/retrieval_eval_report.md`

## Important limitation

The benchmark is intentionally small. It uses prepared local papers in `backend/data/papers`. This is enough for a project milestone and a real ablation, but not enough for a research claim yet.
