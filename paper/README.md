# ScholAR research manuscript

This directory preserves the current research write-up in a venue-neutral format. The canceled submission template, deadlines, and compliance checklist have been removed. Do not add another conference style until the target venue is selected.

## Files

| File | Purpose |
|---|---|
| `manuscript.tex` | Main venue-neutral LaTeX source |
| `manuscript.pdf` | Last compiled research draft |
| `scholar_references.bib` | Shared research bibliography |
| `figures/results.pdf` | Results overview used by the manuscript |
| `Makefile` | Build, clean, and open commands |

The existing PDF is a snapshot from the previous formatted draft. Rebuilding `manuscript.tex` creates a venue-neutral PDF and may change pagination and line breaks.

## Build

From this directory:

```bash
make
```

Equivalent manual commands:

```bash
pdflatex manuscript.tex
bibtex manuscript
pdflatex manuscript.tex
pdflatex manuscript.tex
```

Remove generated intermediates with `make clean`.

## Current research story

The manuscript no longer argues that evidence-ID citation grounding improves faithfulness. The current evidence supports a narrower conclusion:

- evidence IDs prevent the model from inventing a page mapping;
- a real retrieved page does not necessarily support the claim attached to it;
- cosine similarity substantially overstated generated-answer faithfulness;
- ScholAR is less faithful than local vanilla RAG and PaperQA2-style baselines under the stronger judge;
- local vision-assisted M3SciQA localization remains a strong positive systems result.

The abstract, discussion, and conclusion already reflect this change. Human validation of the judge is still missing.

## Before adapting to another venue

1. Complete expert human evaluation.
2. Finish and evaluate a claim-support verification or repair intervention.
3. Freeze all headline tables against traceable result files.
4. Choose a venue whose scope matches the resulting contribution.
5. Create a venue-specific branch or isolated style layer.
6. Apply anonymity, page-limit, disclosure, ethics, and reproducibility requirements only after checking the selected venue's current policies.
7. Keep the venue-neutral source as the durable research version.

## Result provenance

Every quantitative statement should point to a file under `evaluation/results/`. The canonical interpretation and caveats are recorded in [`docs/EXPERIMENTS.md`](../docs/EXPERIMENTS.md). If a rerun changes a result, investigate and document the cause before updating the manuscript.
