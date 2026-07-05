# paper/ — ScholAR AAAI-27 Paper Folder

This directory contains the LaTeX source for the ScholAR research paper targeting
**AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence), formatted
against the **official AAAI-27 Author Kit** (`AuthorKit27/` at the repo root).

---

## Files

| File | Description |
|------|-------------|
| `scholar_aaai27.tex` | **Main LaTeX source** — anonymous submission (`Anonymous Submission`, empty affiliations) |
| `scholar_references.bib` | BibTeX bibliography — add all citations here |
| `aaai2027.sty` | Official AAAI-27 style file (copied from `AuthorKit27/`) — **do not modify** |
| `aaai2027.bst` | Official AAAI-27 BibTeX style (author-year via natbib) — **do not modify** |
| `Makefile` | Build shortcuts (`make`, `make clean`, `make open`) |
| `figures/` | Directory for all paper figures (PDF or PNG format) |
| `scholar_aaai27.pdf` | **Compiled output** — viewable PDF |

If either `.sty`/`.bst` file changes upstream, re-copy from `AuthorKit27/` rather than
editing the copies here — the author kit explicitly forbids modifying the style file.

---

## How to Compile

```bash
# Full compilation (references resolved)
cd paper/
make

# Or manually:
pdflatex scholar_aaai27.tex
bibtex scholar_aaai27
pdflatex scholar_aaai27.tex
pdflatex scholar_aaai27.tex

# Open the PDF
make open
# or
open scholar_aaai27.pdf
```

---

## AAAI-27 Deadlines

| Milestone | Date |
|-----------|------|
| Abstract deadline | **July 21, 2026** |
| Full paper deadline | **July 28, 2026** |

The official author kit is already in use (`AuthorKit27/`, copied into this folder
as `aaai2027.sty`/`aaai2027.bst`) — no placeholder style file remains.

---

## Format Notes (Official Kit Specifics)

- **Two-column, letter-size**, enforced automatically by `aaai2027.sty`. Do not load
  `geometry`, `fullpage`, or any package that alters margins/columns.
- **`hyperref` is explicitly forbidden** by the author kit — do not add it back.
  `times`, `helvet`, `courier` are loaded automatically by the style; do not
  `\usepackage` them directly.
- **Citations are author-year via `natbib` + `aaai2027.bst`** (e.g. "(Lewis et al. 2020)"),
  not numbered brackets. Existing `\cite{key}` calls render correctly as-is.
- **Section numbering is on** (`\setcounter{secnumdepth}{1}`) because the paper uses
  `Section~\ref{...}` cross-references throughout. The kit's own default is `0`
  (no numbers) — don't reset this to 0 without first removing/rewriting every
  `Section~\ref{...}` reference in the text.
- **Anonymous submission**: `\usepackage[submission]{aaai2027}` + `\author{Anonymous Submission}`
  + empty `\affiliations{}`. Before uploading, also strip the PDF's metadata with a
  metadata-cleaning tool (the author kit requires this for blind review).
- 7 pages of main content max, up to 2 additional pages reserved *only* for references
  (9 pages total max). Verify main content doesn't spill past page 7 before submitting —
  check where the bibliography actually starts in the compiled PDF, not just the total
  page count.

---

## Current Section Outline

```
Abstract
1  Introduction                          (motivation prose + itemized contributions)
2  Related Work
   2.1  Retrieval-Augmented Generation
   2.2  Scientific Literature RAG Systems
   2.3  Scientific Document Understanding
   2.4  Visual Document Retrieval
   2.5  Multi-Modal and Multi-Document Scientific QA
   2.6  Question Answering over Documents
   2.7  Citation Grounding and Faithfulness
3  Problem Formulation
4  Method
   4.1  Document Processing Pipeline
   4.2  Page-Preserving Chunking          ← Algorithm 1
   4.3  BM25-Primary Retrieval with Heuristic Reranking
   4.4  Indirect Citation Grounding
   4.5  Visual Grounding
   4.6  Multi-Document Extension
   4.7  Study Goal Generation (product feature, not independently benchmarked)
   4.8  Faithfulness Evaluation: SummaC-ZS + Semantic Coverage
   4.9  System Architecture
5  Evaluation
   5.1  Retrieval Benchmark
   5.2  Faithfulness Benchmark
   5.3  Visual Grounding Benchmark
   5.4  Multi-Document Locality Benchmark
   5.5  Local Model Comparison
   5.6  Does Reading the Image Help? A Caption-Only Ablation
   5.7  Baseline Systems
   5.8  Retrieval Results
   5.9  Faithfulness Results
   5.10 Ablation: CFS vs. Retrieval Depth
   5.11 Ablation: CFS by Claim Type
6  Discussion
7  Future Work
8  Conclusion
References
```

---

## Adding Figures

1. Place your figure as a `.pdf` or `.png` file in the `figures/` folder.
2. In the `.tex` file, uncomment the `\includegraphics` line in the relevant figure block.
3. Example:
   ```latex
   \includegraphics[width=\columnwidth]{pipeline}
   ```
   (no extension needed if you use `\graphicspath{{figures/}}` which is already set)

Note: the paper currently has no figures — the earlier caption-only
`fig:pipeline` placeholder (Method section) was removed since it had no
actual image. Add a real pipeline diagram using the steps above if desired.

---

## Adding References

Add new BibTeX entries to `scholar_references.bib`, then cite them with `\cite{key}`.
Bibliography style is `aaai2027.bst` (set automatically by the style file — do not
add a `\bibliographystyle` command), producing author-year citations via `natbib`.

---

## Reproducibility Checklist

AAAI-27 requires a separate Reproducibility Checklist submission (not embedded in
the main paper — check the conference's submission form for where to upload it).
See `AuthorKit27/ReproducibilityChecklist.tex` for the template.
