# paper/ — ScholAR AAAI-27 Paper Folder

This directory contains the LaTeX source for the ScholAR research paper targeting
**AAAI-27** (The Forty-First AAAI Conference on Artificial Intelligence).

---

## Files

| File | Description |
|------|-------------|
| `scholar_aaai27.tex` | **Main LaTeX source** — edit this file to write the paper |
| `scholar_references.bib` | BibTeX bibliography — add all citations here |
| `aaai27.sty` | Approximate AAAI formatting style (replace with official kit when released) |
| `Makefile` | Build shortcuts (`make`, `make clean`, `make open`) |
| `figures/` | Directory for all paper figures (PDF or PNG format) |
| `scholar_aaai27.pdf` | **Compiled output** — viewable PDF |

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
| Official AAAI-27 author kit release | Expected ~June/July 2026 |

> [!IMPORTANT]
> When the **official AAAI-27 author kit** is released at https://aaai.org/conference/aaai/aaai-27/,
> replace `aaai27.sty` with the official `aaai27.sty` from the author kit ZIP. The formatting
> template and section structure in `scholar_aaai27.tex` will remain valid.

---

## Paper Section Guide

The `.tex` file has detailed comment blocks in every section guiding what to write.
Each section comment includes:
- **Target word count / column length**
- **Paragraph structure guide** (what each paragraph should cover)
- **What figures/tables belong** in that section

### Section Outline

```
1. Abstract          (150–200 words, 4 things: problem/gap/method/result)
2. Introduction      (~450 words, contributions as itemized list)
3. Related Work      (RAG, Scientific Doc Understanding, DocQA, Faithfulness)
4. Problem Formulation (formal task definition with notation)
5. Method
   5.1 Document Processing Pipeline
   5.2 Page-Preserving Chunking          ← Algorithm 1
   5.3 BM25-Primary Retrieval
   5.4 Indirect Citation Grounding
   5.5 Study Goal Generation
   5.6 System Architecture
6. Evaluation
   6.1 Retrieval Benchmark
   6.2 Baseline Systems
   6.3 Results                           ← Table 1
   6.4 Ablation Study                    ← Table 2 (to fill)
   6.5 Generation Evaluation (Planned)
7. Discussion
8. Future Work
9. Conclusion
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

---

## Adding References

Add new BibTeX entries to `scholar_references.bib`, then cite them with `\cite{key}`.
The bibliography style is currently set to `plain` (clean numbered references).
It will be switched to `aaai-named` when the official AAAI author kit is available.
