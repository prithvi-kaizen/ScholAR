# ScholAR: EACL 2027 System Demonstration Paper

This directory contains the manuscript sources, figures, tables, style files, and build configuration for the EACL 2027 System Demonstrations submission:

**ScholAR: Inspectable Local-First Question Answering for Scientific Papers**

## Track Requirements
- **Venue**: EACL 2027 System Demonstrations Track
- **Submission Deadline**: September 22, 2026 (23:59 AoE)
- **Review Policy**: Single-Blind (Author names and affiliations included)
- **Length**: Up to 6 content pages (excluding references, limitations, ethics, and appendices)
- **Video Requirement**: Screencast $\le 2.5$ minutes
- **Package Availability**: Open-source repository / live demo link

## Building the Paper
```bash
make compile
```
Or directly:
```bash
pdflatex main.tex
BSTINPUTS=style: bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Installable Demo Package

The execution-ready package and release procedure is specified in
[`RECOMMENDED_PACKAGE_IMPLEMENTATION_PLAN.md`](RECOMMENDED_PACKAGE_IMPLEMENTATION_PLAN.md).
It keeps the public installable Demo release separate from the anonymous
Industry artifact and defines the package allowlist, clean-install test,
release tag, paper/video synchronization, and stop-ship gates.

## Directory Structure
- `main.tex`: Root LaTeX document
- `sections/`: Section files (abstract, introduction, related_work, system, demo, evaluation, availability, conclusion)
- `tables/`: Compact result and baseline tables
- `figs/`: TikZ architecture diagram and authentic UI demo screenshot
- `style/`: Official unmodified ACL style files (`acl.sty`, `acl_natbib.bst`)
- `build/`: Build gates (`evaluation_gates.tex`)
