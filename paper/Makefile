# -----------------------------------------------------------------------
# Makefile for the venue-neutral ScholAR manuscript
# Usage:
#   make        -> compile to PDF (full 3-pass)
#   make clean  -> remove build artifacts
#   make open   -> open the compiled PDF
# -----------------------------------------------------------------------

MAIN = manuscript
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean open

all:
	$(LATEX) $(MAIN).tex
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN).tex
	$(LATEX) $(MAIN).tex

clean:
	rm -f $(MAIN).aux $(MAIN).bbl $(MAIN).blg $(MAIN).log \
	       $(MAIN).out $(MAIN).toc $(MAIN).lof $(MAIN).lot \
	       $(MAIN).synctex.gz missfont.log

open:
	open $(MAIN).pdf
