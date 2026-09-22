LATEX ?= pdflatex
PYTHON ?= $(shell which python3)

.PHONY: compile clean verify

compile:
	test -f style/acl.sty
	test -f style/acl_natbib.bst
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	BSTINPUTS=style: bibtex main
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex

verify:
	PYTHONPATH=../.. $(PYTHON) ../../evaluation/validate_demo_paper.py --paper-dir .

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
	rm -f sections/*.aux
