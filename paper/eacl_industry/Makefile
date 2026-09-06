LATEX ?= pdflatex
PYTHON ?= ../../.venv/bin/python

.PHONY: verify draft compile submission-build submission clean

verify:
	$(PYTHON) ../../evaluation/prepare_paper_tables.py --paper-dir . --pending-ok
	$(PYTHON) ../../evaluation/validate_paper.py --paper-dir .

compile:
	test -f style/acl.sty
	test -f style/acl_natbib.bst
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	BSTINPUTS=style: bibtex main
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex
	$(LATEX) -interaction=nonstopmode -halt-on-error main.tex

draft: verify compile

submission-build:
	$(PYTHON) ../../evaluation/prepare_paper_tables.py --paper-dir .
	$(MAKE) compile
	$(PYTHON) ../../evaluation/validate_submission_pdf.py --paper-dir . --output build/pdf_compliance_report.json

submission: submission-build
	$(PYTHON) ../../evaluation/validate_paper.py --paper-dir . --submission

clean:
	rm -f main.aux main.bbl main.blg main.log main.out main.pdf
	rm -f build/release_table_gate.json build/release_table_gate.tex build/pdf_compliance_report.json
