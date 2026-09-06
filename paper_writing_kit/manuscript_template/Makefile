all:
	pdflatex -interaction=nonstopmode main.tex
	-bibtex main
	pdflatex -interaction=nonstopmode main.tex
	pdflatex -interaction=nonstopmode main.tex

clean:
	rm -f *.aux *.log *.out *.bbl *.blg *.toc
