# Build the v4 manuscript. Requires pdflatex + bibtex (MiKTeX).
# Usage: make        (or run the three commands below on Windows)

all: main.pdf

main.pdf: main.tex refs.bib $(wildcard sections/*.tex) $(wildcard figures/*.pdf)
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	bibtex main
	pdflatex -interaction=nonstopmode -halt-on-error main.tex
	pdflatex -interaction=nonstopmode -halt-on-error main.tex

clean:
	-del /q main.aux main.bbl main.blg main.log main.out 2>NUL || rm -f main.aux main.bbl main.blg main.log main.out

.PHONY: all clean
