all:
	make main
	make clean

main:
	latexmk -pdf report.tex

clean:
	latexmk -c
	\rm *.bbl
