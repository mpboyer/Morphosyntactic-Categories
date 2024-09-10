all:
	make clean
	make report
	make clean

fall:
	make clean
	make pres
	make clean

report:
	latexmk -pdf --shell-escape report.tex

pres:
	latexmk -pdf --shell-escape pres.tex

clean:
	latexmk -c;
	\rm -f *.bbl;
