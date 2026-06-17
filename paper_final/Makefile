.PHONY: html docx pdf clean

html:
	pandoc main.tex --standalone --embed-resources --mathjax --resource-path=. -o build/specguard_chem_review.html

docx:
	pandoc main.tex --resource-path=. -o build/specguard_chem_review.docx

pdf:
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex
	pdflatex -interaction=nonstopmode -halt-on-error -output-directory build main.tex

clean:
	rm -f build/*.aux build/*.log build/*.out build/*.toc
