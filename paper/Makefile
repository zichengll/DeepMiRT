TEX = main
FIGURES = figures/fig1_architecture.pdf figures/fig2_benchmark.pdf figures/fig3_performance.pdf

.PHONY: all clean figures tectonic

# Default: use pdflatex + bibtex
all: $(TEX).pdf

$(TEX).pdf: $(TEX).tex references.bib $(FIGURES)
	@if command -v tectonic >/dev/null 2>&1; then \
		echo "Building with tectonic..."; \
		tectonic -X compile $(TEX).tex; \
	elif command -v pdflatex >/dev/null 2>&1; then \
		echo "Building with pdflatex..."; \
		pdflatex -interaction=nonstopmode $(TEX); \
		bibtex $(TEX); \
		pdflatex -interaction=nonstopmode $(TEX); \
		pdflatex -interaction=nonstopmode $(TEX); \
	else \
		echo "Error: neither tectonic nor pdflatex found. Install one of them."; \
		exit 1; \
	fi

figures/fig1_architecture.pdf: figures/fig1_architecture.py
	python $<

figures/fig2_benchmark.pdf: figures/fig2_benchmark.py
	python $<

figures/fig3_performance.pdf: figures/fig3_performance.py
	python $<

figures: $(FIGURES)

clean:
	rm -f $(TEX).aux $(TEX).bbl $(TEX).blg $(TEX).log $(TEX).out $(TEX).pdf $(TEX).toc
