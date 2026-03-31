# Makefile for Autonomous Rocket AI OS Research Paper

.PHONY: all pdf clean html

all: pdf

pdf: Autonomous_Rocket_AI_OS_Research_Paper.pdf

Autonomous_Rocket_AI_OS_Research_Paper.pdf: Autonomous_Rocket_AI_OS_Research_Paper.tex references.bib
	@echo "Building PDF with pdflatex..."
	pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex
	bibtex Autonomous_Rocket_AI_OS_Research_Paper
	pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex
	pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex
	@echo "PDF built successfully!"

clean:
	@echo "Cleaning generated files..."
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
	rm -f Autonomous_Rocket_AI_OS_Research_Paper.pdf

# Alternative: Build with pandoc if LaTeX not available
html: Autonomous_Rocket_AI_OS_Research_Paper.html

Autonomous_Rocket_AI_OS_Research_Paper.html: Autonomous_Rocket_AI_OS_Research_Paper.tex references.bib
	@echo "Building HTML with pandoc..."
	pandoc Autonomous_Rocket_AI_OS_Research_Paper.tex --bibliography=references.bib --citeproc -s -o Autonomous_Rocket_AI_OS_Research_Paper.html
	@echo "HTML built successfully!"

# Show help
help:
	@echo "Available targets:"
	@echo "  make all/pdf   - Build PDF version of the paper"
	@echo "  make html      - Build HTML version of the paper"
	@echo "  make clean     - Clean generated files"
	@echo "  make help      - Show this help"