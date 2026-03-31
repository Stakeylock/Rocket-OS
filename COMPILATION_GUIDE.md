# Compilation Guide for Autonomous Rocket AI OS Research Paper

This guide explains how to compile the LaTeX version of the research paper into PDF format.

## Prerequisites

To compile the LaTeX paper, you need:

1. **LaTeX Distribution** (TeX Live, MikTeX, or MacTeX)
2. **BibTeX** (usually included with LaTeX distributions)
3. **Make** (optional, for using the provided Makefile)

## Method 1: Using the Makefile (Recommended)

If you have `make` installed:

```bash
# Navigate to the paper directory
cd /path/to/ros/directory

# Compile the paper
make all

# This will generate:
# - Autonomous_Rocket_AI_OS_Research_Paper.pdf
```

To clean generated files:

```bash
make clean
```

## Method 2: Manual Compilation

If you prefer to compile manually or don't have `make`:

```bash
# Navigate to the paper directory
cd /path/to/ros/directory

# Run pdflatex (first pass)
pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex

# Run bibtex to process references
bibtex Autonomous_Rocket_AI_OS_Research_Paper

# Run pdflatex second time (to resolve references)
pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex

# Run pdflatex third time (to resolve table of contents, etc.)
pdflatex -interaction=nonstopmode Autonomous_Rocket_AI_OS_Research_Paper.tex

# The PDF will be generated as:
# Autonomous_Rocket_AI_OS_Research_Paper.pdf
```

## Method 3: Using Pandoc (Alternative)

If LaTeX is not available but you have Pandoc:

```bash
# Convert to HTML
pandoc Autonomous_Rocket_AI_OS_Research_Paper.tex --bibliography=references.bib --citeproc -s -o Autonomous_Rocket_AI_OS_Research_Paper.html

# Convert to PDF (requires LaTeX to be installed in the background)
pandoc Autonomous_Rocket_AI_OS_Research_Paper.tex --bibliography=references.bib --citeproc -o Autonomous_Rocket_AI_OS_Research_Paper.pdf
```

## Troubleshooting

### Missing Packages

If you encounter missing LaTeX package errors, install the required packages using your TeX distribution's package manager:

- For TeX Live: `tlmgr install <package-name>`
- For MikTeX: Use the MikTeX Console
- For MacTeX: Use the TeX Live Utility

Commonly needed packages:
- `geometry`
- `graphicx`
- `booktabs`
- `hyperref`
- `xcolor`
- `amsmath, amssymb, amsfonts`

### Compilation Errors

If you get compilation errors:

1. Check that all files are in the same directory:
   - `Autonomous_Rocket_AI_OS_Research_Paper.tex`
   - `references.bib`

2. Ensure the `.tex` file doesn't have syntax errors by checking the log file:
   - `Autonomous_Rocket_AI_OS_Research_Paper.log`

3. Try deleting auxiliary files and recompiling:
   ```bash
   rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lof *.lot
   ```

## Expected Output

Successful compilation will produce:
- `Autonomous_Rocket_AI_OS_Research_Paper.pdf` - The formatted research paper ready for submission
- Auxiliary files: `.aux`, `.bbl`, `.blg`, `.log`, `.out`, `.toc` (can be safely deleted)

## Paper Specifications

- **Format**: IEEE Conference Paper (10pt)
- **Length**: Approximately 6-8 pages when compiled
- **Bibliography**: IEEEtran style with 6 references
- **Figures**: Referenced in text (actual figures would need to be included separately)