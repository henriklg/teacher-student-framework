# How to compile thesis.tex

## Setup
1. Install texlive: sudo apt install texlive-latex-extra
2. Make sure the editor running Biber, not BibTeX: biber build/%

## Tree
- TODO

## Compile
1. First compile each subfile with pdflatex to generate aux files
2. Compile the thesis.tex with quick build 
3. pdflatex + bibtex + pdflatex (x2) + view pdf
