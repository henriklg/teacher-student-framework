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


### Requirements:
- TexLive 2017/19 
- Texmaker 5.0.2
- Biblatex 3.8
- Biber 2.12
[How to install LaTex on Ubuntu 18.04 Bionic Beaver Linux](https://linuxconfig.org/how-to-install-latex-on-ubuntu-18-04-bionic-beaver-linux "Latex installation")

```
henrik@X1carbon:~> which pdflatex
/usr/local/texlive/2019/bin/x86_64-linux/pdflatex
henrik@X1carbon:~> which biber
/usr/local/texlive/2019/bin/x86_64-linux/biber
henrik@X1carbon:~> which bibtex
/usr/local/texlive/2019/bin/x86_64-linux/bibtex
henrik@X1carbon:~> pdflatex --version
pdfTeX 3.14159265-2.6-1.40.20 (TeX Live 2019)
henrik@X1carbon:~> biber --version
biber version: 2.12
henrik@X1carbon:~> bibtex --version
BibTeX 0.99d (TeX Live 2019)
```
