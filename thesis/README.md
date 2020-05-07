# How to compile thesis.tex

## Setup
1. Install texlive: sudo apt install texlive-latex-extra 
[How to install LaTex on Ubuntu 18.04](https://linuxconfig.org/how-to-install-latex-on-ubuntu-18-04-bionic-beaver-linux "Latex installation")
2. Set up editor (TexMaker) to export to build folder
3. Make sure the editor running Biber, not BibTeX: biber build/%
4. Configure editor to run quick build: "pdflatex + biblatex + pdflatex (x2) + view pdf"

## Structure tree
```
thesis
    ├── 01-introduction.tex
    ├── 02-background.tex
    ├── 03-methodology.tex
    ├── 04-experiments.tex
    ├── 05-conclusions.tex
    ├── build
    │   ├── 01-introduction.aux
    │   ├── 01-introduction.bcf
    ...
    │   ├── thesis.aux
    │   ├── thesis.lot
    │   ├── thesis.out
    │   ├── thesis.pdf
    │   └── thesis.toc
    ├── duo
    │   ├── duo_forside_example.tex
    │   ├── duoforside.sty
    │   ├── duoforside.tex
    │   ├── duomasterforside.sty
    │   ├── DUO_UiO_segl.eps.bb
    │   ├── DUO_UiO_segl.eps.gz
    │   └── DUO_UiO_segl.png
    ├── figures
    ├── ifimaster
    │   ├── ifimaster.cls
    │   ├── mymaster.pdf
    │   └── mymaster.tex
    ├── README.md
    └── thesis.tex
```

## Compile
1. First compile each subfile with pdflatex to generate aux files in build/
2. Compile the thesis.tex with quick build from setup step 4.


### Requirements:
- TexLive 2017/19 
- Texmaker 5.0.2
- Biblatex 3.8
- Biber 2.12


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
