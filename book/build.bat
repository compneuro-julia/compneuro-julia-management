@echo off
python ipynb2latex.py
lualatex main
upbibtex main
upmendex -r -c -s main.ist -g main
lualatex main
lualatex main