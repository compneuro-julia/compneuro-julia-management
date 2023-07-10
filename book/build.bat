@echo off
python ipynb2latex.py
lualatex main
upmendex -r -c -s main.ist -g main
lualatex main