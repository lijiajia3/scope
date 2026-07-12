#!/bin/sh
cd ~/Desktop/SCOPE
while ! grep -q V3_MODELS_DONE v3_models.log 2>/dev/null; do sleep 60; done
python3.13 figs_models.py > figs_models_final.log 2>&1
cp out/figs/F15*.pdf out/figs/F16*.pdf paper/figs/
cd paper && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1 && pdflatex -interaction=nonstopmode main.tex > /dev/null 2>&1
echo V3_POSTPROCESS_DONE
