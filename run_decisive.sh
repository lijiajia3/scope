#!/bin/sh
cd ~/Desktop/SCOPE
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 8
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8
python3.13 evaluate_full.py --dataset both
echo DECISIVE_DONE
