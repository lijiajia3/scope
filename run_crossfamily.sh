#!/bin/sh
cd ~/Desktop/SCOPE
echo "=== DeepSeek-V3 cross-family validation ==="
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 8 --tag dsv3 --model "deepseek-ai/DeepSeek-V3"
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag dsv3 --model "deepseek-ai/DeepSeek-V3"
echo CROSSFAMILY_DONE
