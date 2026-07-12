#!/bin/sh
cd ~/Desktop/SCOPE
echo "=== B1 GLM-4-9B 跨家族 ==="
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 8 --tag glm --model "THUDM/GLM-4-9B-0414"
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag glm --model "THUDM/GLM-4-9B-0414"
echo "=== B5 温度敏感性 (Qwen-7B, en) ==="
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag t05 --temp 0.5 --model "Qwen/Qwen2.5-7B-Instruct"
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag t12 --temp 1.2 --model "Qwen/Qwen2.5-7B-Instruct"
echo "=== B2 重复采样 (Qwen-7B en, 3组独立K=20) ==="
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag rep1 --model "Qwen/Qwen2.5-7B-Instruct"
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag rep2 --model "Qwen/Qwen2.5-7B-Instruct"
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag rep3 --model "Qwen/Qwen2.5-7B-Instruct"
echo BATCH1_DONE
