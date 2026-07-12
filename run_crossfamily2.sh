#!/bin/bash
# 跨家族解码: 用 SiliconFlow 上可用的非-Qwen 家族, 直接回应审稿人"只测一个家族"。
# Ling-mini-2.0 (inclusionAI, 小MoE, 最可能落 graded 区) + Hunyuan-A13B (Tencent, 另一家族)。
cd ~/Desktop/SCOPE
export PYTHONUNBUFFERED=1

# EN 是 regime 判别关键(Qwen-7B r=0.86 vs GLM/DeepSeek r≈0), 两个模型 EN 先跑。
echo "=== [1/4] Ling-mini-2.0 EN ==="
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 12 \
  --model inclusionAI/Ling-mini-2.0 --tag ling 2>&1 | tail -3

echo "=== [2/4] Hunyuan-A13B EN ==="
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 12 \
  --model tencent/Hunyuan-A13B-Instruct --tag hunyuan 2>&1 | tail -3

echo "=== [3/4] Ling-mini-2.0 ZH ==="
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 12 \
  --model inclusionAI/Ling-mini-2.0 --tag ling 2>&1 | tail -3

echo "=== [4/4] Hunyuan-A13B ZH ==="
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 12 \
  --model tencent/Hunyuan-A13B-Instruct --tag hunyuan 2>&1 | tail -3

echo "=== crossfamily eval ==="
python3.13 crossfamily_eval.py 2>&1 | grep -vE "Warning"
echo "=== ALL_CROSSFAMILY2_DONE ==="
