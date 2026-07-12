#!/bin/sh
cd ~/Desktop/SCOPE
while ! grep -q BATCH1_DONE batch1.log 2>/dev/null; do sleep 60; done
echo "=== B3 K=100 深采样 (en 含漏抽句) ==="
python3.13 deepen_k100.py
echo "=== B4 全测试集敏感性 (en minent=1) ==="
python3.13 full_experiment.py --dataset en --n 900 --K 20 --workers 8 --tag enfull --minent 1 --model "Qwen/Qwen2.5-7B-Instruct"
echo "=== B6 verbalized confidence 基线 ==="
python3.13 verbalized.py
echo BATCH2_DONE
