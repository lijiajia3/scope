#!/bin/bash
# Hunyuan-MT-7B (tencent, 7B翻译模型) — 最小的可及非-Qwen通用模型, 冲 graded 区。
# 独立 tag=hymt, 与主批次不同文件, 不冲突。workers=6 避免与主批次抢并发。
cd ~/Desktop/SCOPE
export PYTHONUNBUFFERED=1
echo "=== Hunyuan-MT-7B EN ==="
python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 6 \
  --model tencent/Hunyuan-MT-7B --tag hymt 2>&1 | tail -3
echo "=== Hunyuan-MT-7B ZH ==="
python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 6 \
  --model tencent/Hunyuan-MT-7B --tag hymt 2>&1 | tail -3
echo "=== HYMT_DONE ==="
