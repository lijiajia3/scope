#!/bin/sh
cd ~/Desktop/SCOPE
for tag_model in "32b:Qwen/Qwen2.5-32B-Instruct" "72b:Qwen/Qwen2.5-72B-Instruct"; do
  tag=${tag_model%%:*}; model=${tag_model#*:}
  echo "=== $tag $model ==="
  python3.13 full_experiment.py --dataset zh --n 520 --K 20 --workers 8 --tag $tag --model "$model"
  python3.13 full_experiment.py --dataset en --n 500 --K 20 --workers 8 --tag $tag --model "$model"
done
echo V3_MODELS_DONE
