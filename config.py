# -*- coding: utf-8 -*-
"""SCOPE 全局配置。所有脚本从这里取路径/常量，避免散落。"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
OUT_DIR = os.path.join(ROOT, "out")
os.makedirs(OUT_DIR, exist_ok=True)

# 自带样本数据（脱离 OntoSpan 也能跑）；若存在真实 gold 路径可用环境变量覆盖。
SAMPLE_GOLD = os.path.join(DATA_DIR, "sample_gold.json")
REAL_GOLD = os.environ.get(
    "SCOPE_GOLD",
    os.path.join(DATA_DIR, "clinical_zh_gold_520.json"),
)

# 本地模型（已缓存、走 MPS/CPU，不联网）。
MODEL_NAME = os.environ.get("SCOPE_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")

# 产物文件
DECODES = os.path.join(OUT_DIR, "decodes.json")       # decode.py 产出
ESTIMATES = os.path.join(OUT_DIR, "estimates.json")   # estimate.py 产出
RESULTS = os.path.join(OUT_DIR, "results.json")       # evaluate.py 产出

# 抽取提示词（中文医疗 NER）
EXTRACT_SYS = (
    "你是医疗信息抽取器。从给定句子中抽取所有医疗命名实体"
    "（药物/疾病/临床表现/身体部位/微生物/医疗程序/检验项目/医疗设备/科室）。"
    "只输出实体，用中文顿号、分隔，不要解释，不要编号，不要重复。"
)
