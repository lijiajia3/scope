# -*- coding: utf-8 -*-
"""
第 1 步：重复随机解码抽取。设备自适应：CUDA(4070) > MPS > CPU。
对每条句子采样解码 K 次(GPU 上一次前向出 K 个样本)，得到 K 个实体集合。
产出 out/decodes.json: [{"id","text","entities":[...gold...],"decodes":[[...],[...]]}]
"""
import os, sys, re, json, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from data import load
from model_utils import build_generator, pick_device


def parse_entities(raw):
    raw = re.split(r"\n\n|解释|说明|注[:：]", raw)[0]
    parts = re.split(r"[、,，;；\n]", raw)
    seen, out = set(), []
    for p in parts:
        p = p.strip(" ：:·-.。()（）0123456789、")
        if p and p not in seen:
            seen.add(p); out.append(p)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="用 OntoSpan 真实 gold（若存在）")
    ap.add_argument("--n", type=int, default=12, help="句子数（4070 上可 200+）")
    ap.add_argument("--K", type=int, default=6, help="每句解码次数（4070 上可 12~20）")
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument("--load_4bit", action="store_true", help="4070 上跑 7B 时开(需 bitsandbytes)")
    a = ap.parse_args()

    data, src = load(a.real, a.n)
    print(f"[data] {len(data)} sents from {src} | K={a.K} temp={a.temperature} dev={pick_device()}", flush=True)
    gen_many = build_generator(config.MODEL_NAME, config.EXTRACT_SYS, load_4bit=a.load_4bit)

    recs, t0 = [], time.time()
    for i, item in enumerate(data):
        raws = gen_many(item["text"], a.K, a.temperature)
        decodes = [parse_entities(r) for r in raws]
        recs.append({"id": item["id"], "text": item["text"],
                     "entities": item["entities"], "decodes": decodes})
        distinct = len(set().union(*[set(d) for d in decodes])) if decodes else 0
        print(f"  {i+1}/{len(data)} {item['id']} gold={len(item['entities'])} "
              f"distinct_over_K={distinct}", flush=True)
    json.dump({"model": config.MODEL_NAME, "K": a.K, "temperature": a.temperature,
               "records": recs}, open(config.DECODES, "w"), ensure_ascii=False, indent=2)
    print(f"[done] {len(recs)} records -> {config.DECODES}  ({time.time()-t0:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
