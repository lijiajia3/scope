# -*- coding: utf-8 -*-
"""数据加载。默认用自带 sample_gold.json；--real 时若存在 OntoSpan gold 则用真数据。
统一格式：[{"id","text","entities":[...]}]。可独立运行做自检。"""
import json, os, random, argparse
import config


def _spans(item):
    s = set()
    for ann in item.get("annotations", []):
        for r in ann.get("result", []):
            if r.get("type") in ("labels", "hypertextlabels"):
                t = r.get("value", {}).get("text")
                if isinstance(t, list):
                    t = "".join(map(str, t))
                if isinstance(t, str) and t.strip():
                    s.add(t.strip())
    return sorted(s)


def _text(item):
    d = item.get("data", {}) or {}
    for k in ("text", "content", "sentence"):
        if isinstance(d.get(k), str) and d[k].strip():
            return d[k].strip()
    return ""


def load(use_real=False, n=None, min_ents=3, seed=7):
    """返回统一格式的样本列表。use_real=True 且真实 gold 存在时用真数据。"""
    if use_real and os.path.exists(config.REAL_GOLD):
        raw = json.load(open(config.REAL_GOLD, encoding="utf-8"))
        rows = [(_text(it), _spans(it)) for it in raw]
        rows = [(t, e) for t, e in rows if t and len(e) >= min_ents and len(t) < 200]
        random.Random(seed).shuffle(rows)
        data = [{"id": f"r{i:04d}", "text": t, "entities": e} for i, (t, e) in enumerate(rows)]
        src = config.REAL_GOLD
    else:
        data = json.load(open(config.SAMPLE_GOLD, encoding="utf-8"))
        src = config.SAMPLE_GOLD
    if n:
        data = data[:n]
    return data, src


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true", help="尝试用 OntoSpan 真实 gold")
    ap.add_argument("--n", type=int, default=None)
    a = ap.parse_args()
    d, src = load(a.real, a.n)
    print(f"[data] {len(d)} items from {src}")
    ex = d[0]
    print(f"  例: id={ex['id']}  text={ex['text'][:40]}...")
    print(f"      gold entities({len(ex['entities'])}): {ex['entities']}")
    print(f"  平均实体/句 = {round(sum(len(x['entities']) for x in d)/len(d), 2)}")
