# -*- coding: utf-8 -*-
"""P1-5: 标准 exact-match P/R/F1, 与论文正文用的 containment 匹配对照。
对 pooled union 和 stable set (>=20% decode) 两种输出各算一遍, 便于与
CoNLL-2003 / 临床 NER 文献的 exact-match 结果比较。复用已保存的 decode。"""
import json, os, re
from estimate_v2 import canonicalize
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def norm(s, en):
    s = s.strip().lower()
    if en:
        s = re.sub(r"^(the|a|an)\s+", "", s)
        s = s.strip(" .,\"'`-")
    else:
        s = s.strip(" 。.、，\"'`-")
    return s


def prf_exact(pred, gold, en):
    gp = {norm(g, en) for g in gold}
    pp = {norm(p, en) for p in pred if norm(p, en)}
    tp = len(pp & gp)
    P = tp / len(pp) if pp else 0.0
    R = tp / len(gp) if gp else 0.0
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return tp, len(pp), len(gp), P, R, F


def stable_set(cdec, thr=0.2):
    K = len(cdec)
    from collections import Counter
    c = Counter()
    for d in cdec:
        for e in set(d):
            c[e] += 1
    return [e for e, n in c.items() if n >= thr * K]


def run(ds):
    en = ds == "en"
    blob = json.load(open(os.path.join(OUT, f"est_v2_{ds}.json")))
    agg = {"union": [0, 0, 0], "stable": [0, 0, 0]}  # tp, npred, ngold (micro)
    for r in blob["records"]:
        cdec = canonicalize(r["decodes"], en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        stab = stable_set(cdec)
        gold = r["gold"]
        for key, pred in (("union", union), ("stable", stab)):
            tp, npred, ngold, *_ = prf_exact(pred, gold, en)
            agg[key][0] += tp; agg[key][1] += npred; agg[key][2] += ngold
    res = {}
    for key, (tp, npred, ngold) in agg.items():
        P = tp / npred if npred else 0.0
        R = tp / ngold if ngold else 0.0
        F = 2 * P * R / (P + R) if (P + R) else 0.0
        res[key] = {"P": round(P, 3), "R": round(R, 3), "F1": round(F, 3)}
    return res


out = {}
for ds in ("zh", "en"):
    out[ds] = run(ds)
    print(f"[{ds}] exact-match (micro):")
    for key in ("union", "stable"):
        m = out[ds][key]
        print(f"    {key:<7} P={m['P']} R={m['R']} F1={m['F1']}")
json.dump(out, open(os.path.join(OUT, "exact_match.json"), "w"), indent=1)
print("[done] -> out/exact_match.json")
