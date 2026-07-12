# -*- coding: utf-8 -*-
"""R3-3: abstention 两个阈值的敏感性扫描 (审稿人: 之前只扫了 K)。
弃权规则: abstain if K<5 or singleton_frac>=s_thr or cov_gt<c_thr。
扫 s_thr∈{0.5,0.6,0.7} × c_thr∈{0.4,0.5,0.6}, 报 certified/abstained recall +
gain + 认证比例, 证明分离对这两个超参稳健。"""
import json, os
from estimate_v2 import canonicalize, estimate_one_v2
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs, en):
    if en:
        g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def load(ds):
    en = ds == "en"
    recs = {}
    for L in open(os.path.join(OUT, f"full_decodes_{ds}_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    items = []
    for r in recs.values():
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold:
            continue
        tr = sum(1 for g in gold if hit(g, union, en)) / len(gold)
        est = estimate_one_v2(decs, K, en)
        items.append((est.get("singleton_frac", 1.0), est.get("cov_gt", 0.0), K, tr))
    return items


def evaluate(items, s_thr, c_thr):
    cert, abst = [], []
    for sf, cov, K, tr in items:
        if K < 5 or sf >= s_thr or cov < c_thr:
            abst.append(tr)
        else:
            cert.append(tr)
    cr = sum(cert) / len(cert) if cert else 0.0
    ar = sum(abst) / len(abst) if abst else 0.0
    return cr, ar, cr - ar, len(cert) / (len(cert) + len(abst))


out = {}
for ds in ("en", "zh"):
    items = load(ds)
    out[ds] = {}
    print(f"=== {ds} (n={len(items)}) | 默认 s=0.6 c=0.5 ===")
    print(f"{'s_thr':>6}{'c_thr':>6}{'certR':>8}{'abstR':>8}{'gain':>8}{'cert%':>7}")
    for s in (0.5, 0.6, 0.7):
        for c in (0.4, 0.5, 0.6):
            cr, ar, g, frac = evaluate(items, s, c)
            out[ds][f"s{s}_c{c}"] = {"certR": round(cr, 3), "abstR": round(ar, 3),
                                     "gain": round(g, 3), "cert_frac": round(frac, 3)}
            star = " *" if (s == 0.6 and c == 0.5) else ""
            print(f"{s:>6}{c:>6}{cr:>8.3f}{ar:>8.3f}{g:>8.3f}{frac:>7.2f}{star}")
    gains = [v["gain"] for v in out[ds].values()]
    print(f"  gain 范围: [{min(gains):.3f}, {max(gains):.3f}]  极差 {max(gains)-min(gains):.3f}")
json.dump(out, open(os.path.join(OUT, "threshold_sensitivity.json"), "w"), indent=1)
print("[done] -> out/threshold_sensitivity.json")
