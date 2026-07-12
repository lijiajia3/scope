# -*- coding: utf-8 -*-
"""P1-4: bootstrap 显著性 — 在 sampling-limited 目标上比较 coverage risk 与
semantic-entropy / self-consistency 的 AUROC 差, 给出 95% CI。
输出供 5.5 节正文引用。复用 est_v2 decode + canonicalize。"""
import json, os, math, random
from collections import Counter
from estimate_v2 import canonicalize
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
random.seed(0)


def hit(g, ents, en):
    g = g.lower().strip()
    return any((g in e.lower().strip()) or (e.lower().strip() in g) for e in ents)


def build(ds):
    en = ds == "en"
    blob = json.load(open(os.path.join(OUT, f"est_v2_{ds}.json")))
    rows = []
    for r in blob["records"]:
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        half = cdec[:max(1, K // 2)]
        uh = sorted(set().union(*[set(d) for d in half])) if half else []
        gold = r["gold"]
        samp = sum(1 for g in gold if (not hit(g, uh, en)) and hit(g, union, en))
        sets = [set(d) for d in cdec]
        js = []
        for i in range(len(sets)):
            for j in range(i + 1, len(sets)):
                u = sets[i] | sets[j]
                js.append(len(sets[i] & sets[j]) / len(u) if u else 1.0)
        sc = sum(js) / len(js) if js else 1.0
        setstr = [json.dumps(sorted(s), ensure_ascii=False) for s in sets]
        c = Counter(setstr); tot = sum(c.values())
        ent = -sum((v / tot) * math.log(v / tot) for v in c.values()) / math.log(max(2, tot))
        rows.append({"y": int(samp > 0), "risk": 1 - r.get("cov_gt", 0.0),
                     "ent": ent, "isc": 1 - sc})
    return rows


def auroc(rows, key):
    pos = [r[key] for r in rows if r["y"]]; neg = [r[key] for r in rows if not r["y"]]
    if not pos or not neg:
        return float("nan")
    h = sum((1 if p > q else 0.5 if p == q else 0) for p in pos for q in neg)
    return h / (len(pos) * len(neg))


def ci(rows, ka, kb, B=2000):
    n = len(rows); diffs = []
    for _ in range(B):
        bs = [rows[random.randrange(n)] for _ in range(n)]
        if not any(r["y"] for r in bs) or all(r["y"] for r in bs):
            continue
        diffs.append(auroc(bs, ka) - auroc(bs, kb))
    diffs.sort()
    return diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]


out = {}
for ds in ("en", "zh"):
    rows = build(ds)
    npos = sum(r["y"] for r in rows)
    ar, ae, ai = auroc(rows, "risk"), auroc(rows, "ent"), auroc(rows, "isc")
    lo_e, hi_e = ci(rows, "risk", "ent")
    lo_i, hi_i = ci(rows, "risk", "isc")
    out[ds] = {"n": len(rows), "n_pos": npos, "auroc_risk": round(ar, 3),
               "auroc_ent": round(ae, 3), "auroc_isc": round(ai, 3),
               "d_ent": round(ar - ae, 3), "ci_ent": [round(lo_e, 3), round(hi_e, 3)],
               "d_isc": round(ar - ai, 3), "ci_isc": [round(lo_i, 3), round(hi_i, 3)]}
    print(f"[{ds}] sampling target: n={len(rows)} pos={npos}")
    print(f"   risk={ar:.3f} entropy={ae:.3f} selfcons={ai:.3f}")
    print(f"   risk-entropy  {ar-ae:+.3f}  95%CI [{lo_e:+.3f},{hi_e:+.3f}]")
    print(f"   risk-selfcons {ar-ai:+.3f}  95%CI [{lo_i:+.3f},{hi_i:+.3f}]")
json.dump(out, open(os.path.join(OUT, "triage_significance.json"), "w"), indent=1)
print("[done] -> out/triage_significance.json")
