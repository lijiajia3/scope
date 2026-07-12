# -*- coding: utf-8 -*-
"""mechanism_auroc.py — A2: 解释 semantic-entropy / self-consistency 在 CoNLL-EN 上
AUROC<0.5 的机理。崩溃句(K次解码近空集)→集合分布高度一致→entropy≈0, self-consistency≈1
→两基线判为"最可靠", 恰是漏抽最重的句子, 于是系统性反向。
产出 out/mechanism_auroc.json + 供画图的分组信号分布。
"""
import json, os, math
from estimate_v2 import canonicalize
from collections import Counter

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs, en):
    if en:
        g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def signals(decs, en):
    cdec = canonicalize(decs, en)
    sets = [set(d) for d in cdec]
    union = sorted(set().union(*sets)) if sets else []
    freq = Counter()
    for s in sets:
        for e in s:
            freq[e] += 1
    K = len(decs); U = len(freq); N = sum(freq.values())
    f1 = sum(1 for v in freq.values() if v == 1)
    cov_gt = 1 - f1 / N if N else 0.0
    # semantic entropy: 归一化 distinct decode-set 分布熵
    setstr = [json.dumps(sorted(s), ensure_ascii=False) for s in sets]
    c = Counter(setstr); tot = sum(c.values())
    ent = -sum((v/tot)*math.log(v/tot) for v in c.values())/math.log(max(2, tot))
    # self-consistency: 平均 pairwise Jaccard
    js = []
    for i in range(len(sets)):
        for j in range(i+1, len(sets)):
            u = sets[i] | sets[j]
            js.append(len(sets[i] & sets[j])/len(u) if u else 1.0)
    selfc = sum(js)/len(js) if js else 1.0
    return {"union_size": U, "cov_risk": round(1-cov_gt, 4),
            "sem_entropy": round(ent, 4), "self_cons": round(selfc, 4),
            "inv_self_cons": round(1-selfc, 4)}


def run(ds):
    en = (ds == "en")
    recs = {}
    for L in open(os.path.join(OUT, f"full_decodes_{ds}_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    rows = []
    for r in recs.values():
        decs = r["decodes"]; en2 = en
        cdec = canonicalize(decs, en2)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold:
            continue
        missed = [g for g in gold if not hit(g, union, en2)]
        s = signals(decs, en2)
        s["has_miss"] = int(len(missed) > 0)
        s["collapse"] = int(s["union_size"] <= 1)   # 崩溃句判据
        rows.append(s)
    return rows


def summ(rows, grp):
    sub = [r for r in rows if grp(r)]
    if not sub:
        return None
    def m(k): return round(sum(r[k] for r in sub)/len(sub), 4)
    return {"n": len(sub), "cov_risk": m("cov_risk"), "sem_entropy": m("sem_entropy"),
            "self_cons": m("self_cons"), "miss_rate": round(sum(r["has_miss"] for r in sub)/len(sub), 3)}


def main():
    out = {}
    for ds in ("en", "zh"):
        rows = run(ds)
        collapse = summ(rows, lambda r: r["collapse"] == 1)
        healthy = summ(rows, lambda r: r["collapse"] == 0)
        out[ds] = {"collapse": collapse, "healthy": healthy, "rows": rows}
        print(f"\n=== {ds} ===")
        print(f"{'group':<12}{'n':>5}{'miss_rate':>10}{'cov_risk':>10}{'sem_entropy':>12}{'self_cons':>11}")
        for name, g in (("collapse", collapse), ("healthy", healthy)):
            if g:
                print(f"{name:<12}{g['n']:>5}{g['miss_rate']:>10}{g['cov_risk']:>10}"
                      f"{g['sem_entropy']:>12}{g['self_cons']:>11}")
        # 机理判定
        if collapse and healthy:
            print(f"  → 崩溃句 miss_rate={collapse['miss_rate']}(最需预警) 但 "
                  f"sem_entropy={collapse['sem_entropy']}(≈0, 基线判'最确定'), "
                  f"self_cons={collapse['self_cons']}(≈1, 基线判'最一致')")
            print(f"     → 熵/一致性把'自信的沉默'读作可靠, 系统性反向; "
                  f"cov_risk={collapse['cov_risk']}(正确判高风险)")
    json.dump(out, open(os.path.join(OUT, "mechanism_auroc.json"), "w"), ensure_ascii=False, indent=1)
    print("\n[done] -> out/mechanism_auroc.json")


if __name__ == "__main__":
    main()
