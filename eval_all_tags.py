# -*- coding: utf-8 -*-
"""eval_all_tags.py — 统一评估器: 对所有存在的 (ds, tag) 解码算全套指标 + bootstrap 95% CI。
覆盖 B1(GLM跨家族)/B2(方差CI)/B4(全集)/B5(温度) 的指标计算。
用法: python3.13 eval_all_tags.py            # 自动检测所有 tag
      python3.13 eval_all_tags.py 7b glm dsv3 # 指定 tag
产出 out/all_tags_metrics.json + 终端表。
"""
import json, os, sys, math, glob, random
from estimate_v2 import estimate_one_v2, canonicalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs, en):
    if en:
        g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def pearson(xs, ys):
    n = len(xs)
    if n < 3: return None
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return round(cov/math.sqrt(vx*vy), 4) if vx*vy > 0 else None


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0]); n1 = sum(labels); n0 = len(labels)-n1
    if n1 == 0 or n0 == 0: return None
    rs = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j+1 < len(pairs) and pairs[j+1][0] == pairs[i][0]: j += 1
        ar = (i+j)/2+1
        for k in range(i, j+1):
            if pairs[k][1] == 1: rs += ar
        i = j+1
    return (rs-n1*(n1+1)/2)/(n1*n0)


def per_sentence(ds, tag):
    p = os.path.join(OUT, f"full_decodes_{ds}_{tag}.jsonl")
    if not os.path.exists(p): return None
    recs = {}
    for L in open(p, encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    recs = list(recs.values())
    if len(recs) < 50: return None
    en = (ds == "en")
    rows = []
    for r in recs:
        decs = r["decodes"]; K = len(decs)
        est = estimate_one_v2(decs, K, en)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold: continue
        tr = sum(1 for g in gold if hit(g, union, en))/len(gold)
        half = cdec[:max(1, K//2)]
        uh = sorted(set().union(*[set(d) for d in half])) if half else []
        missed = [g for g in gold if not hit(g, union, en)]
        samp = sum(1 for g in gold if (not hit(g, uh, en)) and hit(g, union, en))
        rows.append({"cov": est.get("cov_gt", 0.0), "tr": tr, "status": est["status"],
                     "n_gold": len(gold), "n_miss": len(missed), "samp": samp,
                     "has_miss": int(len(missed) > 0)})
    return rows


def metrics(rows, n_boot=1999, seed=0):
    def calc(sub):
        cov = [r["cov"] for r in sub]; tr = [r["tr"] for r in sub]
        cert = [r["tr"] for r in sub if r["status"] == "certified"]
        abst = [r["tr"] for r in sub if r["status"] == "abstain"]
        tot_miss = sum(r["n_miss"] for r in sub); tot_samp = sum(r["samp"] for r in sub)
        m = {
            "calib_r": pearson(cov, tr),
            "gap": (sum(cov)/len(cov) - sum(tr)/len(tr)) if cov else None,
            "cert_recall": (sum(cert)/len(cert)) if cert else None,
            "abst_recall": (sum(abst)/len(abst)) if abst else None,
            "abstain_rate": len(abst)/len(sub),
            "frac_sys": (tot_miss-tot_samp)/max(1, tot_miss),
            "triage_auroc": auroc([r["has_miss"] for r in sub], [1-r["cov"] for r in sub]),
        }
        return m
    point = calc(rows)
    # bootstrap CI
    rng = random.Random(seed); m2 = len(rows)
    boot = {k: [] for k in point}
    for _ in range(n_boot):
        sub = [rows[rng.randrange(m2)] for _ in range(m2)]
        c = calc(sub)
        for k, v in c.items():
            if v is not None: boot[k].append(v)
    out = {}
    for k, v in point.items():
        if v is None:
            out[k] = None; continue
        b = sorted(boot[k])
        lo = b[int(0.025*len(b))] if b else None
        hi = b[int(0.975*len(b))] if b else None
        out[k] = {"est": round(v, 4), "ci": [round(lo, 4), round(hi, 4)] if b else None}
    return out


def main():
    tags = sys.argv[1:] if len(sys.argv) > 1 else None
    if not tags:
        # 自动检测
        found = set()
        for f in glob.glob(os.path.join(OUT, "full_decodes_*_*.jsonl")):
            base = os.path.basename(f)[len("full_decodes_"):-len(".jsonl")]
            parts = base.split("_", 1)
            if len(parts) == 2:
                found.add(parts[1])
        tags = sorted(found)
    print("tags:", tags)
    allm = {}
    for ds in ("zh", "en"):
        allm[ds] = {}
        for tag in tags:
            rows = per_sentence(ds, tag)
            if rows is None: continue
            m = metrics(rows)
            allm[ds][tag] = {"n": len(rows), **m}
            def fmt(k):
                x = m[k]
                if x is None: return "  --  "
                return f"{x['est']:+.3f}[{x['ci'][0]:+.2f},{x['ci'][1]:+.2f}]" if x['ci'] else f"{x['est']:.3f}"
            print(f"{ds}/{tag:<8} n={len(rows):>3}  calib_r={fmt('calib_r')}  cert={fmt('cert_recall')}  "
                  f"abst={fmt('abst_recall')}  sys={fmt('frac_sys')}  triage={fmt('triage_auroc')}")
    json.dump(allm, open(os.path.join(OUT, "all_tags_metrics.json"), "w"), ensure_ascii=False, indent=1)
    print("[done] -> out/all_tags_metrics.json")


if __name__ == "__main__":
    main()
