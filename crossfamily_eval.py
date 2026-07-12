# -*- coding: utf-8 -*-
"""crossfamily_eval.py — 跨模型家族稳健性: DeepSeek-V3(dsv3) vs Qwen2.5-7B(7b)。
对每个 (ds, tag) 从 full_decodes_{ds}_{tag}.jsonl 用 v2 估计算 SCOPE 的四组核心指标,
证明校准/弃权/分解/分诊结论在完全不同的模型家族上复现。
产出 out/crossfamily.json + 终端对比表。
运行(解码完成后): python3.13 crossfamily_eval.py
"""
import json, os, math
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
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n1 = sum(labels); n0 = len(labels)-n1
    if n1 == 0 or n0 == 0: return None
    rs = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j+1 < len(pairs) and pairs[j+1][0] == pairs[i][0]: j += 1
        ar = (i+j)/2+1
        for k in range(i, j+1):
            if pairs[k][1] == 1: rs += ar
        i = j+1
    return round((rs-n1*(n1+1)/2)/(n1*n0), 4)


def evaluate(ds, tag):
    p = os.path.join(OUT, f"full_decodes_{ds}_{tag}.jsonl")
    if not os.path.exists(p): return None
    recs = {}
    for L in open(p, encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    recs = list(recs.values())
    if len(recs) < 100: return None
    en = (ds == "en")
    covs, trs = [], []
    cert_tr, abst_tr = [], []
    tot_missed = tot_samp = 0
    labels, risks = [], []
    for r in recs:
        decs = r["decodes"]; K = len(decs)
        est = estimate_one_v2(decs, K, en)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold: continue
        tr = sum(1 for g in gold if hit(g, union, en))/len(gold)
        cov = est.get("cov_gt", 0.0)
        covs.append(cov); trs.append(tr)
        (cert_tr if est["status"] == "certified" else abst_tr).append(tr)
        half = cdec[:max(1, K//2)]
        uh = sorted(set().union(*[set(d) for d in half])) if half else []
        missed = [g for g in gold if not hit(g, union, en)]
        samp = sum(1 for g in gold if (not hit(g, uh, en)) and hit(g, union, en))
        tot_missed += len(missed); tot_samp += samp
        labels.append(int(len(missed) > 0)); risks.append(1-cov)
    return {"n": len(recs),
            "calib_r": pearson(covs, trs),
            "gap": round(sum(covs)/len(covs) - sum(trs)/len(trs), 4),
            "cert_recall": round(sum(cert_tr)/len(cert_tr), 4) if cert_tr else None,
            "abst_recall": round(sum(abst_tr)/len(abst_tr), 4) if abst_tr else None,
            "abstain_gain": round((sum(cert_tr)/len(cert_tr) if cert_tr else 0) -
                                  (sum(abst_tr)/len(abst_tr) if abst_tr else 0), 4),
            "frac_systematic": round(tot_missed/max(1, tot_missed+tot_samp), 3),
            "triage_auroc": auroc(labels, risks)}


def main():
    out = {}
    print(f"{'ds/model':<16}{'n':>5}{'calib_r':>9}{'gap':>8}{'cert':>7}{'abst':>7}{'gain':>7}{'sys%':>7}{'triage':>8}")
    for ds in ("zh", "en"):
        out[ds] = {}
        for tag, name in (("7b", "Qwen-7B"), ("glm", "GLM-4-9B"),
                          ("ling", "Ling-mini-2.0"), ("hunyuan", "Hunyuan-A13B"),
                          ("hymt", "Hunyuan-MT-7B"), ("llama", "Llama-3.1-8B"),
                          ("dsv3", "DeepSeek-V3")):
            m = evaluate(ds, tag)
            out[ds][tag] = m
            if m:
                print(f"{ds+'/'+name:<16}{m['n']:>5}{str(m['calib_r']):>9}{m['gap']:>8}"
                      f"{str(m['cert_recall']):>7}{str(m['abst_recall']):>7}{m['abstain_gain']:>7}"
                      f"{m['frac_systematic']*100:>6.0f}{str(m['triage_auroc']):>8}")
            else:
                print(f"{ds+'/'+name:<16}  (not ready)")
    json.dump(out, open(os.path.join(OUT, "crossfamily.json"), "w"), ensure_ascii=False, indent=1)
    print("[done] -> out/crossfamily.json")


if __name__ == "__main__":
    main()
