# -*- coding: utf-8 -*-
"""evaluate_v2.py — v2 判决分析(读 out/est_v2_{ds}.json)。相对 v1 增加:
  A 校准: GT/Chao coverage vs true_recall, 全体+certified, Pearson;
  B 弃权: abstain vs certified 真召回;
  C 分诊(重定位): 目标不是"抓所有漏抽"(75%系统性,原则不可测), 而是
    C1 抓"采样遗漏型"漏抽(增K可救回) 的 AUROC;
    C2 漏抽分解: 系统性(never-produced) vs 采样遗漏(sampling-limited);
  D 与基线对比: coverage 信号 vs semantic-entropy(set_entropy)/self-consistency/random。
产出 out/res_v2_{ds}.json。
"""
import json, os, math, argparse, random
from collections import Counter
import config
from estimate_v2 import norm_en, norm_zh, canonicalize

OUT = config.OUT_DIR


def hit(g, obs, en):
    if en:
        g = g.lower()
        return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def auroc(labels, scores):
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    rs = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j+1 < len(pairs) and pairs[j+1][0] == pairs[i][0]:
            j += 1
        ar = (i+j)/2+1
        for k in range(i, j+1):
            if pairs[k][1] == 1:
                rs += ar
        i = j+1
    return round((rs - n1*(n1+1)/2)/(n1*n0), 4)


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return round(cov/math.sqrt(vx*vy), 4) if vx*vy > 0 else None


def review_curve(recs, key, reverse=True, fracs=(0.1, 0.2, 0.3, 0.5)):
    tot = sum(r["_target"] for r in recs)
    if tot == 0:
        return {}
    order = sorted(recs, key=lambda r: (r[key] is None, r[key]), reverse=reverse)
    out = {}
    for f in fracs:
        k = max(1, int(len(order)*f))
        out[f"top{int(f*100)}"] = round(sum(r["_target"] for r in order[:k])/tot, 3)
    return out


def run(ds):
    en = (ds == "en")
    blob = json.load(open(os.path.join(OUT, f"est_v2_{ds}.json"), encoding="utf-8"))
    recs = []
    for r in blob["records"]:
        if "cov_gt" not in r:               # 空记录(U=0)补默认
            r = {**r, "cov_gt": 0.0, "cov_chao": 0.0, "singleton_frac": 1.0}
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        tr = sum(1 for g in gold if hit(g, union, en))/len(gold) if gold else None
        missed = [g for g in gold if not hit(g, union, en)]
        # 采样遗漏型 vs 系统型: 后半解码能否救回前半漏的
        half = cdec[:K//2]
        uh = sorted(set().union(*[set(d) for d in half])) if half else []
        # 系统漏抽 = 最终就漏; 采样遗漏 = 前半漏但最终有
        sampling_missed = sum(1 for g in gold if (not hit(g, uh, en)) and hit(g, union, en))
        # 集合熵(≈semantic entropy 简化) & 自洽(平均 pairwise Jaccard)
        setstr = [json.dumps(sorted(set(d)), ensure_ascii=False) for d in cdec]
        c = Counter(setstr); tot = sum(c.values())
        set_entropy = -sum((v/tot)*math.log(v/tot) for v in c.values())/math.log(max(2, tot))
        # self-consistency: 平均 pairwise Jaccard(越低越不稳)
        sets = [set(d) for d in cdec]
        js = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                u = sets[i] | sets[j]
                js.append(len(sets[i] & sets[j])/len(u) if u else 1.0)
        self_consistency = sum(js)/len(js) if js else 1.0
        recs.append({**r, "true_recall": tr, "n_missed": len(missed),
                     "has_miss": int(len(missed) > 0),
                     "sampling_missed": sampling_missed,
                     "has_sampling_miss": int(sampling_missed > 0),
                     "set_entropy": round(set_entropy, 4),
                     "self_consistency": round(self_consistency, 4),
                     "risk_gt": round(1 - r["cov_gt"], 4)})

    res = {"dataset": ds, "n": len(recs), "K": blob["K"]}
    cert = [r for r in recs if r["status"] == "certified"]
    absta = [r for r in recs if r["status"] == "abstain"]

    # A 校准
    trall = [r["true_recall"] for r in recs if r["true_recall"] is not None]
    A = {"mean_true_recall": round(sum(trall)/len(trall), 4)}
    for est in ("cov_gt", "cov_chao"):
        vv = [(r[est], r["true_recall"]) for r in recs if r["true_recall"] is not None]
        A[f"mean_{est}"] = round(sum(x for x, _ in vv)/len(vv), 4)
        A[f"gap_{est}"] = round(A[f"mean_{est}"] - A["mean_true_recall"], 4)
        A[f"corr_{est}"] = pearson([x for x, _ in vv], [y for _, y in vv])
    res["A_calibration"] = A

    # B 弃权
    B = {"n_certified": len(cert), "n_abstain": len(absta)}
    if cert and absta:
        B["certified_recall"] = round(sum(r["true_recall"] for r in cert)/len(cert), 4)
        B["abstain_recall"] = round(sum(r["true_recall"] for r in absta)/len(absta), 4)
        B["recall_gain"] = round(B["certified_recall"] - B["abstain_recall"], 4)
    res["B_abstain"] = B

    # C 漏抽分解: 以 K/2 为基准预算, 该预算下漏掉的 gold 实体分成两个不相交子集:
    #   systematic     = 全 K 并集仍然没有(再多采样也救不回) = n_missed
    #   sampling_limited = K/2 漏但被后半解码救回(present in full union) = sampling_missed
    # 二者之并 = missed@(K/2), 是一个真正的划分; frac = systematic / missed@(K/2)。
    tot_systematic = sum(r["n_missed"] for r in recs)          # 全 K 并集也没有
    tot_sampling = sum(r["sampling_missed"] for r in recs)     # 半预算漏但全预算救回
    tot_omissions = tot_systematic + tot_sampling             # missed @ (K/2)
    res["C_decomposition"] = {
        "omissions_half_budget": tot_omissions,
        "systematic": tot_systematic,
        "sampling_limited": tot_sampling,
        "frac_systematic": round(tot_systematic/max(1, tot_omissions), 3)}

    # D 分诊: 两个目标 × 多信号
    def triage(target_key):
        labels = [r[target_key] for r in recs]
        rr = [dict(r, _target=r["sampling_missed"] if target_key == "has_sampling_miss" else r["n_missed"]) for r in recs]
        d = {}
        for name, key, rev in (("risk_gt", "risk_gt", True),
                               ("set_entropy", "set_entropy", True),
                               ("inv_self_consistency", "self_consistency", False),
                               ("singleton_frac", "singleton_frac", True)):
            d[name] = {"auroc": auroc(labels, [(-r[key] if not rev else r[key]) for r in recs]),
                       "review": review_curve(rr, key, reverse=rev)}
        # random
        tot = sum(x["_target"] for x in rr)
        rnd = {}
        for f in (0.1, 0.2, 0.3, 0.5):
            acc = 0
            for s in range(5):
                o = rr[:]; random.Random(s).shuffle(o)
                k = max(1, int(len(o)*f)); acc += sum(x["_target"] for x in o[:k])/max(1, tot)
            rnd[f"top{int(f*100)}"] = round(acc/5, 3)
        d["RANDOM"] = {"auroc": 0.5, "review": rnd}
        return d
    res["D_triage_allmiss"] = triage("has_miss")
    res["D_triage_sampling"] = triage("has_sampling_miss")

    json.dump(res, open(os.path.join(OUT, f"res_v2_{ds}.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n==== v2 {ds} | n={res['n']} K={res['K']} ====")
    print(f"A 校准: GT cov={A['mean_cov_gt']} (gap {A['gap_cov_gt']}, r={A['corr_cov_gt']})  "
          f"true_recall={A['mean_true_recall']}")
    if "recall_gain" in B:
        print(f"B 弃权: certified {B['n_certified']}句 recall={B['certified_recall']} "
              f"vs abstain {B['n_abstain']}句 {B['abstain_recall']}  (gain +{B['recall_gain']})")
    dc = res["C_decomposition"]
    print(f"C 分解: 半预算漏抽 {dc['omissions_half_budget']} = 系统性 {dc['systematic']} ({dc['frac_systematic']*100:.0f}%) "
          f"+ 采样遗漏 {dc['sampling_limited']}")
    for tgt, lab in (("D_triage_allmiss", "所有漏抽"), ("D_triage_sampling", "采样遗漏型")):
        print(f"D 分诊[{lab}]:  {'signal':<20}{'AUROC':>7}  top20  top50")
        for k, v in res[tgt].items():
            rv = v["review"]
            print(f"    {k:<20}{str(v['auroc']):>7}  {rv.get('top20','-'):>5}  {rv.get('top50','-'):>5}")
    print(f"[done] -> res_v2_{ds}.json")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zh", "en", "both"], default="both")
    a = ap.parse_args()
    for ds in (["zh", "en"] if a.dataset == "both" else [a.dataset]):
        run(ds)
