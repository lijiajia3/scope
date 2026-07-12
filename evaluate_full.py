# -*- coding: utf-8 -*-
"""evaluate_full.py — 决定性实验的判决分析(读 out/full_estimates_{ds}.json)。

三组证据 + go/no-go 判决:
  A 校准: self_coverage(f1/K 粗估与 Chao-Jost)vs 真召回;逐句相关
  B 弃权: abstain 组 vs certified 组的真召回
  C 分诊(决定性): 逐句"是否有漏抽"的 AUROC + 复核 top-X% 捞回漏抽实体的比例,
    SCOPE 信号(missing_mass) vs 基线(random/一致性/集合熵/union大小/singleton占比)

用法: python3.13 evaluate_full.py --dataset zh   (或 en / both)
产出: out/full_results_{ds}.json + 终端判决
"""
import json, os, math, argparse, random
from collections import Counter
import config

OUT = config.OUT_DIR


def match_hit(g, observed, en=False):
    if en:
        g = g.lower()
        return any((g in o.lower()) or (o.lower() in g) for o in observed)
    return any((g in o) or (o in g) for o in observed)


def recall_of(gold, observed, en=False):
    if not gold:
        return None
    return sum(1 for g in gold if match_hit(g, observed, en)) / len(gold)


def missed_of(gold, observed, en=False):
    return [g for g in gold if not match_hit(g, observed, en)]


def auroc(labels, scores):
    """label 1 = 有漏抽(高风险), score 越大越该排前。"""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n1 = sum(labels); n0 = len(labels) - n1
    if n1 == 0 or n0 == 0:
        return None
    rank_sum = 0.0; i = 0
    while i < len(pairs):
        j = i
        while j + 1 < len(pairs) and pairs[j + 1][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + j) / 2 + 1
        for k in range(i, j + 1):
            if pairs[k][1] == 1:
                rank_sum += avg_rank
        i = j + 1
    return round((rank_sum - n1 * (n1 + 1) / 2) / (n1 * n0), 4)


def review_curve(recs, key, fracs=(0.1, 0.2, 0.3, 0.5), reverse=True):
    """按信号排序复核 top-X%,捞回全部漏抽实体的比例。"""
    total_missed = sum(len(r["missed"]) for r in recs)
    if total_missed == 0:
        return {}
    order = sorted(recs, key=lambda r: (r[key] is None, r[key]), reverse=reverse)
    out = {}
    for f in fracs:
        k = max(1, int(len(order) * f))
        caught = sum(len(r["missed"]) for r in order[:k])
        out[f"top{int(f*100)}"] = {"reviewed": k,
                                   "caught": caught,
                                   "recall_of_missed": round(caught / total_missed, 3)}
    return out


def signals(rec):
    """从解码频次结构算全部 label-free 信号(风险方向:越大越危险)。"""
    K = rec["K"]; U = rec["U"]
    freq_vals = rec.get("_freqs")
    mm = rec["missing_mass"]
    s = {
        "scope_missing_mass": mm,
        "singleton_frac": rec["singleton_frac"],
        "union_size": U,
        "inconsistency": None,   # 1 - mean(freq/K)
        "set_entropy": None,     # 解码集合分布的归一化熵
    }
    if freq_vals:
        s["inconsistency"] = round(1 - sum(freq_vals) / (len(freq_vals) * K), 4)
    sets = rec.get("_sets")
    if sets:
        c = Counter(sets)
        tot = sum(c.values())
        H = -sum((v / tot) * math.log(v / tot) for v in c.values())
        s["set_entropy"] = round(H / math.log(max(2, tot)), 4)
    if rec.get("self_cov_chao") is not None:
        s["scope_chao_risk"] = round(1 - rec["self_cov_chao"], 4)
    return s


def run(ds):
    est = json.load(open(os.path.join(OUT, f"full_estimates_{ds}.json"), encoding="utf-8"))
    dec_path = os.path.join(OUT, f"full_decodes_{ds}.jsonl")
    raw = {}
    for L in open(dec_path, encoding="utf-8"):
        r = json.loads(L); raw[r["id"]] = r["decodes"]
    en = (ds == "en")

    recs = []
    for r in est["records"]:
        decodes = raw.get(r["id"], [])
        K = len(decodes) or r["K"]
        freq = Counter()
        for d in decodes:
            for e in set(d):
                freq[e] += 1
        rec = dict(r)
        rec["K"] = K
        rec["_freqs"] = list(freq.values())
        rec["_sets"] = [json.dumps(sorted(set(d)), ensure_ascii=False) for d in decodes]
        rec["true_recall"] = recall_of(r["gold"], r["observed_union"], en)
        rec["missed"] = missed_of(r["gold"], r["observed_union"], en)
        rec["has_miss"] = int(len(rec["missed"]) > 0)
        rec.update(signals(rec))
        recs.append(rec)

    res = {"dataset": ds, "model": est["model"], "K": est["K"], "n": len(recs)}

    # ---- A 校准 ----
    cert = [r for r in recs if r["status"] == "certified"]
    A = {}
    if cert:
        mc = lambda k: round(sum(r[k] for r in cert if r[k] is not None) /
                             max(1, sum(1 for r in cert if r[k] is not None)), 4)
        A = {"n_certified": len(cert),
             "mean_self_cov_f1K": mc("self_coverage"),
             "mean_self_cov_chao": mc("self_cov_chao"),
             "mean_true_recall": mc("true_recall"),
             "gap_f1K": round(mc("self_coverage") - mc("true_recall"), 4),
             "gap_chao": round(mc("self_cov_chao") - mc("true_recall"), 4)}
        xs = [r["self_coverage"] for r in cert]; ys = [r["true_recall"] for r in cert]
        mx, my = sum(xs)/len(xs), sum(ys)/len(ys)
        cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
        vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
        A["pearson_selfcov_vs_recall"] = round(cov / math.sqrt(vx*vy), 4) if vx*vy > 0 else None
    res["A_calibration"] = A

    # ---- B 弃权 ----
    absta = [r for r in recs if r["status"] == "abstain"]
    B = {"n_abstain": len(absta), "n_certified": len(cert)}
    if absta and cert:
        B["abstain_mean_true_recall"] = round(sum(r["true_recall"] for r in absta)/len(absta), 4)
        B["certified_mean_true_recall"] = round(sum(r["true_recall"] for r in cert)/len(cert), 4)
    res["B_abstain"] = B

    # ---- C 分诊(决定性) ----
    labels = [r["has_miss"] for r in recs]
    C = {"n": len(recs), "n_with_miss": sum(labels),
         "total_missed_entities": sum(len(r["missed"]) for r in recs),
         "signals": {}}
    sig_keys = ["scope_missing_mass", "scope_chao_risk", "singleton_frac",
                "inconsistency", "set_entropy", "union_size"]
    for k in sig_keys:
        vals = [r.get(k) for r in recs]
        if all(v is None for v in vals):
            continue
        vv = [(0.0 if v is None else v) for v in vals]
        C["signals"][k] = {"auroc": auroc(labels, vv),
                           "review": review_curve(
                               [dict(r, **{k: (0.0 if r.get(k) is None else r[k])}) for r in recs], k)}
    # random 基线(期望 = X%),用 5 次重排的均值近似
    rnd = {}
    for f in (0.1, 0.2, 0.3, 0.5):
        tot = sum(len(r["missed"]) for r in recs)
        acc = 0
        for seed in range(5):
            o = recs[:]; random.Random(seed).shuffle(o)
            k = max(1, int(len(o)*f))
            acc += sum(len(r["missed"]) for r in o[:k]) / max(1, tot)
        rnd[f"top{int(f*100)}"] = round(acc/5, 3)
    C["random_baseline"] = rnd
    res["C_triage"] = C

    dst = os.path.join(OUT, f"full_results_{ds}.json")
    slim = {k: v for k, v in res.items()}
    json.dump(slim, open(dst, "w"), ensure_ascii=False, indent=1)

    # ---- 终端判决 ----
    print(f"\n======== {ds} | n={res['n']} | K={res['K']} | {res['model']} ========")
    if A:
        print(f"A 校准: self_cov(f1/K)={A['mean_self_cov_f1K']}  chao={A['mean_self_cov_chao']}  "
              f"true_recall={A['mean_true_recall']}  gap(f1K)={A['gap_f1K']}  "
              f"r={A['pearson_selfcov_vs_recall']}")
    if "abstain_mean_true_recall" in B:
        print(f"B 弃权: abstain={B['n_abstain']} 句 recall={B['abstain_mean_true_recall']}  "
              f"vs certified={B['n_certified']} 句 {B['certified_mean_true_recall']}")
    print(f"C 分诊: {C['n_with_miss']}/{C['n']} 句有漏抽, 共 {C['total_missed_entities']} 个漏抽实体")
    print(f"  {'signal':<20}{'AUROC':>7}   top10  top20  top30  top50")
    for k, v in C["signals"].items():
        rv = v["review"]
        row = "  ".join(f"{rv.get(f'top{p}',{}).get('recall_of_missed','-'):>5}" for p in (10,20,30,50))
        print(f"  {k:<20}{str(v['auroc']):>7}   {row}")
    row = "  ".join(f"{rnd[f'top{p}']:>5}" for p in (10, 20, 30, 50))
    print(f"  {'RANDOM':<20}{'0.5':>7}   {row}")
    print(f"[done] -> {dst}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zh", "en", "both"], default="both")
    a = ap.parse_args()
    for ds in (["zh", "en"] if a.dataset == "both" else [a.dataset]):
        if os.path.exists(os.path.join(OUT, f"full_estimates_{ds}.json")):
            run(ds)
        else:
            print(f"[skip] {ds}: no estimates yet")
