# -*- coding: utf-8 -*-
"""diagnose_v2.py — 深度诊断,为 v2 方法迭代定方向。读 out/full_decodes_{ds}.jsonl。

回答四个问题:
 Q1 估计器: f1/K vs Good-Turing(f1/N) vs Chao1-coverage,哪个是方向正确、校准的自覆盖下界?
 Q2 零相关: self-coverage 与 true-recall 为何逐句几乎不相关? (range restriction? 正交?)
 Q3 分诊失败机理: 漏抽实体是"低概率采样遗漏"(增大K可采到)还是"系统永不产出"(任何K都采不到)?
    → 对每句画 union(K') 增长曲线, 看 missed 实体是否在 K 增大时被采回。
 Q4 v2 候选信号: 什么 label-free 量与 has_miss 相关最强?
产出: out/diag_{ds}.json + 终端报告 + 供画图的中间量。
"""
import json, os, math, argparse, random
from collections import Counter
import config

OUT = config.OUT_DIR


def load(ds):
    recs = []
    for L in open(os.path.join(OUT, f"full_decodes_{ds}.jsonl"), encoding="utf-8"):
        recs.append(json.loads(L))
    seen = {}
    for r in recs:
        seen[r["id"]] = r
    return [seen[k] for k in sorted(seen)]


def hit(g, obs, en):
    if en:
        g = g.lower()
        return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def gt_estimators(decodes, K):
    """三套自覆盖估计。"""
    freq = Counter()
    for d in decodes:
        for e in set(d):
            freq[e] += 1
    U = len(freq)
    if U == 0:
        return None
    f1 = sum(1 for v in freq.values() if v == 1)
    f2 = sum(1 for v in freq.values() if v == 2)
    N = sum(freq.values())                       # 总 incidence
    # 1) 原实现 f1/K
    cov_f1K = 1 - f1 / K
    # 2) 标准 Good-Turing coverage: 1 - f1/N
    cov_gt = 1 - f1 / N if N else 0.0
    # 3) Chao-Jost incidence coverage
    if f1 == 0:
        cov_chao = 1.0
    else:
        denom = (K - 1) * f1 + 2 * f2
        cov_chao = 1 - (f1 / N) * ((K - 1) * f1 / denom if denom else 1.0)
    # Chao1 richness (估计真实"物种"总数, 含未见)
    chao1 = U + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    return {"U": U, "f1": f1, "f2": f2, "N": N,
            "cov_f1K": round(cov_f1K, 4), "cov_gt": round(cov_gt, 4),
            "cov_chao": round(cov_chao, 4), "chao1_richness": round(chao1, 2)}


def union_growth(decodes, K):
    """union 大小随 K' 的增长曲线(打乱顺序平均 3 次), 归一化到 K' 处 union/最终union。"""
    finalU = len(set().union(*[set(d) for d in decodes])) if decodes else 0
    if finalU == 0:
        return []
    curve = []
    for Kp in range(1, K + 1):
        accs = []
        for seed in range(3):
            o = decodes[:]; random.Random(seed).shuffle(o)
            u = set().union(*[set(d) for d in o[:Kp]])
            accs.append(len(u))
        curve.append(round(sum(accs) / len(accs) / finalU, 4))
    return curve


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return None
    mx, my = sum(xs)/n, sum(ys)/n
    cov = sum((x-mx)*(y-my) for x, y in zip(xs, ys))
    vx = sum((x-mx)**2 for x in xs); vy = sum((y-my)**2 for y in ys)
    return round(cov/math.sqrt(vx*vy), 4) if vx*vy > 0 else None


def run(ds):
    recs = load(ds)
    en = (ds == "en")
    rows = []
    growth_missed = []   # 每句: (finalU, missed 实体在增大K时是否被采回)
    for r in recs:
        decs = r["decodes"]; K = len(decs)
        gt = gt_estimators(decs, K)
        if gt is None:
            continue
        union = sorted(set().union(*[set(d) for d in decs]))
        gold = r["gold"]
        tr = sum(1 for g in gold if hit(g, union, en)) / len(gold) if gold else None
        missed = [g for g in gold if not hit(g, union, en)]
        # missed 实体: 它们在多少个解码里出现过? (定义上 union 外 = 0 个) → 确认系统漏抽
        # 反过来验证: 若把 K 减半, 有多少 gold 从 covered 变 missed (= 低概率采样遗漏)
        half = decs[:K//2]
        union_half = sorted(set().union(*[set(d) for d in half])) if half else []
        recovered_by_second_half = sum(
            1 for g in gold if (not hit(g, union_half, en)) and hit(g, union, en))
        rows.append({
            "id": r["id"], "n_gold": len(gold), "true_recall": tr,
            "n_missed": len(missed), "has_miss": int(len(missed) > 0),
            "recovered_2nd_half": recovered_by_second_half,
            **gt})
        growth_missed.append({"id": r["id"], "curve": union_growth(decs, K),
                              "n_missed": len(missed)})

    cert = [x for x in rows if x["cov_f1K"] is not None]
    def m(k, sub=cert):
        v = [x[k] for x in sub if x[k] is not None]
        return round(sum(v)/len(v), 4) if v else None

    # Q1 估计器对比: 各自 gap = cov - true_recall
    tr_all = [x["true_recall"] for x in rows if x["true_recall"] is not None]
    report = {"ds": ds, "n": len(rows),
              "mean_true_recall": round(sum(tr_all)/len(tr_all), 4)}
    for est in ("cov_f1K", "cov_gt", "cov_chao"):
        report[f"mean_{est}"] = m(est, rows)
        report[f"gap_{est}"] = round(m(est, rows) - report["mean_true_recall"], 4)
        report[f"corr_{est}_recall"] = pearson(
            [x[est] for x in rows if x["true_recall"] is not None],
            [x["true_recall"] for x in rows if x["true_recall"] is not None])
        # 违反下界方向的句子比例 (cov < true_recall)
        viol = sum(1 for x in rows if x["true_recall"] is not None and x[est] < x["true_recall"])
        report[f"frac_violate_{est}"] = round(viol/len(tr_all), 3)

    # Q3 分诊机理: missed 实体几乎全是 never-produced;
    #   recovered_2nd_half>0 的句子 = 后半采样把前半漏的采回了 = 低概率遗漏存在的证据
    n_recov = sum(x["recovered_2nd_half"] for x in rows)
    total_gold = sum(x["n_gold"] for x in rows)
    total_missed = sum(x["n_missed"] for x in rows)
    report["Q3_sampling_recovery"] = {
        "total_gold": total_gold, "total_missed_final": total_missed,
        "recovered_by_2nd_half": n_recov,
        "interpretation": "recovered_by_2nd_half 相对 total_missed 越小 → missed 越是系统漏抽(增K无救)"}

    # Q4 候选信号 vs has_miss (逐句, 点二列相关)
    labels = [x["has_miss"] for x in rows]
    report["Q4_signal_corr_hasmiss"] = {}
    for k in ("cov_f1K", "cov_gt", "cov_chao", "f1", "U", "N", "chao1_richness"):
        vals = [x[k] for x in rows]
        report["Q4_signal_corr_hasmiss"][k] = pearson(vals, [float(l) for l in labels])

    json.dump({"report": report, "rows": rows, "growth": growth_missed},
              open(os.path.join(OUT, f"diag_{ds}.json"), "w"), ensure_ascii=False, indent=1)

    print(f"\n===== DIAG {ds}: n={report['n']} true_recall={report['mean_true_recall']} =====")
    print(f"{'estimator':<10}{'mean':>8}{'gap':>9}{'corr_r':>9}{'%violate':>10}")
    for est in ("cov_f1K", "cov_gt", "cov_chao"):
        print(f"{est:<10}{report['mean_'+est]:>8}{report['gap_'+est]:>9}"
              f"{str(report['corr_'+est+'_recall']):>9}{report['frac_violate_'+est]:>10}")
    q3 = report["Q3_sampling_recovery"]
    print(f"Q3 分诊机理: 最终漏抽 {q3['total_missed_final']} 个, 其中'后半采样能救回'的 {q3['recovered_by_2nd_half']} 个")
    print(f"   → {'系统漏抽为主(增K无救)' if q3['recovered_by_2nd_half']<0.15*max(1,q3['total_missed_final']) else '存在采样遗漏(增K有救)'}")
    print("Q4 信号 vs has_miss 相关:", {k: v for k, v in report["Q4_signal_corr_hasmiss"].items()})
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zh", "en", "both"], default="both")
    a = ap.parse_args()
    for ds in (["zh", "en"] if a.dataset == "both" else [a.dataset]):
        run(ds)
