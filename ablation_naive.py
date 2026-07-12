# -*- coding: utf-8 -*-
"""ablation_naive.py — A1 朴素弃权基线消融(决定贡献2措辞)。
对比三条弃权规则在双语料上的 certified/abstained recall 分离:
  naive1: union 实体数 <=1 即弃权
  naive2: union 实体数 <=2 即弃权
  scope : SCOPE 完整规则(K<5 / singleton_frac>=0.6 / cov_gt<0.5)
核心问题: SCOPE 相对朴素计数规则的非平凡增量有多大?
读 out/full_decodes_{ds}_7b.jsonl。产出 out/ablation_naive.json + 终端表。
"""
import json, os, math
from estimate_v2 import estimate_one_v2, canonicalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs, en):
    if en:
        g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)
    return any((g in o) or (o in g) for o in obs)


def boot_ci(pairs, stat, n_boot=1999, seed=0):
    """pairs: list, stat(sub)->float; 句子层重抽 95% CI。"""
    import random
    rng = random.Random(seed)
    vals = []
    m = len(pairs)
    for _ in range(n_boot):
        sub = [pairs[rng.randrange(m)] for _ in range(m)]
        v = stat(sub)
        if v is not None:
            vals.append(v)
    vals.sort()
    lo = vals[int(0.025 * len(vals))]
    hi = vals[int(0.975 * len(vals))]
    return round(lo, 4), round(hi, 4)


def gap_stat(recs_cert_abst):
    cert = [tr for flag, tr in recs_cert_abst if flag == "cert"]
    abst = [tr for flag, tr in recs_cert_abst if flag == "abst"]
    if not cert or not abst:
        return None
    return (sum(cert) / len(cert)) - (sum(abst) / len(abst))


def run(ds):
    en = (ds == "en")
    recs = {}
    for L in open(os.path.join(OUT, f"full_decodes_{ds}_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    rows = []  # (tr, union_size, scope_status)
    for r in recs.values():
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold:
            continue
        tr = sum(1 for g in gold if hit(g, union, en)) / len(gold)
        est = estimate_one_v2(decs, K, en)
        rows.append((tr, len(union), est["status"]))

    rules = {
        "naive_u1": lambda tr, u, s: "abst" if u <= 1 else "cert",
        "naive_u2": lambda tr, u, s: "abst" if u <= 2 else "cert",
        "scope":    lambda tr, u, s: "cert" if s == "certified" else "abst",
    }
    out = {"dataset": ds, "n": len(rows)}
    for name, rule in rules.items():
        labeled = [(rule(tr, u, s), tr) for tr, u, s in rows]
        cert = [tr for f, tr in labeled if f == "cert"]
        abst = [tr for f, tr in labeled if f == "abst"]
        gap = gap_stat(labeled)
        ci = boot_ci(labeled, gap_stat)
        out[name] = {
            "n_cert": len(cert), "n_abst": len(abst),
            "cert_recall": round(sum(cert) / len(cert), 4) if cert else None,
            "abst_recall": round(sum(abst) / len(abst), 4) if abst else None,
            "gap": round(gap, 4) if gap is not None else None,
            "gap_ci95": ci,
        }
    return out


def ranking_curve(ds):
    """相同保留率下的公平对比: cov_gt 排序 vs union_size 排序 vs random。
    对每个保留率 p, 取该信号 top-p% 最可信句子作 certified, 报 cert_recall。
    这消除了固定规则弃权数量不同的混淆, 是 GT 相对朴素计数的真实增量度量。"""
    import random
    en = (ds == "en")
    recs = {}
    for L in open(os.path.join(OUT, f"full_decodes_{ds}_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    items = []  # (cov_gt, union_size, singleton_frac, true_recall)
    for r in recs.values():
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, en)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        gold = r["gold"]
        if not gold:
            continue
        tr = sum(1 for g in gold if hit(g, union, en)) / len(gold)
        est = estimate_one_v2(decs, K, en)
        items.append((est.get("cov_gt", 0.0), len(union), est.get("singleton_frac", 1.0), tr))
    n = len(items)
    curves = {}
    signals = {
        "cov_gt":   lambda it: it[0],           # SCOPE 连续自覆盖 (高=可信)
        "union_sz": lambda it: it[1],           # 朴素计数 (多=可信)
        "random":   None,
    }
    for name, key in signals.items():
        row = {}
        for p in (0.5, 0.6, 0.7, 0.8, 0.9):
            k = max(1, int(n * p))
            if name == "random":
                accs = []
                for s in range(20):
                    o = items[:]; random.Random(s).shuffle(o)
                    sub = o[:k]; accs.append(sum(x[3] for x in sub) / len(sub))
                row[f"keep{int(p*100)}"] = round(sum(accs)/len(accs), 4)
            else:
                o = sorted(items, key=key, reverse=True)[:k]
                row[f"keep{int(p*100)}"] = round(sum(x[3] for x in o) / len(o), 4)
        curves[name] = row
    return curves


def main():
    results = {}
    print(f"{'ds/rule':<18}{'n_cert':>7}{'n_abst':>7}{'cert_r':>8}{'abst_r':>8}{'gap':>8}{'gap_CI95':>18}")
    for ds in ("zh", "en"):
        r = run(ds)
        results[ds] = r
        for rule in ("naive_u1", "naive_u2", "scope"):
            d = r[rule]
            ci = d["gap_ci95"]
            print(f"{ds+'/'+rule:<18}{d['n_cert']:>7}{d['n_abst']:>7}"
                  f"{str(d['cert_recall']):>8}{str(d['abst_recall']):>8}{str(d['gap']):>8}"
                  f"  [{ci[0]:+.3f},{ci[1]:+.3f}]")
        # SCOPE 相对 naive 的非平凡增量
        scope_gap = r["scope"]["gap"]; n1 = r["naive_u1"]["gap"]
        print(f"  → {ds}: SCOPE gap={scope_gap:+.4f} vs naive(u<=1) gap={n1:+.4f}  "
              f"非平凡增量={scope_gap-n1:+.4f}")
    print("\n=== 公平对比: 相同保留率下 certified recall (排序信号) ===")
    for ds in ("zh", "en"):
        cur = ranking_curve(ds)
        results[ds]["ranking_curve"] = cur
        print(f"--- {ds} ---   " + "  ".join(f"keep{p}" for p in (50,60,70,80,90)))
        for name in ("cov_gt", "union_sz", "random"):
            vals = "  ".join(f"{cur[name][f'keep{p}']:.3f}" for p in (50,60,70,80,90))
            print(f"  {name:<10} {vals}")
        # cov_gt 相对 union_sz 的平均增量
        adv = sum(cur["cov_gt"][f"keep{p}"]-cur["union_sz"][f"keep{p}"] for p in (50,60,70,80,90))/5
        print(f"  → cov_gt vs union_sz 平均 certified-recall 增量: {adv:+.4f}")
    json.dump(results, open(os.path.join(OUT, "ablation_naive.json"), "w"), ensure_ascii=False, indent=1)
    print("[done] -> out/ablation_naive.json")


if __name__ == "__main__":
    main()
