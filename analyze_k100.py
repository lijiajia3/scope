# -*- coding: utf-8 -*-
"""analyze_k100.py — B3: K=20 vs K=100 systematic share 对比。
含漏抽句加采到 K=100 后, 看"系统性遗漏"份额是否随 K 下降。
下降=部分"系统性"实为采样不足; 不降=真能力缺口, 支撑现有结论。
读 full_decodes_en_k100.jsonl(含漏抽句K=100) + en_7b(K=20)。
"""
import json, os
from estimate_v2 import canonicalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs):
    g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)


def sys_share(recs, Kcap):
    """对给定句子集, 用前 Kcap 次解码算 systematic share。"""
    tot_miss = tot_samp = 0
    for r in recs:
        decs = r["decodes"][:Kcap]
        cdec = canonicalize(decs, True)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        half = cdec[:max(1, Kcap//2)]
        uh = sorted(set().union(*[set(d) for d in half])) if half else []
        gold = r["gold"]
        missed = [g for g in gold if not hit(g, union)]
        samp = sum(1 for g in gold if (not hit(g, uh)) and hit(g, union))
        tot_miss += len(missed); tot_samp += samp
    # systematic = 全 Kcap 并集也没有 (tot_miss); 采样遗漏 = 半预算漏但被救回 (tot_samp);
    # 二者之并 = missed@(Kcap/2); frac_systematic = systematic / that union。
    return tot_miss / max(1, tot_miss + tot_samp), tot_miss


def main():
    p = os.path.join(OUT, "full_decodes_en_k100.jsonl")
    if not os.path.exists(p):
        print("[skip] k100 data not ready"); return
    recs = {}
    for L in open(p, encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    recs = [r for r in recs.values() if len(r["decodes"]) >= 100]
    print(f"[k100] {len(recs)} sentences with K>=100")
    print(f"{'K':>5}{'systematic_share':>18}{'total_missed':>14}{'recovered_vs_K20':>18}")
    base_miss = None
    for Kcap in (20, 40, 60, 80, 100):
        share, miss = sys_share(recs, Kcap)
        if base_miss is None: base_miss = miss
        rec = base_miss - miss
        print(f"{Kcap:>5}{share:>18.3f}{miss:>14}{rec:>18}")
    # 结论
    s20, m20 = sys_share(recs, 20)
    s100, m100 = sys_share(recs, 100)
    print(f"\n漏抽实体数: K=20 {m20} → K=100 {m100} (增大5倍采样多救回 {m20-m100} 个)")
    print(f"→ {'系统性为主(增K救回<10%, 真能力缺口)' if (m20-m100)<0.1*m20 else '存在采样不足(增K可观救回)'}")
    json.dump({"note": "K=20 vs K=100 on omission sentences",
               "missed_K20": m20, "missed_K100": m100,
               "recovered": m20-m100}, open(os.path.join(OUT, "k100_result.json"), "w"))
    print("[done] -> out/k100_result.json")


if __name__ == "__main__":
    main()
