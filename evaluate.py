# -*- coding: utf-8 -*-
"""
第 3 步：用 gold 验证 + 主动标注分诊（论文卖点）。纯 Python，读 out/estimates.json。

产出三样：
  A. 自覆盖 vs 真召回：证书证的是"自覆盖"，这里量它比真召回高多少（= 系统性漏抽质量，诚实边界）。
  B. 弃权是否合理：abstain 的句子真召回是否确实更差/更不可判。
  C. 主动标注分诊：按 label-free 的 missing_mass 排序，复核前 X% 高风险句能捞回多少真实漏抽
     （对比随机顺序）——这是"省标注"的实用性证据。
"""
import json, argparse
import config


def match(gold, observed):
    """宽松包含匹配：gold 实体 g 被 observed 里某项包含或包含即算命中。"""
    hit = 0
    for g in gold:
        if any((g in o) or (o in g) for o in observed):
            hit += 1
    return hit


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--estimates", default=config.ESTIMATES)
    a = ap.parse_args()
    blob = json.load(open(a.estimates, encoding="utf-8"))
    recs = blob["records"]

    for r in recs:
        g = r["entities"]; obs = r["observed_union"]
        hit = match(g, obs)
        r["true_recall"] = round(hit / len(g), 4) if g else 1.0
        r["missed"] = len(g) - hit
        # 稳定集(去单次噪声)下的召回：诊断噪声对 union 召回的贡献
        st = r.get("stable_union", obs)
        r["stable_recall"] = round(match(g, st) / len(g), 4) if g else 1.0

    cert = [r for r in recs if r["status"] == "certified"]
    absta = [r for r in recs if r["status"] == "abstain"]

    # A. 自覆盖 vs 真召回（certified 上）
    A = {}
    if cert:
        A["mean_self_coverage"] = round(sum(r["self_coverage"] for r in cert)/len(cert), 3)
        A["mean_true_recall"] = round(sum(r["true_recall"] for r in cert)/len(cert), 3)
        A["gap_selfcov_minus_truerecall"] = round(A["mean_self_coverage"] - A["mean_true_recall"], 3)
        A["mean_stable_recall"] = round(sum(r["stable_recall"] for r in cert)/len(cert), 3)
        A["mean_singleton_frac"] = round(sum(r.get("singleton_frac", 0) for r in cert)/len(cert), 3)

    # B. 弃权合理性
    B = {"n_abstain": len(absta), "n_certified": len(cert)}
    if absta:
        B["abstain_mean_true_recall"] = round(sum(r["true_recall"] for r in absta)/len(absta), 3)
    if cert:
        B["certified_mean_true_recall"] = round(sum(r["true_recall"] for r in cert)/len(cert), 3)

    # C. 主动标注分诊：按 missing_mass 降序 vs 随机，看捞回真实漏抽的比例
    total_missed = sum(r["missed"] for r in recs)
    C = {"total_missed_entities": total_missed}
    if total_missed > 0:
        by_risk = sorted(recs, key=lambda r: r["missing_mass"], reverse=True)
        n = len(recs)
        for frac in (0.2, 0.3, 0.5):
            k = max(1, int(round(frac * n)))
            caught = sum(r["missed"] for r in by_risk[:k])
            C[f"review_top_{int(frac*100)}pct"] = {
                "sents_reviewed": k,
                "missed_caught": caught,
                "recall_of_missed": round(caught / total_missed, 3),
                "random_baseline": round(frac, 3),
            }

    summary = {"model": blob.get("model"), "K": blob.get("K"), "n": len(recs),
               "A_selfcov_vs_truerecall": A, "B_abstain_sanity": B, "C_triage": C}
    json.dump({"summary": summary, "records": recs},
              open(config.RESULTS, "w"), ensure_ascii=False, indent=2)

    print("=== SCOPE 结果 ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("\n解读：")
    if A:
        g = A["gap_selfcov_minus_truerecall"]
        if g > 0.1:
            print(f"  A 自覆盖 {A['mean_self_coverage']} > 真召回 {A['mean_true_recall']}（差 +{g}）"
                  f"：证书乐观——系统性漏抽 label-free 测不到（危险方向）。")
        elif g < -0.1:
            print(f"  A ⚠️ 自覆盖 {A['mean_self_coverage']} << 真召回 {A['mean_true_recall']}（差 {g}）"
                  f"：证书悲观且失真——多为解码噪声制造的假 singleton 抬高了缺失质量，"
                  f"缺失质量此时不可信（本模型下方法未标定）。")
        else:
            print(f"  A 自覆盖 {A['mean_self_coverage']} ≈ 真召回 {A['mean_true_recall']}：标定良好。")
    if total_missed > 0 and "review_top_30pct" in C:
        c30 = C["review_top_30pct"]
        verdict = ("有效" if c30["recall_of_missed"] > c30["random_baseline"] + 0.05
                   else "无效（不优于随机，主卖点在本数据上不成立）")
        print(f"  C 主动标注：复核前 30% 高风险句捞回 {c30['recall_of_missed']:.0%} 真实漏抽"
              f"（随机 {c30['random_baseline']:.0%}）→ {verdict}。")
    print(f"\n[done] -> {config.RESULTS}")


if __name__ == "__main__":
    main()
