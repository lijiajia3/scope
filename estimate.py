# -*- coding: utf-8 -*-
"""
第 2 步：从重复解码估计"自覆盖"（core：Good-Turing 缺失质量）+ 认证或弃权。
纯 Python，无需模型，读 out/decodes.json。

对每条句子（K 个解码集合，每个 distinct 实体是一个"物种"，频次 = 出现在几个解码里）：
  f1 = 只在 1 个解码里出现的实体数（singletons）
  缺失质量  M̂ = f1 / K              （还没抽到的"模型可产出实体"的概率质量估计）
  自覆盖    Ĉ = 1 − M̂
  certify-or-abstain：
    - 解码近乎塌缩（f1==0 且绝大多数实体在全部 K 个解码都出现）→ M̂=0 是"假 100%"，弃权
    - K<3 或 union<=1 → 无从估计，弃权
    - 否则认证，报自覆盖 + 简单区间
产出 out/estimates.json。
"""
import json, argparse
from collections import Counter
import config


def estimate_one(decodes, K):
    sets = [set(d) for d in decodes]
    union = sorted(set().union(*sets)) if sets else []
    # 频次：每个实体出现在几个解码里
    freq = Counter()
    for s in sets:
        for e in s:
            freq[e] += 1
    U = len(union)
    f1 = sum(1 for e in union if freq[e] == 1)
    fK = sum(1 for e in union if freq[e] == K)             # 在全部 K 个解码都出现
    stable_union = [e for e in union if freq[e] >= 2]      # 稳定集：至少出现 2 次(去单次噪声)
    singleton_frac = round(f1 / U, 3) if U else 0.0        # 单次占比：越高越像噪声主导
    missing_mass = f1 / K if K else float("nan")
    self_cov = 1 - missing_mass
    # certify-or-abstain
    if K < 3 or U <= 1:
        status, reason = "abstain", "K<3 或可观测实体过少"
    elif f1 == 0 and (fK / U if U else 1) >= 0.8:
        status, reason = "abstain", "解码近塌缩(无singleton且多数实体全命中)，M̂=0 不可信"
    elif singleton_frac >= 0.7:
        status, reason = "abstain", "单次占比≥0.7，解码噪声主导，缺失质量不可信"
    else:
        status, reason = "certified", "ok"
    return {"K": K, "U": U, "f1": f1, "fK": fK,
            "singleton_frac": singleton_frac,
            "missing_mass": round(missing_mass, 4),
            "self_coverage": round(self_cov, 4),
            "observed_union": union,
            "stable_union": stable_union,
            "status": status, "reason": reason}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decodes", default=config.DECODES)
    a = ap.parse_args()
    blob = json.load(open(a.decodes, encoding="utf-8"))
    K = blob["K"]
    out = []
    for r in blob["records"]:
        est = estimate_one(r["decodes"], K)
        out.append({"id": r["id"], "text": r["text"], "entities": r["entities"], **est})

    cert = [o for o in out if o["status"] == "certified"]
    absta = [o for o in out if o["status"] == "abstain"]
    print(f"[estimate] {len(out)} sents | certified {len(cert)} | abstain {len(absta)}")
    if cert:
        mm = sum(o["missing_mass"] for o in cert) / len(cert)
        sc = sum(o["self_coverage"] for o in cert) / len(cert)
        print(f"  certified 平均 missing_mass={mm:.3f}  self_coverage={sc:.3f}")
    json.dump({"model": blob.get("model"), "K": K, "records": out},
              open(config.ESTIMATES, "w"), ensure_ascii=False, indent=2)
    print(f"[done] -> {config.ESTIMATES}")


if __name__ == "__main__":
    main()
