# -*- coding: utf-8 -*-
"""estimate_v2.py — SCOPE v2 自覆盖估计。修 v1 的三处硬伤:
  (1) 弃用 f1/K(非有效界; 噪声灌入), 用标准 Good-Turing coverage 1 - f1/N 与 Chao-Jost;
  (2) 跨 K 个解码做实体变体归并(canonicalize), 消除"同一实体不同写法"虚增的 singleton;
  (3) certify-or-abstain 阈值基于归并后的稳定结构。
读 out/full_decodes_{ds}.jsonl → 写 out/est_v2_{ds}.json。纯 Python。
"""
import json, os, re, argparse
from collections import Counter, defaultdict
import config

OUT = config.OUT_DIR


def norm_en(e):
    e = e.lower().strip()
    e = re.sub(r"^(the|a|an)\s+", "", e)
    e = re.sub(r"[\s\-]+", " ", e).strip(" .,'\"")
    return e


def norm_zh(e):
    return e.strip(" 。.、，,\"'`*-（）()")


def canonicalize(decodes, en):
    """把所有解码里的实体做变体归并: 互为子串/规范化后相同的 → 归到最长的 canonical 形式。
    返回 canonical 解码集合列表 + canonical→原始变体映射。"""
    norm = norm_en if en else norm_zh
    # 收集所有出现过的规范化实体及其原始最长代表
    reps = {}                       # normed -> canonical surface (最长)
    for d in decodes:
        for e in d:
            ne = norm(e)
            if not ne:
                continue
            if ne not in reps or len(e) > len(reps[ne]):
                reps[ne] = e
    keys = list(reps)
    # 子串归并: 若 a 是 b 的子串(规范化后), 合并到更长的 b
    parent = {k: k for k in keys}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    keys_by_len = sorted(keys, key=len)
    for i, a in enumerate(keys_by_len):
        for b in keys_by_len[i+1:]:
            if a and a in b and (len(a) >= 2 or not en):   # 英文避免单字符误并
                parent[find(a)] = find(b); break
    canon_of = {}
    for k in keys:
        root = find(k)
        canon_of[k] = reps[root]
    # 重写每个解码为 canonical 集合
    cdecodes = []
    for d in decodes:
        s = set()
        for e in d:
            ne = norm(e)
            if ne in canon_of:
                s.add(canon_of[ne])
        cdecodes.append(sorted(s))
    return cdecodes


def estimate_one_v2(decodes, K, en, do_canon=True):
    cdec = canonicalize(decodes, en) if do_canon else [sorted(set(map(norm_en if en else norm_zh, d))) for d in decodes]
    freq = Counter()
    for d in cdec:
        for e in set(d):
            freq[e] += 1
    U = len(freq)
    if U == 0:
        return {"status": "abstain", "reason": "空", "K": K, "U": 0}
    f1 = sum(1 for v in freq.values() if v == 1)
    f2 = sum(1 for v in freq.values() if v == 2)
    N = sum(freq.values())
    # Good-Turing coverage
    cov_gt = 1 - f1 / N
    # Chao-Jost incidence coverage
    if f1 == 0:
        cov_chao = 1.0
    else:
        denom = (K - 1) * f1 + 2 * f2
        cov_chao = 1 - (f1 / N) * ((K - 1) * f1 / denom if denom else 1.0)
    chao1 = U + (f1 * f1 / (2 * f2) if f2 else f1 * (f1 - 1) / 2)
    singleton_frac = f1 / U
    stable = [e for e in freq if freq[e] >= max(2, K * 0.2)]   # 出现在≥20%解码
    # certify-or-abstain: 归并后仍高度不稳定 → 弃权
    if K < 5 or U < 1:
        status, reason = "abstain", "K/U 过小"
    elif singleton_frac >= 0.6:
        status, reason = "abstain", "归并后单次占比≥0.6, 采样不稳"
    elif cov_gt < 0.5:
        status, reason = "abstain", "自覆盖<0.5"
    else:
        status, reason = "certified", "ok"
    return {"K": K, "U": U, "f1": f1, "f2": f2, "N": N,
            "cov_gt": round(cov_gt, 4), "cov_chao": round(cov_chao, 4),
            "chao1_richness": round(chao1, 2),
            "singleton_frac": round(singleton_frac, 3),
            "canon_union": sorted(freq), "stable_union": stable,
            "status": status, "reason": reason}


def run(ds):
    en = (ds == "en")
    recs = {}
    for L in open(os.path.join(OUT, f"full_decodes_{ds}.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    out = []
    for rid in sorted(recs):
        r = recs[rid]; K = len(r["decodes"])
        est = estimate_one_v2(r["decodes"], K, en)
        out.append({"id": rid, "text": r["text"], "gold": r["gold"],
                    "decodes": r["decodes"], **est})
    cert = sum(1 for o in out if o["status"] == "certified")
    dst = os.path.join(OUT, f"est_v2_{ds}.json")
    json.dump({"dataset": ds, "K": out[0]["K"] if out else None, "records": out},
              open(dst, "w"), ensure_ascii=False, indent=1)
    print(f"[v2 {ds}] {len(out)} sents | certified {cert} | abstain {len(out)-cert} -> {dst}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zh", "en", "both"], default="both")
    a = ap.parse_args()
    for ds in (["zh", "en"] if a.dataset == "both" else [a.dataset]):
        run(ds)
