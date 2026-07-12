# -*- coding: utf-8 -*-
"""eval_triage_baselines.py — B6: 分诊 AUROC 加入 verbalized-confidence 第三基线。
四信号对比: SCOPE coverage-risk / semantic-entropy / 1-self-consistency / 1-verbalized-conf。
读 en_7b decodes + verbalized_en.json。产出 out/triage_baselines.json + bootstrap CI。
"""
import json, os, math, random
from collections import Counter
from estimate_v2 import estimate_one_v2, canonicalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")


def hit(g, obs, en=True):
    g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)


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


def main():
    vpath = os.path.join(OUT, "verbalized_en.json")
    if not os.path.exists(vpath):
        print("[skip] verbalized data not ready"); return
    verb = json.load(open(vpath))
    recs = {}
    for L in open(os.path.join(OUT, "full_decodes_en_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    rows = []
    for rid, r in recs.items():
        if rid not in verb or verb[rid] is None: continue
        decs = r["decodes"]; K = len(decs)
        cdec = canonicalize(decs, True)
        sets = [set(d) for d in cdec]
        union = sorted(set().union(*sets)) if sets else []
        est = estimate_one_v2(decs, K, True)
        setstr = [json.dumps(sorted(s), ensure_ascii=False) for s in sets]
        c = Counter(setstr); tot = sum(c.values())
        ent = -sum((v/tot)*math.log(v/tot) for v in c.values())/math.log(max(2, tot))
        js = []
        for i in range(len(sets)):
            for j in range(i+1, len(sets)):
                u = sets[i] | sets[j]
                js.append(len(sets[i] & sets[j])/len(u) if u else 1.0)
        selfc = sum(js)/len(js) if js else 1.0
        gold = r["gold"]
        missed = [g for g in gold if not hit(g, union)]
        rows.append({"has_miss": int(len(missed) > 0),
                     "cov_risk": 1-est.get("cov_gt", 0.0),
                     "sem_ent": ent, "inv_selfc": 1-selfc,
                     "inv_verb": verb[rid]})
    labels = [r["has_miss"] for r in rows]
    sigs = {"SCOPE_cov_risk": "cov_risk", "semantic_entropy": "sem_ent",
            "inv_self_consistency": "inv_selfc", "inv_verbalized_conf": "inv_verb"}
    out = {"n": len(rows)}
    rng = random.Random(0)
    print(f"n={len(rows)} (含 verbalized 分数的句子)")
    for name, key in sigs.items():
        a = auroc(labels, [r[key] for r in rows])
        boot = []
        for _ in range(1999):
            idx = [rng.randrange(len(rows)) for _ in range(len(rows))]
            sub = [rows[i] for i in idx]
            av = auroc([r["has_miss"] for r in sub], [r[key] for r in sub])
            if av is not None: boot.append(av)
        boot.sort()
        ci = [round(boot[int(0.025*len(boot))], 3), round(boot[int(0.975*len(boot))], 3)]
        out[name] = {"auroc": round(a, 4), "ci95": ci}
        print(f"  {name:<24} AUROC={a:.3f}  95%CI[{ci[0]},{ci[1]}]")
    json.dump(out, open(os.path.join(OUT, "triage_baselines.json"), "w"), ensure_ascii=False, indent=1)
    print("[done] -> out/triage_baselines.json")


if __name__ == "__main__":
    main()
