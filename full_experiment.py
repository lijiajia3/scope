# -*- coding: utf-8 -*-
"""full_experiment.py — SCOPE 决定性实验(go/no-go):规模化解码 + 自覆盖估计。

数据:
  zh = OntoSpan 520 句中文医疗金标(LabelStudio 格式,本地)
  en = CoNLL-2003 test(HF 缓存 tomaarsen/conll2003),同 survey 的加载/清洗逻辑
模型: SiliconFlow API(默认 Qwen/Qwen2.5-7B-Instruct),每句采样 K 次
产出: out/full_decodes_{ds}.jsonl(逐句 checkpoint,可断点续跑)
      out/full_estimates_{ds}.json(f1/K 粗估 + Chao-Jost incidence 覆盖两套估计)

用法:
  python3.13 full_experiment.py --dataset zh --smoke      # 冒烟: n=8, K=4
  python3.13 full_experiment.py --dataset zh --n 520 --K 20
  python3.13 full_experiment.py --dataset en --n 500 --K 20
"""
import json, os, re, sys, time, argparse, random, threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import config
from estimate import estimate_one

OUT = config.OUT_DIR
GOLD_ZH = config.REAL_GOLD  # OntoSpan medical_labelstudio_gold_520.json

SYS_EN = (
    "You are a named-entity extractor. Extract every named entity in the sentence. "
    'Output only JSON: {"entities":[{"text":"entity","type":"type"}]}. '
    "Types are one of: person/organization/location/misc. Do not explain."
)

# ---------------- data loading ----------------

def load_zh(n):
    blob = json.load(open(GOLD_ZH, encoding="utf-8"))
    rows = []
    for i, item in enumerate(blob):
        text = item["data"]["text"]
        res = item.get("annotations", [{}])[0].get("result", [])
        gold = set()
        for r in res:
            t = r.get("value", {}).get("text", "")
            if isinstance(t, str) and t.strip():
                gold.add(t.strip())
        gold = sorted(gold)
        if len(gold) >= 2 and len(text) < 200:
            rows.append({"id": f"zh{i:04d}", "text": text, "gold": gold})
    random.Random(0).shuffle(rows)
    return rows[:n]


def load_en(n, minent=2):
    from datasets import load_dataset
    ds = load_dataset("tomaarsen/conll2003", split="test")
    names = ds.features["ner_tags"].feature.names
    def ents(r):
        out, cur = [], []
        for tok, ti in zip(r["tokens"], r["ner_tags"]):
            tag = names[ti]
            if tag[0] == "B":
                if cur: out.append(" ".join(cur))
                cur = [tok]
            elif tag[0] == "I" and cur:
                cur.append(tok)
            else:
                if cur: out.append(" ".join(cur)); cur = []
        if cur: out.append(" ".join(cur))
        return out
    rows = []
    for i, r in enumerate(ds):
        u = sorted(set(ents(r)))
        if len(u) >= minent:
            txt = " ".join(r["tokens"])
            txt = re.sub(r"\s+([,.;:!?%])", r"\1", txt).replace(" '", "'")
            if len(txt) < 220:
                rows.append({"id": f"en{i:04d}", "text": txt, "gold": u})
    random.Random(0).shuffle(rows)
    return rows[:n]

# ---------------- decoding ----------------

def make_client(base_url=None):
    import openai
    # base_url 指向本地 ollama (OpenAI 兼容端点) 时不需要真实 key。
    if base_url and "localhost" in base_url:
        return openai.OpenAI(api_key="ollama", base_url=base_url, timeout=180)
    # DeepInfra / Together 等第三方平台: 从对应 key 文件读取。
    if base_url and "deepinfra" in base_url:
        key = os.environ.get("DEEPINFRA_API_KEY", "") or \
            open(os.path.expanduser("~/.deepinfra_key")).read().strip()
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=120)
    if base_url and "together" in base_url:
        key = os.environ.get("TOGETHER_API_KEY", "") or \
            open(os.path.expanduser("~/.together_key")).read().strip()
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=120)
    if base_url and "groq" in base_url:
        key = os.environ.get("GROQ_API_KEY", "") or \
            open(os.path.expanduser("~/.groq_key")).read().strip()
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=120)
    if base_url and "fireworks" in base_url:
        key = os.environ.get("FIREWORKS_API_KEY", "") or \
            open(os.path.expanduser("~/.fireworks_key")).read().strip()
        return openai.OpenAI(api_key=key, base_url=base_url, timeout=120)
    key = os.environ.get("SILICONFLOW_API_KEY", "") or \
        open(os.path.expanduser("~/.siliconflow_key")).read().strip()
    return openai.OpenAI(api_key=key,
                         base_url=base_url or "https://api.siliconflow.cn/v1", timeout=120)


def parse_zh(raw):
    raw = re.sub(r"^[^:：]*[:：]", "", raw.strip()) if ("：" in raw[:12] or ":" in raw[:12]) else raw.strip()
    parts = re.split(r"[、,，;；\n]+", raw)
    return sorted({p.strip(" 。.\"'`*-") for p in parts if 0 < len(p.strip(" 。.\"'`*-")) <= 30})


def parse_en(raw):
    mo = re.search(r"\{.*\}", raw, re.S)
    if not mo:
        return []
    try:
        o = json.loads(mo.group(0))
        ents = o.get("entities")
        if not isinstance(ents, list):
            return []
        return sorted({str(e.get("text", "")).strip() for e in ents
                       if isinstance(e, dict) and str(e.get("text", "")).strip()})
    except Exception:
        return []


def decode_one(client, model, ds, text, K, temp, topp):
    if ds == "zh":
        msgs = [{"role": "system", "content": config.EXTRACT_SYS},
                {"role": "user", "content": "句子:" + text}]
        parse = parse_zh
    else:
        msgs = [{"role": "system", "content": SYS_EN},
                {"role": "user", "content": "Sentence: " + text}]
        parse = parse_en
    # en 抽取要求严格 JSON; 对非-Qwen 模型(Llama 等)用 json_object 模式保证可解析,
    # 若后端不支持则回退到普通模式(retry 循环兜底)。
    extra = {"response_format": {"type": "json_object"}} if ds == "en" else {}
    decs = []
    for _ in range(K):
        for attempt in range(5):
            try:
                r = client.chat.completions.create(
                    model=model, messages=msgs, max_tokens=300,
                    temperature=temp, top_p=topp,
                    **(extra if attempt < 3 else {}))
                decs.append(parse(r.choices[0].message.content or ""))
                break
            except Exception:
                time.sleep(3 * (attempt + 1))
        else:
            decs.append([])
    return decs

# ---------------- chao-jost incidence coverage (补充估计) ----------------

def chao_coverage(decodes, K):
    """Incidence-based sample coverage (Chao & Jost 2012).
    Q1/Q2 = 恰好出现在 1/2 个解码中的实体数, T = 总 incidence 数。
    C_hat = 1 - (Q1/T) * [ (K-1)Q1 / ((K-1)Q1 + 2Q2) ]"""
    freq = Counter()
    for d in decodes:
        for e in set(d):
            freq[e] += 1
    T = sum(freq.values())
    if T == 0:
        return None
    Q1 = sum(1 for v in freq.values() if v == 1)
    Q2 = sum(1 for v in freq.values() if v == 2)
    if Q1 == 0:
        return 1.0
    denom = (K - 1) * Q1 + 2 * Q2
    corr = ((K - 1) * Q1 / denom) if denom > 0 else 1.0
    return round(1 - (Q1 / T) * corr, 4)

# ---------------- main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["zh", "en"], required=True)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--topp", type=float, default=0.95)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tag", default="7b", help="模型标签, 决定输出文件后缀")
    ap.add_argument("--minent", type=int, default=2)
    ap.add_argument("--base_url", default=None, help="覆盖 API 端点, 如本地 ollama http://localhost:11434/v1")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        a.n, a.K = 8, 4

    rows = load_zh(a.n) if a.dataset == "zh" else load_en(a.n, a.minent)
    print(f"[data] {a.dataset} n={len(rows)} K={a.K} model={a.model}", flush=True)

    ckpt = os.path.join(OUT, f"full_decodes_{a.dataset}_{a.tag}.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for L in open(ckpt, encoding="utf-8"):
            try:
                done.add(json.loads(L)["id"])
            except Exception:
                pass
    todo = [r for r in rows if r["id"] not in done]
    print(f"[ckpt] done={len(done)} todo={len(todo)}", flush=True)

    client = make_client(a.base_url)
    lock = threading.Lock()
    t0 = time.time(); n_done = 0

    def work(r):
        decs = decode_one(client, a.model, a.dataset, r["text"], a.K, a.temp, a.topp)
        return {**r, "decodes": decs}

    with open(ckpt, "a", encoding="utf-8") as fh, \
         ThreadPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(work, r): r["id"] for r in todo}
        for fu in as_completed(futs):
            rec = fu.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                n_done += 1
                if n_done % 20 == 0 or n_done == len(todo):
                    el = time.time() - t0
                    print(f"[decode] {n_done}/{len(todo)}  {el/60:.1f}min  "
                          f"eta {el/n_done*(len(todo)-n_done)/60:.0f}min", flush=True)

    # estimate over ALL checkpointed records of this dataset
    recs = [json.loads(L) for L in open(ckpt, encoding="utf-8")]
    seen = {}
    for r in recs:
        seen[r["id"]] = r          # 去重,保最后一次
    recs = [seen[k] for k in sorted(seen)]
    out = []
    for r in recs:
        K = len(r["decodes"])
        est = estimate_one(r["decodes"], K)
        est["self_cov_chao"] = chao_coverage(r["decodes"], K)
        out.append({"id": r["id"], "text": r["text"], "gold": r["gold"], **est})
    dst = os.path.join(OUT, f"full_estimates_{a.dataset}_{a.tag}.json")
    json.dump({"model": a.model, "K": a.K, "dataset": a.dataset, "records": out},
              open(dst, "w"), ensure_ascii=False, indent=1)
    cert = sum(1 for o in out if o["status"] == "certified")
    print(f"[estimate] {len(out)} sents, certified {cert}, abstain {len(out)-cert}")
    print(f"[done] -> {dst}", flush=True)


if __name__ == "__main__":
    main()
