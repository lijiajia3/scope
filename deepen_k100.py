# -*- coding: utf-8 -*-
"""deepen_k100.py — B3: 对 en 含漏抽的句子加采到 K=100, 看 systematic share 是否随 K 继续降。
只对含漏抽句深采(省成本)。读 en_7b(K=20)找含漏抽句 → 补采到 K=100 → 存 full_decodes_en_k100.jsonl。
"""
import json, os, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from full_experiment import make_client, decode_one, parse_en
from estimate_v2 import canonicalize

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
K_TARGET = 100


def hit(g, obs):
    g = g.lower(); return any((g in o.lower()) or (o.lower() in g) for o in obs)


def main():
    src = {}
    for L in open(os.path.join(OUT, "full_decodes_en_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); src[r["id"]] = r
    # 找含漏抽句
    todo = []
    for r in src.values():
        cdec = canonicalize(r["decodes"], True)
        union = sorted(set().union(*[set(d) for d in cdec])) if cdec else []
        missed = [g for g in r["gold"] if not hit(g, union)]
        if missed:
            todo.append(r)
    print(f"[deepen] {len(todo)} sentences with omissions, extending {len(src[todo[0]['id']]['decodes'])}->{K_TARGET}")

    ckpt = os.path.join(OUT, "full_decodes_en_k100.jsonl")
    done = set()
    if os.path.exists(ckpt):
        for L in open(ckpt, encoding="utf-8"):
            try: done.add(json.loads(L)["id"])
            except Exception: pass
    todo = [r for r in todo if r["id"] not in done]
    client = make_client()
    lock = threading.Lock(); t0 = time.time(); n = 0

    def work(r):
        have = r["decodes"]
        need = K_TARGET - len(have)
        extra = decode_one(client, "Qwen/Qwen2.5-7B-Instruct", "en", r["text"], need, 0.8, 0.95)
        return {**r, "decodes": have + extra}

    with open(ckpt, "a", encoding="utf-8") as fh, ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(work, r): r["id"] for r in todo}
        for fu in as_completed(futs):
            rec = fu.result()
            with lock:
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n"); fh.flush()
                n += 1
                if n % 20 == 0 or n == len(todo):
                    print(f"[deepen] {n}/{len(todo)} {(time.time()-t0)/60:.1f}min", flush=True)
    print("DEEPEN_DONE")


if __name__ == "__main__":
    main()
