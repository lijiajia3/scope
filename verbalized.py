# -*- coding: utf-8 -*-
"""verbalized.py — B6: verbalized-confidence 第三分诊基线。
对 en 每句问模型"你是否抽全了实体, 完整性信心 0-100", 用 (100-conf)/100 作风险信号。
复用现有 en_7b 的句子集合。产出 out/verbalized_en.json: {id: risk}。
"""
import json, os, re, time, threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from full_experiment import make_client

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
SYS = ("You extracted named entities (person/organization/location/misc) from a sentence. "
       "State your confidence that you found ALL entities, i.e. that none were missed. "
       "Reply with ONLY an integer 0-100 (100 = certain nothing was missed).")


def ask(client, text, entities):
    msg = [{"role": "system", "content": SYS},
           {"role": "user", "content": f"Sentence: {text}\nEntities you extracted: {entities}\nCompleteness confidence (0-100):"}]
    for a in range(4):
        try:
            r = client.chat.completions.create(model="Qwen/Qwen2.5-7B-Instruct",
                messages=msg, max_tokens=8, temperature=0.0)
            m = re.search(r"\d{1,3}", r.choices[0].message.content or "")
            if m:
                return min(100, int(m.group()))
        except Exception:
            time.sleep(3 * (a + 1))
    return None


def main():
    recs = {}
    for L in open(os.path.join(OUT, "full_decodes_en_7b.jsonl"), encoding="utf-8"):
        r = json.loads(L); recs[r["id"]] = r
    ckpt = os.path.join(OUT, "verbalized_en.json")
    done = json.load(open(ckpt)) if os.path.exists(ckpt) else {}
    todo = [r for r in recs.values() if r["id"] not in done]
    client = make_client(); lock = threading.Lock(); n = 0; t0 = time.time()

    def work(r):
        # 用第一个解码的实体作为"已抽取"展示
        ents = ", ".join(sorted(set(r["decodes"][0]))) if r["decodes"] and r["decodes"][0] else "(none)"
        conf = ask(client, r["text"], ents)
        return r["id"], conf

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(work, r) for r in todo]
        for fu in as_completed(futs):
            rid, conf = fu.result()
            with lock:
                done[rid] = (100 - conf) / 100 if conf is not None else None
                n += 1
                if n % 50 == 0 or n == len(todo):
                    json.dump(done, open(ckpt, "w"))
                    print(f"[verbalized] {n}/{len(todo)} {(time.time()-t0)/60:.1f}min", flush=True)
    json.dump(done, open(ckpt, "w"))
    print("VERBALIZED_DONE")


if __name__ == "__main__":
    main()
