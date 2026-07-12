# -*- coding: utf-8 -*-
"""
设备自适应的本地生成器。优先级：CUDA(4070) > MPS(Mac) > CPU。
- CUDA：fp16 直接上卡；7B 可选 --load_4bit(bitsandbytes, 12GB 4070 跑得动)。
- MPS ：fp16。
- CPU ：fp32。
一次前向出 K 个采样样本(num_return_sequences=K)，GPU 上远快于循环 K 次。
用法：gen_many(text, K, temperature) -> [raw_str x K]
"""
import os
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def pick_device():
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_generator(model_name, sys_prompt, load_4bit=False):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = pick_device()
    tok = AutoTokenizer.from_pretrained(model_name)

    kwargs = {}
    if dev == "cuda":
        if load_4bit:
            # 4070(12GB) 上跑 7B：4-bit NF4，计算精度 fp16。需要 bitsandbytes。
            from transformers import BitsAndBytesConfig
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16, bnb_4bit_use_double_quant=True)
            kwargs["device_map"] = "auto"
        else:
            kwargs["torch_dtype"] = torch.float16       # <=3B 直接上卡
    elif dev == "mps":
        kwargs["torch_dtype"] = torch.float16
    else:
        kwargs["torch_dtype"] = torch.float32
    print(f"[load] {model_name} on {dev} (4bit={load_4bit})", flush=True)

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    if "device_map" not in kwargs:          # 未用 device_map 时手动搬到设备
        model = model.to(dev)
    model.eval()
    mdev = next(model.parameters()).device

    def gen_many(text, K, temperature):
        msgs = [{"role": "system", "content": sys_prompt},
                {"role": "user", "content": f"句子：{text}"}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inp = tok(chat, return_tensors="pt").to(mdev)
        do_sample = temperature and temperature > 0
        with torch.no_grad():
            out = model.generate(
                **inp, max_new_tokens=128, do_sample=do_sample,
                temperature=max(temperature, 1e-2) if do_sample else None,
                top_p=0.95, num_return_sequences=K if do_sample else 1,
                pad_token_id=tok.eos_token_id)
        plen = inp["input_ids"].shape[1]
        seqs = out if out.dim() == 2 else out.unsqueeze(0)
        return [tok.decode(s[plen:], skip_special_tokens=True) for s in seqs]

    return gen_many
