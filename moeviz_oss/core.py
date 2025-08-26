import os
import json
import math
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

def load_model(model_id="openai/gpt-oss-20b", device_map="auto"):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="auto", device_map=device_map).eval()
    return tok, model

def _apply_chat(tok, system_prompt, user_prompt):
    if hasattr(tok, "apply_chat_template") and tok.chat_template:
        messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    else:
        text = system_prompt + "\n\nUser: " + user_prompt + "\nAssistant:"
    return text

def _prepare_inputs(tok, system_prompt, question, language=None):
    q = question if language is None else (question if language.lower() == "en" else f"{question}（回答は{language}）")
    text = _apply_chat(tok, system_prompt, q)
    enc = tok(text, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], text

def _generate(tok, model, input_ids, attention_mask, max_new_tokens=128, temperature=0.2):
    gen = model.generate(
        input_ids=input_ids.to(model.device),
        attention_mask=attention_mask.to(model.device),
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        temperature=temperature,
        use_cache=True
    )
    text = tok.decode(gen[0], skip_special_tokens=True)
    gen_len = gen.shape[1] - input_ids.shape[1]
    return gen, text, gen_len

def _find_moe_modules(model):
    kws = ("moe", "router", "gate", "switch")
    mods = []
    for name, module in model.named_modules():
        low = name.lower()
        if any(k in low for k in kws):
            mods.append((name, module))
    return mods

def _collect_moe_topk(model, tok_ids, attn, topk=4):
    records = {}
    hooks = []
    names = []

    def make_hook(name):
        def hook(module, inputs, output):
            with torch.no_grad():
                o = output
                if isinstance(o, tuple):
                    o = o[0]
                if not torch.is_tensor(o):
                    return
                x = o
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                if x.dim() == 3:
                    b, s, h = x.shape
                    probs = torch.softmax(x.float(), dim=-1)
                    p, i = torch.topk(probs, k=min(topk, h), dim=-1)
                    p = p.detach().cpu().numpy()
                    i = i.detach().cpu().numpy()
                    if name not in records:
                        records[name] = {"topk_probs": p, "topk_indices": i}
                    else:
                        oldp = records[name]["topk_probs"]
                        oldi = records[name]["topk_indices"]
                        records[name]["topk_probs"] = np.concatenate([oldp, p], axis=1)
                        records[name]["topk_indices"] = np.concatenate([oldi, i], axis=1)
        return hook

    for name, module in _find_moe_modules(model):
        names.append(name)
        hooks.append(module.register_forward_hook(make_hook(name)))

    with torch.no_grad():
        _ = model(input_ids=tok_ids.to(model.device), attention_mask=attn.to(model.device), use_cache=False)

    for h in hooks:
        h.remove()

    for k in list(records.keys()):
        v = records[k]
        if isinstance(v["topk_probs"], list):
            v["topk_probs"] = np.array(v["topk_probs"])
            v["topk_indices"] = np.array(v["topk_indices"])
    return records

def run_single_trace(tok, model, system_prompt, question, language=None, max_new_tokens=128, temperature=0.2, topk=4):
    inp_ids, attn, text_in = _prepare_inputs(tok, system_prompt, question, language)
    gen, text_out, gen_len = _generate(tok, model, inp_ids, attn, max_new_tokens=max_new_tokens, temperature=temperature)
    full_ids = gen
    full_attn = torch.ones_like(full_ids)
    moe = _collect_moe_topk(model, full_ids, full_attn, topk=topk)
    in_len = inp_ids.shape[1]
    return {"input_len": int(in_len), "gen_len": int(gen_len), "text_in": text_in, "text_out": text_out, "moe": moe}

def _align_min_gen(a, b):
    g = min(a["gen_len"], b["gen_len"])
    sa = a["input_len"]
    sb = b["input_len"]
    return g, sa, sb

def run_pairwise_trace(tok, model, prompt_base, prompt_var, question, language=None, max_new_tokens=128, temperature=0.2, topk=4):
    A = run_single_trace(tok, model, prompt_base, question, language, max_new_tokens, temperature, topk)
    B = run_single_trace(tok, model, prompt_var, question, language, max_new_tokens, temperature, topk)
    g, sa, sb = _align_min_gen(A, B)
    return A, B, g, sa, sb
