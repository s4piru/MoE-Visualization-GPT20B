import os
import json
import argparse
import numpy as np
import pandas as pd
from .core import load_model, run_pairwise_trace
from .metrics import layerwise_jsd_from_topk, summarize_expert_shifts
from .viz import plot_jsd_heatmap, plot_layer_mean_jsd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_json")
    ap.add_argument("--out", default="moe_viz_out")
    args = ap.parse_args()

    with open(args.input_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model_id = cfg.get("model_id", "openai/gpt-oss-20b")
    device_map = cfg.get("device_map", "auto")
    tok, model = load_model(model_id=model_id, device_map=device_map)

    prompt_base = cfg["prompt_base"]
    prompt_var = cfg["prompt_var"]
    question = cfg["question"]
    language = cfg.get("language", None)
    max_new_tokens = int(cfg.get("max_new_tokens", 128))
    temperature = float(cfg.get("temperature", 0.2))
    topk = int(cfg.get("topk", 4))

    A, B, gen_steps, startA, startB = run_pairwise_trace(
        tok, model, prompt_base, prompt_var, question,
        language=language, max_new_tokens=max_new_tokens, temperature=temperature, topk=topk
    )

    jsd_map = layerwise_jsd_from_topk(A["moe"], B["moe"], gen_steps, startA, startB)
    df_shift = summarize_expert_shifts(A["moe"], B["moe"], gen_steps, startA, startB, topn=20)

    os.makedirs(args.out, exist_ok=True)
    plot_jsd_heatmap(jsd_map, os.path.join(args.out, "jsd_heatmap.png"), title="JSD(base vs var)")
    plot_layer_mean_jsd(jsd_map, os.path.join(args.out, "jsd_layer_mean.png"), title="Layer mean JSD")

    with open(os.path.join(args.out, "meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "input_len_base": A["input_len"],
            "gen_len_base": A["gen_len"],
            "input_len_var": B["input_len"],
            "gen_len_var": B["gen_len"],
            "gen_steps_aligned": gen_steps
        }, f, ensure_ascii=False, indent=2)

    if not df_shift.empty:
        df_shift.to_csv(os.path.join(args.out, "expert_shifts_top20.csv"), index=False, encoding="utf-8")
    with open(os.path.join(args.out, "layers.json"), "w", encoding="utf-8") as f:
        json.dump(sorted(list(set(A["moe"].keys()).intersection(set(B["moe"].keys())))), f, ensure_ascii=False, indent=2)
