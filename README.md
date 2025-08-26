MoE (Mixture-of-Experts) models contain many parallel experts and a lightweight router that, for each token, selects only a few experts (Top-K) to run. 

This tool visualizes how those routing choices change when you switch system prompts

You will give a base prompt, a variant prompt, and one question; the tool runs the model twice, records per-step Top-K expert probabilities in MoE/router layers, computes divergence (JSD) over time, and outputs heatmaps plus per-layer rankings of experts with the largest usage shifts. 

The result is useful for red-teaming, evaluation, and debugging

## Install

```bash
pip install .
```

## Run

```bash
moeviz-oss input.json --out out_dir
```

### input.json example

```
{
  "prompt_base": "You are a 50-year-old man living in New York who has no interest in anime and loves watching basketball.",
  "prompt_var":  "You are a 17-year-old high school student living in Tokyo who is an anime and manga expert.",
  "question":    "What is the name of the devil fruit that Luffy ate in One Piece?",
  "language":    "en",
  "max_new_tokens": 128,
  "temperature": 0.2,
  "topk": 4,
  "model_id": "openai/gpt-oss-20b",
  "device_map": "auto"
  "topn_per_layer": 10
}
```

### Outputs under out_dir/:

- `jsd_heatmap.png` — Heatmap of Jensen–Shannon divergence across generation steps (x-axis) and MoE/router layers (y-axis), comparing routing distributions between the **base** and **var** prompts. Higher values indicate stronger distributional shifts.

- `jsd_layer_mean.png` — Bar chart of per-layer mean JSD over the aligned generation window. Tall bars highlight layers whose routing is most sensitive to the prompt change.

- `expert_shifts_per_layer_topN.csv` — Per-layer ranking of experts by the magnitude of change, sorted by **abs(delta)** and taking the top **N** experts within each layer. Columns:
  - `layer` — Layer name.
  - `expert` — Expert ID within the layer (shared between base and var for the same model).
  - `base_mass` — Sum of that expert’s routing probability over all aligned steps on the **base** run (Top-K only; absent experts contribute 0).
  - `var_mass` — Same as above for the **var** run.
  - `delta` — `var_mass − base_mass`.
  - `abs_delta` — `abs(delta)`, used for ranking within each layer.

- `meta.json` — Bookkeeping for sequence lengths:
  - `input_len_base`, `gen_len_base`, `input_len_var`, `gen_len_var`, and `gen_steps_aligned`.

- `layers.json` — List of MoE/router layer names included in analysis (row order for the heatmap).