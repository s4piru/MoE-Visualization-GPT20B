MoE Top-K visualization for gpt-oss given two prompts (base/var) and one question.

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
  "prompt_base": "You are a helpful assistant.",
  "prompt_var":  "You are a terse, compliance-focused assistant.",
  "question":    "Explain how TLS works at a high level.",
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

- jsd_heatmap.png — Layer × generation step JSD (base vs var)

- jsd_layer_mean.png — Mean JSD per layer

- expert_shifts_top20.csv — Top (layer, expert) probability increases (var - base)

- meta.json, layers.json

## Tested Environments

Lambda Cloud — gpu_1x_h100_pcie (Recommended)
- GPU: NVIDIA H100 80GB (PCIe)
- OS: Ubuntu 22.04