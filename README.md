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
}
```

figures and CSVs under out_dir/.