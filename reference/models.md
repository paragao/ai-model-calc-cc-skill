# Model catalog

## Built-in variants

22 variants ship in `configuration/variants_config.py`. Select one with `--model NAME` (case-insensitive, exact match).

| Name | Layers | d | head_dim | Dense layers | dense_ffn | expert_ffn | q/kv heads | Total B | Active B | seq_len | valid_pp |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Llama-3.1-8B | 32 | 4096 | (128) | 32 | 14336 | 0 | 32/8 | 8.03 | 8.03 | 4096 | 1,2,4,8,16,32 |
| Llama-3.1-70B | 80 | 8192 | (128) | 80 | 28672 | 0 | 64/8 | 70.6 | 70.6 | 4096 | 1,2,4,5,8,10,16,20,40,80 |
| Llama-3.1-405B | 126 | 16384 | (128) | 126 | 53248 | 0 | 128/8 | 405.0 | 405.0 | 4096 | 1,2,3,6,7,9,14,18,21,42,63,126 |
| Llama-3.2-3B | 28 | 3072 | (128) | 28 | 8192 | 0 | 24/8 | 3.21 | 3.21 | 4096 | 1,2,4,7,14,28 |
| Llama-3.3-70B | 80 | 8192 | (128) | 80 | 28672 | 0 | 64/8 | 70.6 | 70.6 | 4096 | 1,2,4,5,8,10,16,20,40,80 |
| Llama-4-Scout-17B-16E | 48 | 5120 | (128) | 0 | 16384 | 8192 | 40/8 | 109.0 | 17.0 | 4096 | 1,2,3,4,6,8,12,16,24,48 |
| Llama-4-Maverick-17B-128E | 48 | 5120 | (128) | 24 | 16384 | 8192 | 40/8 | 400.0 | 17.0 | 4096 | 1,2,3,4,6,8,12,16,24,48 |
| Qwen2.5-7B | 28 | 3584 | (128) | 28 | 18944 | 0 | 28/4 | 7.62 | 7.62 | 4096 | 1,2,4,7,14,28 |
| Qwen2.5-32B | 64 | 5120 | (128) | 64 | 27648 | 0 | 40/8 | 32.76 | 32.76 | 4096 | 1,2,4,8,16,32,64 |
| Qwen2.5-72B | 80 | 8192 | (128) | 80 | 29568 | 0 | 64/8 | 72.7 | 72.7 | 4096 | 1,2,4,5,8,10,16,20,40,80 |
| Qwen3-8B | 36 | 4096 | (128) | 36 | 11008 | 0 | 32/8 | 8.04 | 8.04 | 4096 | 1,2,3,4,6,9,12,18,36 |
| Qwen3-32B | 64 | 5120 | **128** | 64 | 25600 | 0 | 64/8 | 30.5 | 30.5 | 4096 | 1,2,4,8,16,32,64 |
| Qwen3-30B-A3B | 48 | 2048 | **128** | 0 | 6144 | 768 | 32/4 | 30.53 | 3.35 | 4096 | 1,2,3,4,6,8,12,16,24,48 |
| Qwen3-235B-A22B | 94 | 4096 | **128** | 0 | 12288 | 1536 | 64/4 | 235.09 | 22.19 | 4096 | 1,2,47,94 |
| Qwen3.5-27B | 64 | 5120 | **256** | 64 | 17408 | 0 | 24/4 | 24.35 | 24.35 | 4096 | 1,2,4,8,16,32,64 |
| Qwen3.5-35B-A3B | 40 | 2048 | **256** | 0 | 0 | 512 | 16/2 | 34.13 | 2.93 | 4096 | 1,2,4,5,8,10,20,40 |
| Qwen3.5-122B-A10B | 48 | 3072 | **256** | 0 | 0 | 1024 | 32/2 | 120.55 | 8.21 | 4096 | 1,2,3,4,6,8,12,16,24,48 |
| Qwen3.5-397B-A17B | 60 | 4096 | **256** | 0 | 0 | 1024 | 32/2 | 393.74 | 14.74 | 4096 | 1,2,3,4,5,6,10,12,15,20,30,60 |
| DeepSeek-V3 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V3.1 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V3.2 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V4 | 61 | 7168 | (56) | 0 | 3072 | 3072 | 128/1 | 1600.0 | 49.0 | 4096 | 1,61 |

### head_dim matters, and the default is often wrong

A **bold** `head_dim` is stored explicitly on the variant. A `(parenthesised)` value is not stored — it is what the engine derives as `d // q_heads`.

This distinction is load-bearing. The Qwen3 and Qwen3.5 MoE models decouple `head_dim` from `d / q_heads`: Qwen3-235B-A22B has `d=4096` and `q_heads=64`, so the default would give 64, but the real value is 128. Deriving it wrongly **halves** the computed attention parameters per layer.

So when you write a custom variant, or copy a row from this table into a `--json` file, **carry `head_dim` across explicitly** for any model where it is bold. Omitting it silently produces a 2x error in attention memory, with no warning.

All `head_dim` values in the table above are verified against the published `config.json` for each model family, not inferred from the stored parameter counts.

### Qwen3.5 is a hybrid-attention family, and the calculator does not model that

This is the most important limitation in the catalog. `Qwen/Qwen3.5-27B/config.json` declares a **3:1 hybrid attention stack**: `full_attention_interval: 4`, with three Gated DeltaNet `linear_attention` layers for every one `full_attention` layer. Across 64 layers that is **16 full-attention layers and 48 linear-attention layers**, the linear ones carrying their own shapes (`linear_key_head_dim: 128`, `linear_num_key_heads: 16`, `linear_num_value_heads: 48`).

The calculator treats every layer as standard full attention. Two consequences:

1. **Attention memory is overstated.** `attn_per_layer` is applied to all 64 layers when only 16 of them have full-attention projections.
2. **Long-context training time is overstated more severely.** The Phase 3 attention term is O(seq_len²) and is applied to every layer, but linear attention is O(seq_len). At 32K context the error compounds badly.

Two further mismatches: Qwen3.5 uses `vocab_size: 248320`, not the 151,936 default in `project_config.py`, and the models are multimodal (`Qwen3_5ForConditionalGeneration`, with a `vision_config`), which the calculator does not account for at all.

Treat Qwen3.5 numbers as an upper bound on attention cost rather than an estimate, and say so when reporting them.

Notes on reading this table:

- `Active B` is what drives training-time FLOPs; `Total B` drives memory. For MoE they differ sharply.
- `valid_pp` lists the pipeline-parallel degrees that divide `layers` evenly. A `PP` outside this list fails validation. DeepSeek at 61 layers is prime, so only `PP=1` or `PP=61` are valid.
- All built-ins default to `seq_len` 4096. Override per variant in a `--json` file for long-context work.
- The built-ins do not set a per-variant expert count; MoE expert count comes from the project-level `N_EXPERTS` in `configuration/project_config.py`. A custom variant can override it with `num_experts`.

## Defining a custom model

Write a JSON file and pass it with `--json FILE`. Only the variants in that file are analyzed; the built-ins are ignored. Start from `model_variant_template.json` in the repo.

Accepted shapes: a `{"variants": [...]}` wrapper, a bare list `[{...}]`, or a single object `{...}`.

**Required** — missing any of these fails with a clear error:

`name`, `layers`, `d`, `dense_layers`, `total_params_B`, `active_params_B`, `valid_pp`

**Strongly recommended for accuracy:**

`expert_ffn`, `dense_ffn`, `q_heads`, `kv_heads`. These drive the auto-computed parameter counts. Omitting `q_heads`/`kv_heads` falls back to a coarse `d^2 * 2.25` attention estimate.

**Optional:**

- `head_dim` — defaults to `d // q_heads`
- `num_experts` (or `n_experts`) — per-variant expert count, else project-level `N_EXPERTS`
- `shared_each` — set above 0 to mark a shared-expert MoE
- `seq_len` — per-variant context length, else project-level `SEQ_LEN`

**Never provide** — these are computed automatically and supplying them will not help:

`attn_per_layer`, `expert_each`, `router_each`, `moe_layer_params`, `dense_layer_params`

**Dense vs MoE:**

- Dense: `expert_ffn = 0`, `dense_layers = layers`, omit `num_experts`.
- MoE: `expert_ffn > 0`, `dense_layers` = count of non-MoE layers (0 if every layer is MoE), set `num_experts`.

### Example

```json
{
  "variants": [
    {
      "name": "MyModel-30B-A3B",
      "layers": 52,
      "d": 2688,
      "dense_layers": 1,
      "dense_ffn": 3712,
      "expert_ffn": 1856,
      "q_heads": 32,
      "kv_heads": 2,
      "num_experts": 128,
      "total_params_B": 30.0,
      "active_params_B": 3.0,
      "seq_len": 4096,
      "valid_pp": [1, 2, 3, 4, 6, 8]
    }
  ]
}
```

```bash
python3 ai_model_calculator.py --json my_model.json --hw-name "128 p5en"
```

### On sequence length

`seq_len` is used for memory, buffers, communication and tokens-per-batch, and in the training-time attention term. Raising it to 32768 increases activation memory, communication volume, and the O(seq_len²) attention FLOPs, so a 32K run estimates a longer time than a 4K run at the same total token budget. Set it deliberately.
