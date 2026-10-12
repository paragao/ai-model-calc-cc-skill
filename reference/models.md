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
| Qwen3.5-27B | 64 | 5120 | **256** | 64 | 17408 | 0 | 24/4 | 26.90 | 26.90 | 4096 | 1,2,4,8,16,32,64 |
| Qwen3.5-35B-A3B | 40 | 2048 | **256** | 0 | 0 | 512 | 16/2 | 34.66 | 3.45 | 4096 | 1,2,4,5,8,10,20,40 |
| Qwen3.5-122B-A10B | 48 | 3072 | **256** | 0 | 0 | 1024 | 32/2 | 122.11 | 9.77 | 4096 | 1,2,3,4,6,8,12,16,24,48 |
| Qwen3.5-397B-A17B | 60 | 4096 | **256** | 0 | 0 | 1024 | 32/2 | 396.35 | 17.35 | 4096 | 1,2,3,4,5,6,10,12,15,20,30,60 |
| DeepSeek-V3 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V3.1 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V3.2 | 61 | 7168 | (56) | 3 | 18432 | 2048 | 128/128 | 671.0 | 36.7 | 4096 | 1,61 |
| DeepSeek-V4 | 61 | 7168 | (56) | 0 | 3072 | 3072 | 128/1 | 1600.0 | 49.0 | 4096 | 1,61 |

### head_dim matters, and the default is often wrong

A **bold** `head_dim` is stored explicitly on the variant. A `(parenthesised)` value is not stored — it is what the engine derives as `d // q_heads`.

This distinction is load-bearing. The Qwen3 and Qwen3.5 MoE models decouple `head_dim` from `d / q_heads`: Qwen3-235B-A22B has `d=4096` and `q_heads=64`, so the default would give 64, but the real value is 128. Deriving it wrongly **halves** the computed attention parameters per layer.

So when you write a custom variant, or copy a row from this table into a `--json` file, **carry `head_dim` across explicitly** for any model where it is bold. Omitting it silently produces a 2x error in attention memory, with no warning.

**`head_dim` now also drives the FLOPs term, not just memory.** Phase 3's O(seq_len²) attention term is sized by the query-projection width `q_heads * (qk_head_dim + v_head_dim) / 2`, which falls back to `head_dim` when the two MLA fields are absent. So a wrong or missing `head_dim` on a bold row understates **estimated training time** as well as attention memory — by up to 18.4% on the affected variants. It is the same field with a second consequence, so the advice above has become more load-bearing rather than less.

**MLA needs two extra fields.** DeepSeek's qk and v head widths differ from each other as well as from `d / q_heads`, so `head_dim` alone cannot express them. Set `qk_head_dim` and `v_head_dim` explicitly:

| field | DeepSeek-V3 / V3.1 / V3.2 | from the published config |
|---|---|---|
| `qk_head_dim` | `192` | `qk_nope_head_dim 128 + qk_rope_head_dim 64` |
| `v_head_dim` | `128` | `v_head_dim` |

Over 128 heads against `d = 7168` that gives a width of `20480`, a **2.86x** correction. All three configs were checked individually rather than assumed identical. Leave both unset for MHA/GQA, where the fallback is correct. **DeepSeek-V4 deliberately carries neither** — its config is unpublished, so its attention FLOPs are a known gap rather than a verified value, and copying V3's numbers across would invent data.

**GQA does not affect this width.** Both matmuls run over `q_heads` regardless of `kv_heads`, because each query head attends with its own (s, s) score matrix.

All `head_dim` values in the table above are verified against the published `config.json` for each model family, not inferred from the stored parameter counts.

### Qwen3.5 is a hybrid-attention family, and the calculator now models that

`Qwen/Qwen3.5-27B/config.json` declares a **3:1 hybrid attention stack**: `full_attention_interval: 4`, with three Gated DeltaNet `linear_attention` layers for every one `full_attention` layer. Across 64 layers that is **16 full-attention layers and 48 linear-attention layers**.

Both layer types are now counted from the published configs. A variant carrying `linear_attn_per_layer` charges `layers // full_attention_interval` layers at `attn_per_layer` and the rest at `linear_attn_per_layer`.

**A Gated DeltaNet layer is LARGER than a full-attention layer, not smaller:**

| Variant | full_attn | GDN | GDN/full |
|---|---|---|---|
| Qwen3.5-27B | 104,857,600 | 115,876,064 | **1.105x** |
| Qwen3.5-35B-A3B | 27,262,976 | 33,718,464 | **1.237x** |
| Qwen3.5-122B-A10B | 78,643,200 | 88,523,008 | **1.126x** |
| Qwen3.5-397B-A17B | 104,857,600 | 118,014,208 | **1.125x** |

Linear attention is cheaper in **time** — O(seq_len) rather than O(seq_len²) — but not in parameters. **An earlier version of this file claimed roughly 86% and called the uniform treatment an overstatement. Both were wrong: it was an understatement.**

The GDN weight set, per transformers' `Qwen3NextGatedDeltaNet`, with `key_dim = linear_num_key_heads * linear_key_head_dim` and `value_dim = linear_num_value_heads * linear_value_head_dim`:

```
in_proj_qkvz  d * (key_dim*2 + value_dim*2)     q, k, v and the z gate
in_proj_ba    d * (linear_num_value_heads * 2)  beta and alpha gates
conv1d        (key_dim*2 + value_dim) * linear_conv_kernel_dim   depthwise causal
out_proj      value_dim * d
```

**The full-attention layer was also wrong.** Every Qwen3.5 config sets `attn_output_gate: true`, which makes `q_proj` output `num_attention_heads * head_dim * 2`. The old `attn_per_layer` values omitted the gate and were **1.43x to 1.47x too small**.

With both corrected, all four variants reconcile to within 1% of their published sizes — 26.90 / 34.66 / 122.11 / 396.35 B against 27 / 35 / 122 / 397 B. That reconciliation is what validates the derivation; the old `total_params_B` of 24.35 was internally consistent with both errors, so nothing inside the catalog disagreed with it.

**Long-context training time** is also correct: the Phase 3 O(seq_len²) term counts only full-attention layers, where it previously applied to every layer, overstating time by 8% at 4K, 54% at 32K and 141% at 128K.

Qwen3.5 carries `vocab: 248320` per variant, so there is no global to adjust. The models are multimodal (`Qwen3_5ForConditionalGeneration`, with a `vision_config`: depth 27, hidden 1152), which this text-only calculator does not account for — that is the remaining sub-1% shortfall against published totals. Do **not** close it by inflating text parameters.

Notes on reading this table:

- `Active B` is what drives training-time FLOPs; `Total B` drives memory. For MoE they differ sharply.
- `valid_pp` lists the pipeline-parallel degrees that divide `layers` evenly. A `PP` outside this list fails validation. DeepSeek at 61 layers is prime, so only `PP=1` or `PP=61` are valid.
- All built-ins default to `seq_len` 4096. Override per variant in a `--json` file for long-context work.
- **Built-ins carry their own `vocab`, `num_experts` and `topk`, which override the globals in `configuration/project_config.py`.** That was already true of `vocab` and `num_experts`; it became true of `topk` only once `resolve_topk` shipped, so this bullet previously described an override the engine did not perform — Llama-4 Scout and Maverick were computed at the global 8 rather than 1 (overstating their expert activation and routing buffers 8x) and Qwen3.5-397B-A17B at 8 rather than 10.
  Measured coverage, rather than a blanket claim: `vocab` is present on 21 of 22 variants, and `num_experts` and `topk` on 10 of the 11 MoE variants. So editing `VOCAB`, `N_EXPERTS` or `TOPK` has no effect on any catalogued model **except DeepSeek-V4**, the sole exception on all three — it carries none of them, so the globals genuinely reach it and editing them does change its output. They are also the fallback for a `--json` variant that omits the fields.
  Two of DeepSeek-V4's three absent values are nonetheless derivable from its own entry rather than unsourced: `num_experts = 384` exactly and integrally from its own `moe_layer_params`, `(25,435,963,392 − 66,060,288 − 2,752,512) / 66,060,288 = 384.0`; and `topk = 8` from its own `active_params_B = 49.0`, where topk=6 yields 41.64 B (−15.0%) and topk=8 yields 49.70 B (+1.4%), the closest of twelve candidates — so the global it falls back to is already the right value. Its `vocab` is the real gap: absent, it resolves to the Qwen3 default 151,936 instead of DeepSeek's 129,280, overstating its two embedding tables 17.5%.
- Expert counts matter more than they look: routed memory is `experts_per_gpu * expert_each`, so falling back to the global 128 overstates a 16-expert model (Llama-4-Scout) 8x and understates a 512-expert one (Qwen3.5-397B) 4x.
- `Total B` for the Llama-4 and Qwen3.5 rows is the **published multimodal** figure. This calculator models text only, so its computed totals sit slightly below: Scout 107.8B computed + 0.87B vision tower = 108.6B against 109.0 published.
- Both Llama-4 variants carry a **shared expert on every MoE layer**, sized at the expert FFN (`3 × 5120 × 8192 = 125,829,120`) — not at `intermediate_size_mlp`. `intermediate_size_mlp: 16384` is vestigial for Scout, whose `interleave_moe_layer_step: 1` makes all 48 layers MoE.
- **Llama-3.2-3B ties its embeddings** (`tied_embeddings: True`) — the only variant that does. Its LM head IS the input embedding matrix.

## Defining a custom model

Write a JSON file and pass it with `--json FILE`. Only the variants in that file are analyzed; the built-ins are ignored. Start from `model_variant_template.json` in the repo.

Accepted shapes: a `{"variants": [...]}` wrapper, a bare list `[{...}]`, or a single object `{...}`.

**Required** — missing any of these fails with a clear error:

`name`, `layers`, `d`, `dense_layers`, `total_params_B`, `active_params_B`, `valid_pp`

**Strongly recommended for accuracy:**

`expert_ffn`, `dense_ffn`, `q_heads`, `kv_heads`. These drive the auto-computed parameter counts. Omitting `q_heads`/`kv_heads` falls back to a coarse `d^2 * 2.25` attention estimate.

**Optional:**

- `head_dim` — defaults to `d // q_heads`. **Supply it explicitly for any Qwen3 MoE or Qwen3.5 model**, which decouple it from `d / q_heads` (128 and 256 respectively); the default is 2x low there. Drives attention **memory and the O(seq_len²) FLOPs width**, so a wrong value understates estimated training time too.
- `qk_head_dim` / `v_head_dim` — **MLA only** (DeepSeek: `192` and `128`). Set both when the qk and v head widths differ from each other, which `head_dim` alone cannot express; the FLOPs width is then `q_heads * (qk_head_dim + v_head_dim) / 2`. Omit for MHA/GQA, where the `head_dim` fallback is correct. Omitting them on an MLA model understates its attention FLOPs by 2.86x, silently.
- `vocab` — per-variant vocabulary, else project-level `VOCAB` (correct for Qwen3 only)
- `tied_embeddings` — `True` when the LM head IS the input embedding matrix, so one table is counted instead of two. Overstates the model by ~12% on a 3B model if missed.
- `num_experts` (or `n_experts`) — per-variant expert count, else project-level `N_EXPERTS`. Drives routed memory directly, so the global fallback can be wrong by 8x.
- `topk` — per-variant routing width, else project-level `TOPK`
- `full_attention_interval` — for a hybrid model interleaving linear attention, 1 full-attention layer per N (Qwen3.5 uses 4). Keeps the O(seq_len²) FLOPs term on full-attention layers only, **and** splits attention memory between the two layer types. Omit for uniform attention.
- `linear_attn_per_layer` — the parameter count of ONE linear-attention layer, used with `full_attention_interval` to charge `layers - layers//interval` layers at this value instead of `attn_per_layer`. Omit for uniform attention, in which case every layer is charged at `attn_per_layer`.
- `kv_lora_rank` / `qk_rope_head_dim` — **MLA only, inference sizing only** (DeepSeek-V3.x: `512` and `64`). The KV cache stores one compressed latent of width `kv_lora_rank + qk_rope_head_dim` per layer, not per-head K and V. Without them the GQA formula reads `kv_heads=128` and overstates DeepSeek-V3's KV cache **24.9x** (1708 vs a real 68.6 KiB/token). Training memory does not use these.
- `kv_cache_unverified` — `True` when the attention/cache shape is unpublished (DeepSeek-V4, MQA-shaped with `kv_heads=1` and no MLA fields). The inference KV figure is then returned **flagged** rather than quietly computed, because neither the MLA nor the GQA formula can be trusted.
- `shared_each` — set above 0 to mark a shared-expert MoE. Note a shared expert is **always active** (it processes every token), so it belongs in `active_params_B` alongside `topk * expert_each`. Easy to miss: Qwen3-30B-A3B and Qwen3-235B-A22B both have `shared_each = 0`, so a formula omitting the term still reproduces them exactly.
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
