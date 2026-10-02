# Configuration

These parameters have **no CLI flags**. To change them you must edit the files. This is the one place the skill legitimately modifies the repository.

## project_config.py

`configuration/project_config.py` — edit per analysis. Values below are the committed defaults.

### Model architecture

| Parameter | Default | Meaning |
|---|---|---|
| `VOCAB` | 151,936 | Vocabulary size. **Fallback only** — every built-in carries its own `vocab`. Qwen3 = 151,936, Llama-3.x = 128,256, Llama-4 = 202,048, Qwen2.5 = 152,064, Qwen3.5 = 248,320, DeepSeek = 129,280. |
| `SEQ_LEN` | 4,096 | Sequence length. Used only when a variant does not set its own `seq_len`. |
| `TOPK` | 8 | Experts routed per token. **Fallback only** — overridden per variant by `topk`. Llama-4 routes to 1. |
| `N_EXPERTS` | 128 | Total experts per MoE layer. **Fallback only** — overridden per variant by `num_experts`. |

These four are fallbacks, not settings to tune per model: a catalogued variant's own value wins, so editing them changes nothing for the built-ins. Set them only when supplying a `--json` variant that omits the fields, or when analyzing **DeepSeek-V4**, the one built-in with no per-variant values (and no public config to verify any).

### Parallelism

| Parameter | Default | Meaning |
|---|---|---|
| `PP` | 8 | Pipeline parallelism. Must appear in the variant's `valid_pp`. |
| `VP` | 4 | Virtual pipeline parallelism, the interleaved schedule. 1 disables it. |
| `TP` | 1 | Tensor parallelism, splitting within layers. |
| `CP` | 1 | Context parallelism, splitting the sequence. **Divides activation memory** as `seq_len / CP`, reaching every sequence-dependent activation term, as well as dividing `DP`. |
| `EP` | 8 | Expert parallelism. Must divide the expert count evenly. Set 1 for dense. |

**Start at `PP=1`.** The committed default of `PP=8, VP=4` is not a recommendation — it is one configuration among many, and pipeline parallelism buys memory relief at the cost of bubble time. Raise `PP` only when the model genuinely does not fit, and check Phase 1 headroom before concluding it must.

Typical `VP` pairings when you do use pipelining: `VP=4` with `PP=8`, `VP=2` with `PP=4`. `VP` increases the send count in Phase 6 while shrinking the bubble, so it is a trade rather than a free win.

Constraint: the cluster's `gpus` must be divisible by `TP * PP * CP`, or validation fails.

### Training

| Parameter | Default | Meaning |
|---|---|---|
| `TOTAL_TOKENS` | 15e12 | Dataset size in tokens. |
| `PRECISION` | `"BF16"` | `"BF16"` or `"FP8"`. Changes **exactly one memory term**, `activation_gb`, plus the throughput peak. This row previously said "throughput-only — it changes no memory term". See the note below the table. |

### Performance estimation

| Parameter | Default | Meaning |
|---|---|---|
| `MFU` | 0.40 | Model FLOPs Utilization, the fraction of peak achieved. 0.35–0.50 is typical. |
| `TIME_ESTIMATE_LOW_MULTIPLIER` | 0.75 | Optimistic bound. |
| `TIME_ESTIMATE_HIGH_MULTIPLIER` | 1.25 | Pessimistic bound. |
| `MARGIN` | 0.02 | **No code consumers.** Was the tolerance for the removed target-match check. Kept as public configuration; widening it would **not** revive that check, since an exact `dp * micro * accum * seq_len` product cannot land within any margin of an arbitrary absolute target. |

`MFU` is the single most influential assumption in any training-time estimate. State the value used whenever you report a duration, and say that halving it doubles the answer.

### Derived

`EXPERTS_PER_GPU = N_EXPERTS // EP` is computed automatically. Do not set it.

## advanced_config.py

`configuration/advanced_config.py` — rarely needs changing, but these constants explain result shape.

| Constant | Value | Meaning |
|---|---|---|
| `TOKENS_PER_BATCH` | 4e6 | Llama 3 405B's **ramp-start** batch (arXiv:2407.21783 §3.4.1: *"an initial batch size of 4M tokens"*), not a steady-state target. The priority-0 check that used to read it has been **removed** as unreachable. It is now the reference batch for `compute_absurd_step_ceiling`, so **raising it tightens the absurdity floor** (`TOTAL_TOKENS / 4e6` = 3,750,000 steps) and flags more configurations. To bound the batch itself, change `MIN_STEPS`/`MAX_STEPS`. |
| `MAX_TOKENS_PER_BATCH` | 600e6 | **Reference only.** The effective cap is derived per configuration by `compute_max_tokens_per_batch(TOTAL_TOKENS, dp, seq_len)` = `max(TOTAL_TOKENS / MIN_STEPS, dp * seq_len)`. The fixed 600e6 implied 25,000 steps, **8x below** this module's own `MIN_STEPS`, so it never fired; the derived cap fires on 352 configurations. |
| `MICROS` | `[1, 2, 4, 8, 16]` | Micro batch sizes swept. |
| `GRAD_ACCUM_VALUES` | `[1, 2, 4, 8]` | Gradient accumulation values swept in Phase 3. `1` is no accumulation, valid and often optimal at large DP. |
| `OPTIM_BYTES_PER_PARAM` | 12 | fp32 master 4 + fp32 momentum 4 + fp32 variance 4. |
| `PARAM_STORAGE_BYTES` | 2 | Bytes the weights actually **occupy**, which is NOT `PARAM_BYTES`. `PARAM_BYTES` is a compute precision; weights stay resident in BF16 at either setting. Every memory term uses this. |
| `OPTIM_BYTES` | derived | `OPTIM_BYTES_PER_PARAM / PARAM_STORAGE_BYTES` → a constant **6.0 at either precision**. A **multiplier on parameter memory**, not bytes per parameter, because `model_mem_gb` carries `PARAM_STORAGE_BYTES`. Dividing by `PARAM_BYTES` instead gives 12.0 at FP8, a **2x overstatement**. |
| `GRAD_BYTES_PER_PARAM` | 2 | BF16 gradient accumulation and reduction. |
| `GRAD_BYTES` | derived | `GRAD_BYTES_PER_PARAM / PARAM_STORAGE_BYTES` → a constant **1.0 at either precision**. Also a **multiplier on parameter memory**. Gradients are NOT stored at `PARAM_BYTES`: FP8 is used for the GEMM inputs while gradients reduce in BF16 or higher, so FP8 gradient memory does not halve. Without this an FP8 run charged 1 byte/param — 1.923 GB on Llama-3.1-8B against the correct 3.846. |
| `LOGITS_BYTES_PER_ELEMENT` | 8 | Bytes per logit element: a bf16 logits tensor plus the fp32 softmax and its gradient. Set to 0 to disable the output-logits term. |
| `LOGITS_FUSED_CE` | `True` | Whether the loss is a fused or chunked cross-entropy, which keeps only one chunk of logits live. `False` charges the full `(micro, seq_len, vocab)` tensor, correct for a plain `nn.CrossEntropyLoss`. Worth **628 configurations** across the sweep, so it is a stated assumption. |
| `LOGITS_CHUNK_TOKENS` | 1024 | Rows whose logits are live at once under fused CE. **An assumption, not a measured value** — real implementations size the chunk from available memory. Above it the logits term stops scaling with micro. |
| `ZERO3_MAX_LIVE_PARAMS` | 1e9 | DeepSpeed `stage3_max_live_parameters`, in parameter elements. |
| `ZERO3_PREFETCH_BUCKET_PARAMS` | 5e8 | DeepSpeed `stage3_prefetch_bucket_size`, in parameter elements. |
| `EMPIRICAL_ACT_MULTIPLIER` | 12 | Activation memory multiplier per layer. Implicitly **bytes per element** (6 tensors × 2 bytes), since the expression divides by `1e9` with no other byte factor. |
| `ACT_BYTES_PER_ELEMENT` | 2 | Bytes per element for the MoE expert activation intermediate. Activations are resident in BF16 at either precision, so this replaced a `PARAM_BYTES` factor that made FP8 save 20-40% on MoE activations and exactly 0% on dense ones — not a physical distinction. Scaling the dense term down instead was **measured and refuted**: it moved the predicted FP8/BF16 ratio from 0.9028 to 0.8528 against a measured **1.0873**. |
| `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER` | 0.45 | Reduction from selective checkpointing. |
| `RECOMPUTE_GRANULARITY` | `None` | `None` or `"full"`. `"full"` recomputes whole blocks, as `recompute_granularity="full"` in a Megatron/NeMo recipe. **Default recomputes nothing**, so results are unchanged until set. |
| `RECOMPUTE_NUM_LAYERS` | 0 | Layers recomputed per **pipeline chunk**, matching Megatron's `recompute_num_layers` under `recompute_method="block"`. The recomputed **fraction** is this over `layers_per_gpu / VP`, so it depends on PP and VP too: at `N = 1` on 80 layers it is **100%** at PP=4/VP=20, **50%** at PP=2/VP=20 and **1.25%** at PP=1. Changing PP therefore moves activation memory several-fold. |
| `RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT` | 2 | Bytes a recomputed block still stores for its own input, so the scale blends the stored and boundary costs rather than being `1 - fraction`. |
| `NCCL_MEM_BUF` | 6.0 | Base NCCL buffer in GB, including CUDA context. |
| `NCCL_EP_SCALING_FACTOR` | 1.0 | Extra GB per EP rank beyond the first. |
| `ATTENTION_FLOPS_FACTOR` | 12 | Factor for the O(seq_len²) attention term. **The width it multiplies is `compute_attention_flops_dim(variant)`, not `d`** — `q_heads * (qk_head_dim + v_head_dim) / 2`, which equals `d` only for standard MHA/GQA. It differs for 10 of 22 variants (seven Qwen3/Qwen3.5 with a stored `head_dim` ≠ `d / q_heads`, and the three MLA DeepSeek entries at 2.86x), where using `d` understated `est_months` by up to 18.4%. Set to 0 to disable the term. |
| `FRAGMENTATION_FACTOR` | 1.24 | Caching-allocator overhead. Empirical, 1.15–1.25. **No double-count with the ZeRO-3 transient buffer**, though it multiplies the total including it: the harness reproducing the measured 150 GB pins `zero_strategy=1` at DP=1, where ZeRO-1 and ZeRO-2 are byte-identical and the buffer is 0.00 GB, so the 1.24x was never fitted to it. Worst case had it been a ZeRO-3 run: 2.00 GB of 124.2 GB, 1.6%. |
| `GPU_MEM_UTILIZATION_THRESHOLD` | 0.92 | Fit target: use at most 92% of GPU memory. |
| `MIN_STEPS` / `MAX_STEPS` | 200,000 / 1,200,000 | Bounds on acceptable step counts, and **the real control on batch size** since the derived cap is computed from `MIN_STEPS`. The band implies a batch window of **12.5M–75.0M tokens**, and both published frontier runs land inside it: Llama 3 405B finishes at 16.0M, DeepSeek-V3 at 62.9M (15360 seqs x 4096, arXiv:2412.19437 §4.2). `MAX_STEPS` is Llama 3's own reported count (*"over 1,200,000 steps"*), and its published 4M→8M→16M ramp reconstructs to 1,154,406 steps — a 3.8% agreement. |

Per-GPU memory constants (`H100_MEM_GB` 80, `H200_MEM_GB` 141, `B200_MEM_GB` 180, `B300_MEM_GB` 268, `GB200_MEM_GB` 186, `A10G_MEM_GB` 24, `L4_MEM_GB` 24, `L40S_MEM_GB` 48) are reference values. Actual memory comes from the hardware entry's `mem_gb`.

Note that `EFA_PP_EFFICIENCY` (0.016) is **not** here — it lives in `utils/phase6_pp_comm.py`.

## Precision

`PRECISION` selects the **compute** (GEMM) precision and which `peak_tflops_*` field Phase 3 reads. It changes **exactly one memory term**: `activation_gb`. This note previously said it changed no memory term at all and that a 704-pair sweep was byte-identical between the two settings; that was wrong on activations, and the sweep now differs there and only there.

An earlier version of this file said "under FP8 the **weights** halve". That is wrong, and wrong in **sign**. On this stack FP8 is requested with `--compute_dtype fp8_cs`, the only precision flag the recipe passes — there is no separate storage or params dtype. Transformer Engine keeps parameters resident in BF16 and *adds* cached FP8 cast copies, each carrying ONE FP32 per-tensor scale factor — `fp8_cs` is *current* scaling, so there is **no amax history buffer** and the metadata is negligible. The one controlled measurement available (Llama 70B LoRA finetune, H200 p5en, 32 GPUs, TP=4 PP=2 VP=20 MBS=1, identical cluster and container with precision the only difference) has FP8 at **41.1 GB allocated against BF16's 37.8 GB — +8.73%**. FP8 cost *more* memory.

So weights are held at `PARAM_STORAGE_BYTES` (2) regardless, gradients at `GRAD_BYTES_PER_PARAM` (2) and optimizer state at fp32 12 bytes/param. **Activations are the exception**: the MoE expert intermediate stays at `ACT_BYTES_PER_ELEMENT` (2), but the d-proportional per-layer term falls by `ACT_PRECISION_SAVING_FRACTION` at FP8 — 3.5 of `EMPIRICAL_ACT_MULTIPLIER`'s 12 bytes per element, **29.17%**, scaled by `(1 - recompute_fraction)`.

That activation saving **is applied**, and it is measured twice over. A 1-GPU TE probe gives 3.5000 B/element at r² = 1.000000 with `3.5*(1-f)` scaling across `f` = 0, 0.25, 0.50, 0.75, 1.00 (jobs 348/350). A 2x4 crossed FP8 × recompute design on real Llama-3.1-8B full pretraining (8×H100, TP=1 PP=1 CP=1) then reproduced it, with the saving and its slope fixed from a pre-existing pair and written down *before* the runs:

| layers recomputed | bf16 | fp8_cs | saving | predicted |
|---|---|---|---|---|
| 0/32 | 84.604 | 78.698 | **+5.906** | +5.906 |
| 5/32 | 79.230 | 74.247 | +4.983 | +4.983 |
| 16/32 | 67.406 | 64.453 | +2.953 | +2.953 |
| 32/32 | 50.208 | 51.543 | −1.335 | +0.000 |

The 0/32 cell discriminates `(1-f)` scaling (predicting 78.698) from pure additivity (79.621) and lands on 78.698 to three decimals. So FP8 and activation checkpointing are **substitutes**: they compete for the same bytes.

**On the WEIGHT side the FP8 figures remain a lower bound.** The cast copies are measured (jobs 350/351, H200, TE 2.9.0, r² = 1.000000, two independent arms) at **2.0 bytes/param scaled by `(1 - recompute_fraction)`**, falling 1.9998 → 1.4998 → 0.9999 → 0.5000 → 0.0001 as `f` goes 0 → 1, because a recomputed block keeps neither its activations nor its fp8 weight copy. A **frozen** base (LoRA) keeps *both* copies but loses the activation saving entirely, since a frozen weight needs no wgrad and wgrad is what consumes the saved GEMM input — which is why the LoRA pair could never test the activation term. That weight constant is **not** applied: at the pair's `f=0.50` it overshoots its measured +3.300 GB by **2.67x**, and the untested cause, pipelined residency, is beyond what a single-GPU probe can measure.

The 32/32 inversion above is not an activation effect either. With the activation term zeroed it is the **fp8 parameter all-gather buffer**, proven causally by re-running that cell with `fp8_param_gather=false`, which recovered the bf16 transient exactly. It is transient rather than resident, so it is not modelled. Note its sign: *disabling* that feature adds **13.959 GB** of steady memory, so it is worth about +12.6 GB net — "FP8 weight cast copies are not resident" holds *because* fp8 parameter storage nets them out, not structurally.

FP8 throughput is roughly 2x BF16 on H100/H200/B200/B300/GB200, L4 and L40S, and every preset carries `peak_tflops_fp8` to reflect that. `g5` is the exception at `0`: A10G is Ampere and has no FP8 tensor cores, so an FP8 run there is refused rather than estimated.

## Before reporting a result

Confirm and state: the variant, the hardware preset, `PP`/`VP`/`TP`/`EP`, `PRECISION`, `MFU`, `TOTAL_TOKENS`, and the ZeRO stage the result assumes. A number without that context is not reproducible and should not be quoted.
