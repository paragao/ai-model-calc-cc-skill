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
| `CP` | 1 | Context parallelism, splitting the sequence. |
| `EP` | 8 | Expert parallelism. Must divide the expert count evenly. Set 1 for dense. |

**Start at `PP=1`.** The committed default of `PP=8, VP=4` is not a recommendation — it is one configuration among many, and pipeline parallelism buys memory relief at the cost of bubble time. Raise `PP` only when the model genuinely does not fit, and check Phase 1 headroom before concluding it must.

Typical `VP` pairings when you do use pipelining: `VP=4` with `PP=8`, `VP=2` with `PP=4`. `VP` increases the send count in Phase 6 while shrinking the bubble, so it is a trade rather than a free win.

Constraint: the cluster's `gpus` must be divisible by `TP * PP * CP`, or validation fails.

### Training

| Parameter | Default | Meaning |
|---|---|---|
| `TOTAL_TOKENS` | 15e12 | Dataset size in tokens. |
| `PRECISION` | `"BF16"` | `"BF16"` or `"FP8"`. |

### Performance estimation

| Parameter | Default | Meaning |
|---|---|---|
| `MFU` | 0.40 | Model FLOPs Utilization, the fraction of peak achieved. 0.35–0.50 is typical. |
| `TIME_ESTIMATE_LOW_MULTIPLIER` | 0.75 | Optimistic bound. |
| `TIME_ESTIMATE_HIGH_MULTIPLIER` | 1.25 | Pessimistic bound. |
| `MARGIN` | 0.02 | Tolerance when matching a target batch size. |

`MFU` is the single most influential assumption in any training-time estimate. State the value used whenever you report a duration, and say that halving it doubles the answer.

### Derived

`EXPERTS_PER_GPU = N_EXPERTS // EP` is computed automatically. Do not set it.

## advanced_config.py

`configuration/advanced_config.py` — rarely needs changing, but these constants explain result shape.

| Constant | Value | Meaning |
|---|---|---|
| `TOKENS_PER_BATCH` | 4e6 | Target tokens per global batch. |
| `MAX_TOKENS_PER_BATCH` | 600e6 | Upper limit on batch tokens. |
| `MICROS` | `[1, 2, 4, 8, 16]` | Micro batch sizes swept. |
| `GRAD_ACCUM_VALUES` | `[1, 2, 4, 8]` | Gradient accumulation values swept in Phase 3. `1` is no accumulation, valid and often optimal at large DP. |
| `OPTIM_BYTES_PER_PARAM` | 12 | fp32 master 4 + fp32 momentum 4 + fp32 variance 4. |
| `OPTIM_BYTES` | derived | `OPTIM_BYTES_PER_PARAM / PARAM_BYTES` → 6.0 at BF16, 12.0 at FP8. A **multiplier on parameter memory**, not bytes per parameter, because `model_mem_gb` already carries `PARAM_BYTES`. Hardcoding 6 understated FP8 optimizer memory by 2x. |
| `ZERO3_MAX_LIVE_PARAMS` | 1e9 | DeepSpeed `stage3_max_live_parameters`, in parameter elements. |
| `ZERO3_PREFETCH_BUCKET_PARAMS` | 5e8 | DeepSpeed `stage3_prefetch_bucket_size`, in parameter elements. |
| `EMPIRICAL_ACT_MULTIPLIER` | 12 | Activation memory multiplier per layer. |
| `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER` | 0.45 | Reduction from selective checkpointing. |
| `NCCL_MEM_BUF` | 6.0 | Base NCCL buffer in GB, including CUDA context. |
| `NCCL_EP_SCALING_FACTOR` | 1.0 | Extra GB per EP rank beyond the first. |
| `ATTENTION_FLOPS_FACTOR` | 12 | Factor for the O(seq_len²) attention term. |
| `FRAGMENTATION_FACTOR` | 1.24 | Caching-allocator overhead. Empirical, 1.15–1.25. Note it multiplies the total **including** the explicit ZeRO-3 transient buffer, while its calibration note claims to cover unmodelled transient buffers; if the calibration run used ZeRO-3 those totals now double-count. The record does not state the stage. |
| `GPU_MEM_UTILIZATION_THRESHOLD` | 0.92 | Fit target: use at most 92% of GPU memory. |
| `MIN_STEPS` / `MAX_STEPS` | 200,000 / 1,200,000 | Bounds on acceptable step counts. |

Per-GPU memory constants (`H100_MEM_GB` 80, `H200_MEM_GB` 141, `B200_MEM_GB` 180, `B300_MEM_GB` 268, `GB200_MEM_GB` 186, `A10G_MEM_GB` 24, `L4_MEM_GB` 24, `L40S_MEM_GB` 48) are reference values. Actual memory comes from the hardware entry's `mem_gb`.

Note that `EFA_PP_EFFICIENCY` (0.016) is **not** here — it lives in `utils/phase6_pp_comm.py`.

## Precision

`PRECISION` selects parameter byte width **and** which `peak_tflops_*` field Phase 3 reads. Under FP8 the weights and gradients halve but the optimizer states do not, because Adam state stays fp32 at 12 bytes per parameter regardless of parameter precision — so memory savings are well under 50%. FP8 throughput is roughly 2x BF16 on H100/H200/B200/B300/GB200, L4 and L40S, and every preset now carries `peak_tflops_fp8` to reflect that. `g5` is the exception at `0`: A10G is Ampere and has no FP8 tensor cores, so an FP8 run there is refused rather than estimated.

## Before reporting a result

Confirm and state: the variant, the hardware preset, `PP`/`VP`/`TP`/`EP`, `PRECISION`, `MFU`, `TOTAL_TOKENS`, and the ZeRO stage the result assumes. A number without that context is not reproducible and should not be quoted.
