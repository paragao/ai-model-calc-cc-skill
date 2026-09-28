# Formulas

Every formula here was read from the engine source, not from prose. Use this file to explain a result or sanity-check one that looks wrong.

Throughout: `d` is hidden dimension, `DP = gpus / (TP * PP * CP)`, and `PARAM_BYTES` is 2 under BF16.

## Phase 1: Memory

Computed per (variant, hardware, ZeRO stage, micro batch size) combination.

- **Model parameters**: `(attn + FFN + expert + embed) * PARAM_BYTES / (TP * PP) + (ln + router) * PARAM_BYTES / PP`. TP shards attention projections, FFN matrices, expert weights and embeddings. Layer norm and router weights are small and replicated across the TP group. Divided further by DP under ZeRO-3.
- **Gradients**: same size as model parameters per GPU. Sharded under ZeRO-2 and ZeRO-3.
- **Optimizer states**: split by EP and DP — `model_mem * dense_frac * OPTIM_BYTES / DP + model_mem * moe_frac * OPTIM_BYTES / eff_dp_moe`, where `eff_dp_moe = DP / EP` when `DP > EP`, else `DP`. This models the fact that expert parameters are shared only within the EP group.
- **Activations**, MoE-aware per layer:
  - MoE layers: `seq * micro * (d * 12 + topk * expert_ffn * 2 * PARAM_BYTES)`
  - Dense layers: `seq * micro * d * 12`
  - Both multiplied by `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER` (0.45).
  - Not divided by TP. This is deliberately conservative.
- **VP activation overhead**, when `PP > 1` and `VP > 1`: `(PP - 1) * VP * layers_per_virtual_chunk * avg_act_per_layer`. These are full, non-checkpointed activations for in-flight micro-batches during interleaved 1F1B warmup.
- **Communication buffers**: `NCCL_MEM_BUF` (6 GB) + `NCCL_EP_SCALING_FACTOR * (EP - 1)` + MoE dispatch + PP send/recv.
  - MoE dispatch: `moe_layers_per_gpu * 2 * micro * seq * d * PARAM_BYTES * (topk / experts_per_gpu)`
  - PP buffers: `2 * micro * seq * d * PARAM_BYTES`, one send plus one receive for a single micro-batch activation.
- **Fragmentation**: the raw total is multiplied by `FRAGMENTATION_FACTOR` (1.24), an empirical PyTorch caching-allocator overhead calibrated against Qwen3-235B on 64 B300 GPUs.

The phase reports the minimum ZeRO stage and maximum micro batch size that fit within `mem_gb * GPU_MEM_UTILIZATION_THRESHOLD` (0.92).

Static estimates of this kind have historically run low against measured peak reserved memory. Treat a result near the ceiling as not fitting.

## Phase 2: Batch configuration

Sweeps `MICROS` (`[1, 2, 4, 8, 16]`) against the memory fit from Phase 1, targeting `TOKENS_PER_BATCH` (4e6) within `MARGIN` (2%), capped at `MAX_TOKENS_PER_BATCH` (600e6).

`global_batch = micro * accum * DP`, and `tokens_per_batch = global_batch * seq_len`.

## Phase 3: Training time

Sweeps `GRAD_ACCUM_VALUES` (`[2, 4, 8]`), deriving global batch and step count from each. Sequence length is per variant, falling back to `SEQ_LEN`.

**FLOPs** — two terms, not just `6ND`:

```
linear_flops    = 6 * TOTAL_TOKENS * active_params_B * 1e9
attention_flops = ATTENTION_FLOPS_FACTOR * layers * seq_len * TOTAL_TOKENS * d
total_flops     = linear_flops + attention_flops
```

The factor 6 is one forward plus two backward passes. `ATTENTION_FLOPS_FACTOR` is 12, covering the QK^T and softmax·V matmuls across forward and backward that `6ND` omits. Per sequence that cost is `factor * layers * seq_len² * d`; across `TOTAL_TOKENS / seq_len` sequences it reduces to the form above, which is **linear in `seq_len`**. So at a fixed token budget a 32K-context run costs more attention FLOPs than a 4K one. The term is skipped entirely if any of `layers`, `d`, `seq_len` or the factor is zero.

**Time:**

```
achieved_tflops_per_sec = peak_tflops * gpus * MFU * efficiency
compute_seconds         = total_flops / (achieved_tflops_per_sec * 1e12)
months                  = compute_seconds / (24 * 3600 * 30.44)
```

**Overhead on top of ideal compute:**

```
bf             = min(bubble_fraction, 0.99)
bubble_seconds = compute_seconds * bf / (1 - bf)
steps          = TOTAL_TOKENS / tok_batch
comm_seconds   = (exposed_comm_ms_per_step / 1000) * steps
total_seconds  = compute_seconds + bubble_seconds + comm_seconds
```

The bubble inflates compute time multiplicatively; exposed communication is added absolutely. The result reduces exactly to the FLOPs-only estimate when both are zero. Reported as `est_months` with its compute / bubble / comm split and each as a percentage of total, then scaled by `TIME_ESTIMATE_LOW_MULTIPLIER` (0.75) and `TIME_ESTIMATE_HIGH_MULTIPLIER` (1.25) for the range.

## Phase 4: ZeRO communication

Both use `DP = gpus / (TP * PP * CP)` and the hardware's `inter_node_bw_gb`.

**ZeRO-2 reduce-scatter** — model plus gradients, so twice model memory:

```
rs_volume_mb = model_mem_gb * 2 / DP * 1000
rs_time_ms   = model_mem_gb * 2 / DP / inter_node_bw_gb * 1000
```

**ZeRO-1 all-gather** — optimizer states only, not gradients. The factor 3 is `OPTIM_BYTES / PARAM_BYTES = 6 / 2`:

```
ag_volume_mb = model_mem_gb * 3 / DP * 1000
ag_time_ms   = ag_volume_mb / inter_node_bw_gb
```

ZeRO-1 moves more bytes per step than ZeRO-2 here (factor 3 against 2) but is rated against lower thresholds, because it communicates once per optimizer step rather than per micro-batch. Overhead ratings are heuristics: ZeRO-2 bands at 2 / 5 / 10 ms, ZeRO-1 at 1.5 / 3 / 7 ms.

## Phase 5: MoE all-to-all

For MoE models, tokens route to expert-holding GPUs via all-to-all. `FWD_BWD_ROUTING_BUFF_PASSES` is 2, for forward and backward. EP ranks inside one node use NVLink; across nodes, EFA. Dense models report zero.

## Phase 6: PP SendRecv

Zero when `PP = 1`.

```
activation_size_bytes = micro * seq_len * d * dtype_bytes
sends_per_microbatch  = 2 * (PP * VP - 1)   if VP > 1   # interleaved
                      = 2 * (PP - 1)        if VP == 1  # standard
total_sends_per_step  = sends_per_microbatch * num_microbatches
pipeline_bubble_pct   = (PP - 1) / (num_microbatches * VP) * 100
exposed_sends         = 2 * (PP - 1)        # warmup fwd + cooldown bwd
estimated_pp_time_ms  = exposed_sends * time_per_send_us / 1000
```

**Communication type** is decided by `is_inter_node = PP > gpus_per_node`, where **`gpus_per_node` defaults to 8 and is not read from the hardware entry**. On a platform with different density, such as GB200 NVL at 4 GPUs per node, this boundary is wrong and so is the resulting NVLink-versus-EFA choice.

**Effective bandwidth:**

- Intra-node: `intra_node_bw_gbps` directly, for example 1800 GB/s on B300, giving roughly 18 µs per send.
- Inter-node: `inter_node_bw_gb * EFA_PP_EFFICIENCY`, where `EFA_PP_EFFICIENCY` is 0.016 (in `utils/phase6_pp_comm.py`), because EFA point-to-point engages only one or two of the available NICs. On a 400 GB/s link that is roughly 3.2 GB/s, giving about 5.2 ms per send.

That three-orders-of-magnitude gap between intra-node and inter-node pipeline sends is why `PP` crossing the node boundary is expensive, and why `PP <= 8` is worth preserving where memory allows.

Only warmup and cooldown sends are serialized; the rest overlap with compute, which is why `exposed_sends` is `2 * (PP - 1)` rather than the full total.
