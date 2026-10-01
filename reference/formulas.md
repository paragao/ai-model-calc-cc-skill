# Formulas

Every formula here was read from the engine source, not from prose. Use this file to explain a result or sanity-check one that looks wrong.

Throughout: `d` is hidden dimension, `DP = gpus / (TP * PP * CP)`, and memory terms use `PARAM_STORAGE_BYTES`, which is **2 at either precision**. `PARAM_BYTES` is a *compute* precision and reaches no memory term.

## Phase 1: Memory

Computed per (variant, hardware, ZeRO stage, micro batch size) combination.

- **Model parameters**: `(attn + FFN + expert + embed) * PARAM_STORAGE_BYTES / (TP * PP) + (ln + router) * PARAM_STORAGE_BYTES / PP`. TP shards attention projections, FFN matrices, expert weights and embeddings. Layer norm and router weights are small and replicated across the TP group. Under ZeRO-3 the **resident** parameters are sharded too, using the same dense/MoE split as the optimizer below, so experts shard across `DP / EP` rather than full DP. ZeRO-1 and ZeRO-2 keep a full replica on every rank.
  - `expert` is `experts_per_gpu * expert_each`, and `experts_per_gpu` comes from the variant's own `num_experts // EP` when it declares one, falling back to the global `N_EXPERTS // EP` otherwise. The fallback is wrong by 8x for a 16-expert model and 4x for a 512-expert one.
  - `embed` counts **two** tables (input embedding + LM head) by default, or **one** when the variant sets `tied_embeddings: True`, where the LM head IS the input embedding matrix.
  - `attn` is split **by layer type** on a hybrid family. A variant carrying `linear_attn_per_layer` charges `layers_per_gpu // full_attention_interval` layers at `attn_per_layer` and the remainder at `linear_attn_per_layer`. A Gated DeltaNet layer is **1.10x to 1.24x** a full-attention layer, so charging every layer at `attn_per_layer` understates a hybrid model — linear attention is cheaper in time, not in parameters. Without the field every layer is full attention.
- **ZeRO-3 transient all-gather buffer**: a sharded layer **cannot execute**. DeepSpeed all-gathers each module's full parameters into a transient buffer immediately before running it and frees them after, so real peak exceeds the resident figure:

  ```
  live_params  = max(largest_unit, min(1e9, largest_unit + 5e8))
  transient_gb = live_params * PARAM_STORAGE_BYTES / 1e9
  ```

  The floor is the largest single gathered unit, because that module cannot run without being gathered whole. The two bounds are DeepSpeed's own documented defaults in **parameter elements** — `stage3_max_live_parameters` 1e9 and `stage3_prefetch_bucket_size` 5e8, per [the config reference](https://www.deepspeed.ai/docs/config-json/). They are config knobs, so this models the **default** configuration; lowering them trades memory back for communication.

  - **The largest gathered unit is usually the embedding table, not a transformer layer** — true for 18 of the 22 built-in variants. Llama-3.2-3B's 128256 x 3072 table exceeds any of its 28 layers.
  - **It does not shrink with DP.** Resident memory is `Psi/Nd` and falls away; the buffer is fixed by the largest module and the config caps. For Llama-3.1-70B it is 2.10 GB at every DP, against resident 2.66 GB at DP=8 and 0.02 GB at DP=1024. Past roughly DP=32 the buffer, not the resident shard, dominates ZeRO-3's parameter footprint.

  Reported as `zero3_transient_gb` in the Phase 1 CSV and as the `AGbuf` console column, blank on ZeRO-1/ZeRO-2 rows. **ZeRO-3 therefore does not always save memory**: at large DP with a small model it can cost more than it saves, and the Phase 1 recommendation says so rather than always advising it.
- **Gradients**: `model_mem * GRAD_BYTES`, sharded under ZeRO-2 and ZeRO-3. `GRAD_BYTES` is a **multiplier on parameter memory**, the same shape as `OPTIM_BYTES`, derived as `GRAD_BYTES_PER_PARAM / PARAM_STORAGE_BYTES` — a constant **1.0** — so the product holds at **2 bytes/param at both BF16 and FP8**. Gradients are NOT stored at `PARAM_BYTES`: an FP8 recipe uses FP8 only for the GEMM inputs and reduces gradients in BF16 or higher. Without the multiplier an FP8 run charged 1 byte/param — 1.923 GB on Llama-3.1-8B against the correct 3.846.
- **Optimizer states**: split by EP and DP — `model_mem * dense_frac * OPTIM_BYTES / DP + model_mem * moe_frac * OPTIM_BYTES / eff_dp_moe`, where `eff_dp_moe = DP / EP` when `DP > EP`, else `DP`. This models the fact that expert parameters are shared only within the EP group.

  `OPTIM_BYTES` is a **multiplier on parameter memory**, not bytes per parameter — `model_mem` carries `PARAM_STORAGE_BYTES`. It is derived as `OPTIM_BYTES_PER_PARAM / PARAM_STORAGE_BYTES`, a constant **6.0**, so both precisions land on 12 bytes/param because Adam state stays fp32 regardless of parameter precision. Dividing by `PARAM_BYTES` instead gives 12.0 at FP8, a **2x overstatement**.

  **`moe_frac` is a fraction OF a TP-divided model memory**, so routed expert weights are divided by TP when computing it, and the expert count comes from the variant's own `num_experts` rather than the global `N_EXPERTS / EP`. A shared `resolve_experts_per_gpu` helper supplies both so the two memory functions cannot disagree. Neither is cosmetic: without the TP division the ratio exceeds 1.0 for **any TP > 1** and clamps, zeroing `dense_frac` and pushing all optimizer and ZeRO-3 parameter sharding onto `eff_dp_moe` — measured on Qwen3-235B-A22B at TP=2, reported optimizer memory **rose** from 5.36 to 8.04 GB while TP had halved the weights. The global expert count was wrong for **7 of 11 MoE variants**, overstating Llama-4-Scout's 16 experts 8x and understating Qwen3.5-397B's 512 by 4x.
- **Activations**, MoE-aware per layer:
  - MoE layers: `(seq / CP) * micro * (d * 12 + topk * expert_ffn * 2 * ACT_BYTES_PER_ELEMENT)`
  - Dense layers: `(seq / CP) * micro * d * 12`
  - Both multiplied by `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER` (0.45), **and by the recompute scale**. `resolve_recompute_scale` derives the recomputed fraction as `RECOMPUTE_NUM_LAYERS / (layers_per_gpu / VP)` — not from `RECOMPUTE_NUM_LAYERS` alone — and a recomputed block stores only its own input at `RECOMPUTE_BOUNDARY_BYTES_PER_ELEMENT` (2). The scale is exactly 1.0 by default. Validated on the controlled pair (jobs 1236/1238), whose recipe recomputes 50%: that configuration's activation estimate goes **23.354 → 13.623 GB**, flipping it from exceeding the real run's 19.635 GB non-weight budget to fitting inside.
  - Not divided by TP. This is deliberately conservative.
  - **`seq` is divided by CP**, because context parallelism shards the sequence. That one division covers every sequence-dependent activation term. Omitting it leaves CP shrinking only `DP`, which *raises* reported memory as CP rises — the wrong direction. Against the measured dgxc job 179 (BF16 pretrain, TP2/PP4/**CP2**, 64 GPUs, seq 8192, 108.08 GB reserved) the divisor moves the ZeRO-1 prediction from **1.49x to 1.17x** of measured, and ZeRO-2 — the closer analogue of a NeMo distributed optimizer — to **1.01x**.
  - **Both expressions are precision-invariant.** The expert intermediate takes `ACT_BYTES_PER_ELEMENT` (2), not `PARAM_BYTES`. An earlier version of this file called the dense/expert disagreement a known open question; it is **resolved**, and resolved *against* the intuitive answer — scaling the dense term down by the FP8 GEMM-input share was measured and refuted, moving the predicted FP8/BF16 total ratio from 0.9028 to 0.8528 against a measured **1.0873**.
- **Output logits**: `live_tokens * vocab * LOGITS_BYTES_PER_ELEMENT` (8), a **separate term from activations**, which are per-layer in `d` and never touch `vocab` (128,256–248,320 here). 8 bytes/element covers a bf16 logits tensor plus the fp32 softmax and its gradient.

  Under the default fused-CE assumption `live_tokens` caps at `LOGITS_CHUNK_TOKENS` (1024), because a fused or chunked cross-entropy walks the flattened `micro * seq_len` rows in chunks and keeps only one live. **Above the chunk the term is flat in micro:**

  | `micro` | fused | unfused |
  |---|---|---|
  | 1 | 1.05 GB | 4.20 GB |
  | 8 | 1.05 GB | 33.62 GB |
  | 16 | 1.05 GB | 67.24 GB |

  Measured over all 10,560 variant × hardware × ZeRO × micro combinations, 3,880 configurations fit fused against 3,252 unfused — the assumption is worth **628 configurations**, so state it when reporting. `LOGITS_CHUNK_TOKENS` is an assumption, not a measured value; real implementations size the chunk from available memory. Set `LOGITS_FUSED_CE = False` for a plain `nn.CrossEntropyLoss` over a materialised tensor. The LM head **weight** gradient is not double-counted — the head's parameters are in `model_mem` already.
- **VP activation overhead**, when `PP > 1` and `VP > 1`: `(PP - 1) * VP * layers_per_virtual_chunk * avg_act_per_layer`, for in-flight micro-batches during interleaved 1F1B warmup. These do **not** take `SELECTIVE_ACT_CHECKPOINTING_MULTIPLIER`, but they **do** take the recompute scale — an earlier version of this file called them "full, non-checkpointed" activations, which became wrong once recomputation was modelled: an in-flight chunk is recomputed on the same schedule as any other. It matters because this term **dominates**: on `Llama-3.1-70B` at PP=2/VP=20 it is **16.106 of the 23.354 GB** total, so leaving it unreduced was the single largest activation error for any recomputing recipe.
- **Communication buffers**: `NCCL_MEM_BUF` (6 GB) + `NCCL_EP_SCALING_FACTOR * (EP - 1)` + MoE dispatch + PP send/recv.
  - MoE dispatch: `moe_layers_per_gpu * 2 * micro * seq * d * PARAM_STORAGE_BYTES * (topk / experts_per_gpu)`
  - PP buffers: `2 * micro * seq * d * PARAM_STORAGE_BYTES`, one send plus one receive for a single micro-batch activation.
- **Fragmentation**: the raw total is multiplied by `FRAGMENTATION_FACTOR` (1.24), an empirical PyTorch caching-allocator overhead calibrated against Qwen3-235B on 64 B300 GPUs.

The phase reports the minimum ZeRO stage and maximum micro batch size that fit within `mem_gb * GPU_MEM_UTILIZATION_THRESHOLD` (0.92).

Static estimates of this kind have historically run low against measured peak reserved memory. Treat a result near the ceiling as not fitting.

## Phase 2: Batch configuration

Sweeps `MICROS` (`[1, 2, 4, 8, 16]`) against `GRAD_ACCUM_VALUES`, with
`global_batch = micro * accum * DP` and `tokens_per_batch = global_batch * seq_len`.
Each configuration is rated 0 (optimal) to 3 (poor) purely on its implied step
count, `TOTAL_TOKENS / tokens_per_batch`.

**It does NOT sweep against Phase 1's memory fit.** That is the single most
important thing to know about this phase. It has no memory model of its own, so it
will rate a configuration "reasonable" at a micro batch Phase 1 already found does
not fit. Pass Phase 1's largest viable micro as `max_viable_micro` and each result
carries `fits_in_memory`; omit it and the field is `None`, claiming nothing. It is
a **report, not a filter** — the priority is unchanged either way.

The scale of it, measured on a full run: **304** (variant, hardware) pairs have a
well-rated configuration that cannot be allocated, and for **216 of those (71%)**
*no* well-rated configuration fits at all. For example `Llama-3.1-8B | 64 p5` rates
exactly one configuration, needing micro=16, where Phase 1 found max viable 4.
Treat Phase 2's output as a shortlist to cross-check against Phase 1, never as a
memory verdict.

**Step assessment is RELATIVE, not banded.** The absolute step bands and the
`TOKENS_PER_BATCH` target check are both **removed**. For each (variant, hardware)
pair the smallest and largest step counts are outliers (priority 3) and the rest
are reasonable (priority 1); **priority 0 and 2 no longer occur**, and
`assessment` reads `reasonable`, `outlier (fewest steps)` or
`outlier (most steps)`.

Why the bands went: the target arm was unreachable, because
`tokens_per_batch = dp * micro * accum * seq_len` with powers-of-two micros and
accums cannot land inside a ±2% window on an arbitrary absolute number — **zero of
14,080** configurations hit it — and the remaining bands condemned **80.8%** of
configurations as "too many steps" while telling the reader nothing. Stated as a
cross-tab so the cost is visible with the benefit: **10,076 configurations moved
from "too many steps" to "reasonable" with nothing about them changed** (reasonable
7.1% → 87.8%).

Two properties worth knowing:
- Outliers are dropped **by value**, so every configuration tied at an extreme is
  marked. Step counts are degenerate — `micro * accum` repeats products, so a
  pair's 17-20 configurations collapse onto only **6-8 distinct** step values.
- Fewer than 3 distinct values leaves nothing to trim, so that pair stays wholly
  reasonable.

The verdict depends on the whole pair, so it cannot be assigned while sweeping:
`analyze_batch_configuration` returns `assessment=None`/`priority=None` and
`classify_step_outliers` fills them in. `generate_batch_recommendations` **raises**
on an unclassified configuration rather than silently matching nothing.

**One absolute floor sits alongside it**, because a relative rule structurally
cannot flag a pair whose *entire* range is absurd — within such a pair the middle
values genuinely are the least extreme, so nothing can object.
`Llama-3.1-8B | 1 g5` spans 28.6M to 3.66B steps and the relative rule calls its
middle band reasonable.

`compute_absurd_step_ceiling(TOTAL_TOKENS)` = `TOTAL_TOKENS / TOKENS_PER_BATCH` =
**3,750,000 steps**, and each configuration carries `steps_absurd`. The reference
is the **smallest batch in published practice**, not a quality target: `4e6` is
Llama 3 405B's ramp-start batch, the smallest it used at any point before doubling
to 8M then 16M (arXiv:2407.21783 §3.4.1), and DeepSeek-V3's smallest was larger
still at 3072 × 4096 = 12.6M (arXiv:2412.19437 §4.2). It is **3.1× looser than
`MAX_STEPS`**, so it is a credibility floor rather than a reinstated band, and it
is one-sided — no too-few-steps counterpart, since the derived cap already
guarantees `steps > MIN_STEPS`.

**The two axes disagree, and that is the point.** `reasonable` means least extreme
*within its pair* and says nothing about credibility: **7,260 CSV rows read
priority 1 while also being `steps_absurd`**. Filtering on priority alone keeps
them.

The pair-level line is deliberately **not** "any absurd configuration in the pair".
Every grid contains `micro=1, accum=1`, the smallest expressible batch and so the
most steps, which makes at least one absurd configuration nearly structural at
**572 of 704 pairs (81%)** — warning there would reproduce the exact failure of the
bands. It fires only when the pair's **largest reasonable batch** is itself absurd:
**132** pairs escalate (every configuration absurd), **88** warn, **484** stay
silent. The 132 are exactly the six small-GPU presets across all 22 variants.

**Now without code consumers**, flagged in place rather than deleted: the four
`LOWER_`/`UPPER_BOUND_*` constants and `MARGIN`. Widening `MARGIN` would **not**
revive the removed check, since the product still cannot land on an arbitrary
absolute target.

**The batch cap is derived, not the `600e6` constant** — see `configuration.md`.

## Phase 3: Training time

Sweeps `GRAD_ACCUM_VALUES` (`[1, 2, 4, 8]`), deriving global batch and step count from each. Sequence length is per variant, falling back to `SEQ_LEN`. The `1` arm means no accumulation, which is valid and often optimal at large DP width; it was previously absent so that arm was never evaluated.

`peak_tflops` is selected by precision — `peak_tflops_bf16` under BF16, `peak_tflops_fp8` under FP8. Earlier versions read the BF16 field unconditionally and no preset defined an FP8 figure, so an FP8 run halved its memory while computing time at BF16 throughput, **overstating FP8 training time by roughly 2x**. Both halves are now fixed: throughput selects on precision, and the memory halving was removed as wrong in sign. **`PRECISION` is a throughput-only setting** and changes no memory term. `g5` carries `0`, because A10G is Ampere with no FP8 tensor cores, and an FP8 run there is refused rather than estimated.

**FLOPs** — two terms, not just `6ND`:

```
linear_flops    = 6 * TOTAL_TOKENS * active_params_B * 1e9
attention_flops = ATTENTION_FLOPS_FACTOR * layers * seq_len * TOTAL_TOKENS * attn_flops_dim
total_flops     = linear_flops + attention_flops
```

The factor 6 is one forward plus two backward passes. `ATTENTION_FLOPS_FACTOR` is 12, covering the QK^T and softmax·V matmuls across forward and backward that `6ND` omits. Per sequence that cost is `factor * layers * seq_len² * W`; across `TOTAL_TOKENS / seq_len` sequences it reduces to the form above, which is **linear in `seq_len`**. So at a fixed token budget a 32K-context run costs more attention FLOPs than a 4K one. The term is skipped entirely if any of `layers`, the width, `seq_len` or the factor is zero.

**`attn_flops_dim` is not `d`.** It is the query-projection width from `compute_attention_flops_dim(variant)`:

```
W = q_heads * (qk_head_dim + v_head_dim) / 2
```

Both matmuls are sized by the head widths rather than by the model's hidden dimension: forward costs `2 * s² * q_heads * qk_head_dim` for QK^T plus `2 * s² * q_heads * v_head_dim` for softmax·V, and `ATTENTION_FLOPS_FACTOR = 12` is stated against a single width. `W` reduces to **exactly `d`** whenever `qk_head_dim == v_head_dim == d / q_heads`, so the **12 conforming** catalog variants are unaffected. It differs for the other **10**, from two independent causes:

| cause | variants | understatement using `d` |
|---|---|---|
| stored `head_dim` ≠ `d / q_heads` | Qwen3-32B, Qwen3-30B-A3B, Qwen3-235B-A22B, Qwen3.5-27B, Qwen3.5-35B-A3B, Qwen3.5-122B-A10B, Qwen3.5-397B-A17B | 1.20x – 2.67x |
| **MLA** (qk and v widths differ from each other too) | DeepSeek-V3, V3.1, V3.2 | 2.86x |

For DeepSeek that is `qk_nope_head_dim 128 + qk_rope_head_dim 64 = 192` and `v_head_dim 128` over 128 heads against `d = 7168`, giving `W = 20480`. Using `d` understated those variants' `est_months` by up to **18.4%**.

**GQA is not a cause.** Both matmuls run over `q_heads` regardless of `kv_heads`, because each query head attends with its own (s, s) score matrix — so a GQA model with `head_dim == d / q_heads` is a no-op here.

A variant carrying no head dims falls back to `d / q_heads`, reproducing the old `d` behaviour. That is a **known gap, not a verified value**: it applies to DeepSeek-V4, whose config is unpublished, so its attention FLOPs are very likely understated by about its siblings' factor. Do not close it by copying V3's numbers across.

The **communication models keep `d`**, because they move d-wide activation vectors. Only this FLOPs term uses the query-projection width.

`layers` here is the count of **full-attention** layers, not all layers. A hybrid model declaring `full_attention_interval: N` contributes `layers // N` (Qwen3.5: 16 of 64), because its other layers use linear attention, which is O(seq_len) and is already covered by the `6ND` term. A variant without that field contributes every layer.

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

### The two relative outputs, `rel_speed` and `efficiency_rating`

Both are relative to the variant's **whole set** of configurations, so neither can be computed while sweeping one config at a time. `calculate_training_metrics` returns `None` for both and `classify_platform_efficiency` fills them in; `generate_training_recommendations` raises on an unclassified config rather than formatting `None`.

```
rel_speed = slowest_est_months / est_months     # for that variant
position  = log(rel_speed) / log(slowest / fastest)
```

`rel_speed` is **how many times faster than the slowest configuration for that variant**, so it is always `>= 1.0` and exactly `1.0` for the slowest config itself. The fastest config's `rel_speed` is therefore the variant's full spread, which ranges from **32.5x** (`Llama-3.1-405B`) to **20,642x** (`Llama-3.2-3B`). Quote it with its anchor named; a bare "20,642x" says nothing.

`efficiency_rating` bands that `position` at **0.80 / 0.60 / 0.40** for ratings 0/1/2, else 3. Empty means never classified, which is **distinct from rating 3** — 3 is the worst real rating.

Two reasons it is a position and not a threshold on `rel_speed`:

- **Comparability.** Per-variant spread differs by 635x across the catalog, so a fixed multiplier means something different for every variant: against a slowest anchor, `rel_speed >= 2.0` is the top few percent of a 32x spread and the bottom few percent of a 20,642x one.
- **Dead bands.** A slowest anchor makes `rel_speed >= 1.0` by construction, so multiplier bands of the form `>= 2.0 / >= 1.5 / >= 1.0 / else` leave the last one **unreachable** — measured at 0 of 5,084 rows, with 97.8% collapsing into rating 0.

The log scale matters because `est_months` within a variant is spread geometrically as GPU counts double; a linear position puts nearly every config against the fast end. Measured over 5,084 rows the bands land at 26.6% / 36.1% / 21.0% / 16.3%.

Because the rating is a within-variant position, it ranks that variant's alternatives and is **not a cross-variant sort key** — sort on `est_months` for absolute comparisons.

## Phase 4: ZeRO communication

Both use `DP = gpus / (TP * PP * CP)` and the hardware's `inter_node_bw_gb`.

Each collective moves **one parameter replica per rank**, and the volume does **not** shrink with DP.

**ZeRO-2 reduce-scatter** — the gradients, one replica:

```
rs_volume_mb = model_mem_gb * (DP - 1) / DP * 1000
rs_time_ms   = rs_volume_mb / inter_node_bw_gb
```

**ZeRO-1 all-gather** — the updated **parameters**, one replica, not the fp32 optimizer state:

```
ag_volume_mb = model_mem_gb * (DP - 1) / DP * 1000
ag_time_ms   = ag_volume_mb / inter_node_bw_gb
```

ZeRO-1 and ZeRO-2 move the **same** volume per collective. ZeRO-1 is rated against lower thresholds because it communicates once per optimizer step rather than per micro-batch, not because it moves fewer bytes. Overhead ratings are heuristics: ZeRO-2 bands at 2 / 5 / 10 ms, ZeRO-1 at 1.5 / 3 / 7 ms.

> **Earlier versions of this file were wrong here**, and the engine no longer behaves as they described. They gave ZeRO-2 as `model_mem_gb * 2 / DP` and ZeRO-1 as `model_mem_gb * 3 / DP`, claiming the 3 derived from `OPTIM_BYTES / PARAM_BYTES`.
>
> The ZeRO paper ([arXiv:1910.02054v3](https://arxiv.org/abs/1910.02054), sec 7.1 and 7.2.1) states that reduce-scatter and all-gather each move a *total data movement of Psi elements*, so baseline DP incurs 2Psi, and that ZeRO's scatter-reduce plus all-gather total *"Psi+Psi=2Psi, exactly the same as the baseline DP"*. The volume is therefore **independent of GPU count**: a ring collective moves `(DP-1)/DP` of a replica per rank, approaching 1 as DP grows rather than falling.
>
> The `/DP` divisor understated ZeRO-1 by 21x and ZeRO-2 by 31x at DP=64. Do not reintroduce it, and do not repeat the related claims that ZeRO-1 has "67% less communication" or that raising DP lowers per-GPU volume.

Per-stage totals per step, from the paper:

| Stage | Total | vs baseline DP |
|---|---|---|
| baseline DP | 2Psi | 1.0x |
| ZeRO-1 | 2Psi | 1.0x |
| ZeRO-2 | 2Psi | 1.0x |
| ZeRO-3 | 3Psi | **1.5x** |

## Phase 4.2: ZeRO-3 communication

ZeRO-3 is the only stage that pays **more** than baseline data parallelism. Paper sec 7.2.2 decomposes its 3Psi, and the phase reports the three collectives separately rather than as one lump:

| Collective | Volume |
|---|---|
| forward parameter all-gather | Psi |
| backward parameter all-gather | Psi |
| gradient reduce-scatter | Psi |
| **total** | **3Psi, 1.5x baseline at every DP** |

Reporting them apart makes the extra 1.5x attributable: it is the two parameter all-gathers, not a costlier reduce-scatter. `baseline_ratio` is emitted so the figure is checkable.

A widely-cited secondary handbook gives Stage 3 as 6Psi (3x). That is **wrong**; the paper says 3Psi.

## Phase 5: MoE all-to-all

For MoE models, tokens route to expert-holding GPUs via all-to-all. An MoE layer performs **two** all-to-alls per direction — a **dispatch** sending tokens to their top-k experts and a **combine** returning the expert outputs — so four per micro-batch once the backward pass is counted. `FWD_BWD_ROUTING_BUFF_PASSES` is 2, for forward and backward. EP ranks inside one node use NVLink; across nodes, EFA. Dense models report zero.

Earlier versions counted one all-to-all per direction, understating MoE routing volume by **2x**: 536.9 MB against 1073.7 MB per layer at 4K tokens, topk=8, d=4096.

`bw_utilization_pct` is **not** achieved bandwidth. Time is computed as `bytes / effective_a2a_bw`, so recovering `bytes / time` returns that same bandwidth and dividing by peak restates `(EP-1)/EP` — it reports 87.5% at EP=8 for every model, platform and batch size. Read it as the modelled ring efficiency.

## Phase 6: PP SendRecv

Zero when `PP = 1`.

```
activation_size_bytes = micro * seq_len * d * dtype_bytes
sends_per_microbatch  = 2 * (PP * VP - 1)   if VP > 1   # interleaved
                      = 2 * (PP - 1)        if VP == 1  # standard
total_sends_per_step  = sends_per_microbatch * num_microbatches
pipeline_bubble_pct   = (PP - 1) / (num_microbatches + PP - 1) * 100          if VP == 1
                      = (PP - 1) / (VP * num_microbatches + PP - 1) * 100     if VP > 1
exposed_sends         = 2 * (PP - 1)        # warmup fwd + cooldown bwd
estimated_pp_time_ms  = exposed_sends * time_per_send_us / 1000
```

The bubble is the standard GPipe/1F1B form, which includes the fill/drain term. Earlier versions gave `(PP - 1) / (num_microbatches * VP)`, a large-`m` approximation that returns **impossible values** below `m < PP` — 175% at PP=8/m=4, 187.5% at PP=16/m=8. A bubble is a fraction of idle time and cannot exceed 100%, and it feeds the Phase 3 estimate directly as a multiplier, so those values inflated training time. The bounded form gives 63.6% and 65.2%.

**Communication type** is decided by `is_inter_node = PP > gpus_per_node`, where `gpus_per_node` is derived from the hardware entry as `gpus / nodes`, falling back to 8 when `nodes` is missing or zero. On a GB200 NVL72 UltraServer all 72 GPUs share one NVLink domain, so `gpus_per_node` is 72 and `PP` up to 72 stays intra-node, where on an 8-per-node platform `PP > 8` would already cross to EFA.

**Effective bandwidth:**

- Intra-node: `intra_node_bw_gbps` directly, for example 1800 GB/s on B300 and GB200, giving roughly 18 µs per send.
- Inter-node: `inter_node_bw_gb * EFA_PP_EFFICIENCY`, where `EFA_PP_EFFICIENCY` is 0.016 (in `utils/phase6_pp_comm.py`), because EFA point-to-point engages only one or two of the available NICs. On a 400 GB/s link that is roughly 3.2 GB/s, giving about 5.2 ms per send.

That three-orders-of-magnitude gap between intra-node and inter-node pipeline sends is why `PP` crossing the node boundary is expensive, and why `PP <= 8` is worth preserving where memory allows.

Only warmup and cooldown sends are serialized; the rest overlap with compute, which is why `exposed_sends` is `2 * (PP - 1)` rather than the full total.
