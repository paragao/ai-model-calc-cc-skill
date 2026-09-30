---
name: ai-model-calc
description: Plan LLM training infrastructure on AWS GPU clusters. Estimates per-GPU memory across ZeRO stages, batch configuration, wall-clock training time, and ZeRO/MoE/pipeline communication overhead. Use when asked how many GPUs a model needs, how long training will take, whether a model fits in memory, or to compare parallelism or hardware choices for Llama, Qwen, DeepSeek or a custom architecture.
---

# AI Model Training Calculator

Wraps the [ai-model-calculator](https://github.com/paragao/ai-model-calculator) Python tool, which runs a six-phase analysis of LLM training infrastructure. This skill drives that tool: it gathers the inputs, invokes the CLI, and interprets the output.

The calculator is a planning instrument, not a benchmark. Its numbers are estimates calibrated against real runs; treat them as a ranking heuristic for comparing configurations rather than a prediction of measured performance.

## Prerequisites

The tool must be available locally. It needs Python 3.8 or newer and uses only the standard library.

```bash
python3 --version
ls ai-model-calculator/ai_model_calculator.py 2>/dev/null || \
  git clone https://github.com/paragao/ai-model-calculator.git
```

Clone it once and reuse it. Ask the user where it lives if a copy may already exist; do not clone over an existing checkout.

**Never edit** `ai_model_calculator.py`, `model_calculations.py`, or anything under `utils/`. Those are the engine. Supply custom models and hardware as JSON files instead (see below).

## The three questions it answers

Establish which one the user is asking before running anything. They need different inputs.

| Mode | Question | Inputs required |
|------|----------|-----------------|
| A | How many instances to finish in time X? | model, dataset size in tokens, target time |
| B | How long will training take? | model, dataset size in tokens, instance type, node count |
| C | What is the minimum to fit this model? | model, instance type |

Mode C is a pure memory-fit calculation and needs no dataset size. It reports inference, fine-tuning and full pre-training side by side.

**Always confirm model, dataset size and mode with the user before running.** Do not assume defaults.

## Running it

Always invoke `ai_model_calculator.py`. Never call `model_calculations.py` directly.

```bash
cd ai-model-calculator && python3 ai_model_calculator.py [FLAGS]
```

| Flag | Effect |
|------|--------|
| `--model NAME` | Analyze only the variant with this name. Case-insensitive, exact match. |
| `--json FILE` | Analyze only the variants defined in FILE, replacing the built-ins. |
| `--hardware FILE` | Analyze only the hardware defined in FILE, replacing the built-in presets. |
| `--hw-name NAME` | Restrict to a single hardware entry by name. |
| `--report` | Emit a concise markdown summary instead of the full output. |
| none | Analyze all 22 built-in variants against all 32 built-in hardware presets. |

Running with no flags is a large cross-product. Scope it with `--model` and `--hw-name` unless the user genuinely wants the sweep.

`--model` filters whichever variant list is in effect, and `--hw-name` whichever hardware list is in effect, so they compose with `--json` and `--hardware`.

Built-in hardware preset names take the form `"<nodes> <platform>"`:

```bash
python3 ai_model_calculator.py --model Llama-3.1-70B --hw-name "128 p5en"
```

Parallelism degrees, token budget, precision and MFU have **no CLI flags**. They live in `configuration/project_config.py` and must be edited there. See `reference/configuration.md`.

## What it produces

Console tables plus CSV/JSON exports in the working directory:
`phase1_memory_results.csv` and `.json`, `phase2_batch_results.csv`,
`phase3_training_results.csv`, `phase4_zero2_comm_results.csv`,
`phase4_1_zero1_comm_results.csv`, `phase4_2_zero3_comm_results.csv`,
`phase5_alltoall_comm_results.csv`. Phase 6 is reported inline.

`phase2_batch_results.csv` carries three columns worth reading carefully.
`fits_in_memory` is a **tri-state**: `True`, `False`, or **blank** meaning "not
assessed" because Phase 1 found no viable micro batch for that pair — blank must
not be rendered as "does not fit". `exceeds_batch_cap` is uniformly `False` in the
default export, because the pipeline filters capped rows out before writing, so it
carries information only for a caller that skips that filter.

`steps_absurd` is the **absolute** axis and is independent of `priority`, which is
purely relative to the pair. **7,260 rows read `priority` 1 ("reasonable") while
also being `steps_absurd`**, so a report filtered on priority alone keeps
configurations whose batch is below the entire published envelope. Never fold the
two together; present `steps_absurd` as its own column.

The six phases:

1. **Memory** — per-GPU breakdown across ZeRO stages, MoE-aware, including dispatch buffers, VP activation overhead and a 1.24x fragmentation factor.
2. **Batch** — micro batch size and gradient accumulation.
3. **Training time** — wall-clock with a low/high range, split into compute, pipeline bubble and exposed communication, at BF16 or FP8 throughput.
4. **ZeRO communication** — reduce-scatter (ZeRO-2) and all-gather (ZeRO-1), one parameter replica per rank each.
4.2. **ZeRO-3 communication** — two parameter all-gathers plus the gradient reduce-scatter: 3*Psi per step, 1.5x baseline DP.
5. **MoE all-to-all** — expert token routing, counting both dispatch and combine.
6. **PP SendRecv** — pipeline point-to-point cost, NVLink vs EFA, bubble percentage.

If a run fails validation, check that `PP` divides the model's `layers` evenly, that `EP` divides the expert count evenly, and that the cluster's `gpus` is divisible by `TP * PP * CP`.

## Interpreting results

Report the configuration that fits, not just the fastest number. State the ZeRO stage and micro batch size a result depends on — a faster configuration that needs ZeRO-3 is a different operational proposition from one that fits at ZeRO-1.

**Do not treat ZeRO-3 as strictly better than ZeRO-2.** It shards resident parameters, but a sharded layer cannot execute, so each module is all-gathered whole into a transient buffer before it runs. That buffer does **not** shrink with DP, so past roughly DP=32 it exceeds the resident shard it replaced, and for a small model at large DP it can cost more memory than the sharding saves. Read the `AGbuf` column, and note ZeRO-3 also pays 1.5x baseline communication where ZeRO-1 and ZeRO-2 pay 1.0x.

Start at `PP=1` and raise pipeline parallelism only when the model genuinely does not fit. A high-PP, high-VP configuration costs bubble time; it is something to justify, not a default. Whether reducing PP wins depends on whether memory has real slack, so check the Phase 1 headroom before recommending it.

Flag when a result sits close to the memory ceiling. The fragmentation factor is empirical and static estimates on this stack have run low against measured peak reserved memory.

## Known caveats

- GPUs per node is derived from each hardware entry as `gpus / nodes`, and decides whether pipeline (Phase 6) and expert (Phase 5) traffic is priced as intra-node NVLink or inter-node EFA. The two differ by roughly three orders of magnitude per send, so the `nodes` value in a custom hardware file matters a great deal — get it right. If `nodes` is absent or zero, the engine falls back to assuming 8 per node.
- Activation memory is not divided by TP, which is deliberately conservative.
- `p6e-gb200`/GB200 NVL72 **is** a built-in preset (`1`, `2`, `4`, `8 p6e-gb200`). A node is one UltraServer -- 72 GPUs in ONE NVLink domain -- so `gpus / nodes` is **72**, not 4. A PP of up to 72 stays on NVLink.
- Several built-in variants decouple `head_dim` from `d / q_heads` (the Qwen3 and Qwen3.5 MoE models use 128 or 256). When writing a custom variant, set `head_dim` explicitly — letting it default silently halves attention parameters on those architectures. See `reference/models.md`.
- The four DeepSeek variants use Multi-head Latent Attention, so their stored attention parameters are not reproducible from the generic Q/K/V/O formula. That is expected, not an error.

## Reference files

Load these as needed rather than up front:

- `reference/models.md` — the 22 built-in variants, and how to define a custom one for `--json`.
- `reference/hardware.md` — the built-in presets, and how to define a custom cluster for `--hardware`.
- `reference/configuration.md` — `project_config.py` and `advanced_config.py` parameters.
- `reference/formulas.md` — the formulas behind each phase, for explaining or sanity-checking a result.
