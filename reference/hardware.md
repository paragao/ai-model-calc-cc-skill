# Hardware catalog

## Built-in presets

32 presets ship in `configuration/hardware_config.py` — each of the 8 platforms below at each of its 4 node counts. Select one with `--hw-name "<nodes> <platform>"`, for example `--hw-name "128 p5en"`.

| Platform | GPUs/node | Mem/GPU | Inter-node GB/s | Intra-node GB/s | Peak BF16 TFLOPS | Peak FP8 TFLOPS | Node counts |
|---|---|---|---|---|---|---|---|
| p5 | 8 | 80 GB | 400 | 900 | 989 | 1979 | 64,128,256,512 |
| p5en | 8 | 141 GB | 400 | 900 | 989 | 1979 | 64,128,256,512 |
| p6-b200 | 8 | 180 GB | 400 | 1800 | 2250 | 4500 | 64,128,256,512 |
| p6-b300 | 8 | 268 GB | 800 | 1800 | 2250 | 4500 | 64,128,256,512 |
| p6e-gb200 | **72** | 186 GB | 3600 | 1800 | 2500 | 5000 | 1,2,4,8 |
| g5 | 8 | 24 GB | 12.5 | 64 | 35 | **0 — no FP8** | 1,4,8,16 |
| g6 | 8 | 24 GB | 12.5 | 64 | 121 | 242 | 1,4,8,16 |
| g6e | 8 | 48 GB | 50 | 64 | 366 | 733 | 1,4,8,16 |

Bandwidth columns are both in **GB/s** (divide an advertised Gbps figure by 8). BF16 figures are **dense, without sparsity**; intra-node figures are NVLink bidirectional totals.

Instance mapping and interconnect:

- **p5.48xlarge** — H100 SXM, 3200 Gbps EFA = 400 GB/s, NVSwitch intra-node
- **p5en.48xlarge** — H200 SXM, 3200 Gbps EFAv3 = 400 GB/s
- **p6-b200.48xlarge** — B200, 3200 Gbps EFAv4 = 400 GB/s, NVLink5
- **p6-b300.48xlarge** — B300 Ultra, 6400 Gbps EFAv4 = 800 GB/s, NVLink5. **BF16 is 2250, the same as B200** — Blackwell Ultra's 1.5x uplift is FP4 only, and AWS qualifies it as "1.5x GPU TFLOPS (at FP4, without sparsity)". Do not scale the BF16 figure by 1.5.
- **p6e-gb200 UltraServer** — GB200 NVL72. A **node is one UltraServer**: `u-p6e-gb200x72` is 18 × `p6e-gb200.36xlarge` (4 GPUs each) = 72 GPUs in **one NVLink domain**, so `gpus_per_node` is 72 and a PP of up to 72 stays on NVLink. Per UltraServer, 28800 Gbps EFAv4 = 3600 GB/s, which is 50 GB/s per GPU — the same ratio as p5.
- **g5.48xlarge** — A10G, **no NVLink**: AWS provisions G5 GPUs in PCIe pass-through mode. ENA 100 Gbps = 12.5 GB/s.
- **g6.48xlarge** / **g6e.48xlarge** — L4 / L40S, PCIe-connected with no NVLink, ENA networking. G6 100 Gbps = 12.5 GB/s, G6e 400 Gbps = 50 GB/s.

The G family is for inference and small-scale fine-tuning, not large distributed training. Its low intra-node bandwidth makes tensor parallelism expensive. These entries assume the `.48xlarge` size (8 GPUs) — smaller sizes have 1 or 4 GPUs and lower network bandwidth, so supply those via `--hardware`.

## GB200 node modelling

`p6e-gb200` **is now a built-in preset** — select `1`, `2`, `4` or `8 p6e-gb200` (72, 144, 288, 576 GPUs). Earlier revisions of this skill said it was not a preset and told you to set `nodes` so that `gpus / nodes` came out at 4. **That was wrong** and would misprice the workload badly.

GPUs per node is derived as `gpus / nodes`, and that ratio decides whether pipeline and expert traffic is priced as NVLink or EFA — roughly three orders of magnitude apart per send. For GB200 the correct ratio is **72**, not 4: `u-p6e-gb200x72` is 18 × `p6e-gb200.36xlarge` (4 GPUs each), but all 72 GPUs share **one NVLink domain**, so a PP of up to 72 stays on NVLink. Modelling a node as the 4-GPU instance pushes any `PP > 4` onto EFA, which is roughly 36× slower per send.

If you do need a custom GB200 size, set `gpus` to 72 × the UltraServer count and `nodes` to the UltraServer count. If `nodes` is missing or zero the engine falls back to 8 per node, which is wrong for this platform.

## Defining custom hardware

Write a JSON file and pass it with `--hardware FILE`. Only those entries are analyzed; the built-in presets are ignored. Start from `hardware_variant_template.json` in the repo.

Accepted shapes: a `{"hardware": [...]}` wrapper, a bare list `[{...}]`, or a single object `{...}`.

**Required** — missing any of these fails with a clear error:

| Field | Meaning |
|---|---|
| `name` | Matched case-insensitively by `--hw-name`. Keep unique within the file. |
| `gpus` | Total GPUs in the cluster |
| `nodes` | Number of nodes |
| `mem_gb` | Memory per GPU in GB |
| `inter_node_bw_gb` | Per-GPU inter-node bandwidth in GB/s, for example EFA |
| `intra_node_bw_gbps` | Intra-node link bandwidth in GB/s, for example NVLink or PCIe |
| `peak_tflops_bf16` | Per-GPU peak BF16 TFLOPS, dense without sparsity |
| `peak_tflops_fp8` | Per-GPU peak FP8 TFLOPS, dense without sparsity. Read instead of the BF16 field when `PRECISION = "FP8"`. Set `0` for a GPU with no FP8 tensor cores, such as A10G, which makes an FP8 run refuse rather than silently compute. **Omitting** the field is a different failure and gets a different message. |

**Optional:** `label` is accepted but unused. `gpus_per_node` is **not read** — see the caution above.

**Validation:** `gpus` must be divisible by `TP * PP * CP` from `configuration/project_config.py`, or the run fails.

### Example

```json
{
  "hardware": [
    {
      "name": "MyCluster-Large",
      "gpus": 512,
      "nodes": 64,
      "mem_gb": 141,
      "inter_node_bw_gb": 400,
      "intra_node_bw_gbps": 900,
      "peak_tflops_bf16": 989,
      "peak_tflops_fp8": 1979
    }
  ]
}
```

```bash
python3 ai_model_calculator.py --model Llama-3.1-70B --hardware my_cluster.json
```

### Converting bandwidth

Network specs are quoted in Gbps; the calculator wants GB/s. Divide by 8. So p5en's 3200 Gbps EFA is entered as `400`, and p6-b300's 6400 Gbps as `800`. Use the same convention for a custom cluster so results stay comparable with the presets.

Note that Phase 6 does not use the full inter-node figure for pipeline send/recv: EFA point-to-point engages only one or two of the available NICs, so the model multiplies by `EFA_PP_EFFICIENCY` (0.016, defined in `utils/phase6_pp_comm.py`), giving roughly 3.2 GB/s on a 400 GB/s link. That derating is applied for you — enter the honest nominal bandwidth in your hardware entry.
