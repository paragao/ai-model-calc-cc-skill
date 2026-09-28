# Hardware catalog

## Built-in presets

28 presets ship in `configuration/hardware_config.py` — each platform below at each of its node counts. Select one with `--hw-name "<nodes> <platform>"`, for example `--hw-name "128 p5en"`.

| Platform | GPUs/node | Mem/GPU | Inter-node GB/s | Intra-node GB/s | Peak BF16 TFLOPS | Node counts |
|---|---|---|---|---|---|---|
| p5 | 8 | 80 GB | 400 | 900 | 989 | 64,128,256,512 |
| p5en | 8 | 141 GB | 400 | 900 | 989 | 64,128,256,512 |
| p6-b200 | 8 | 180 GB | 400 | 1800 | 2250 | 64,128,256,512 |
| p6-b300 | 8 | 268 GB | 800 | 1800 | 3375 | 64,128,256,512 |
| g5 | 8 | 24 GB | 100 | 600 | 35 | 1,4,8,16 |
| g6 | 8 | 24 GB | 100 | 64 | 121 | 1,4,8,16 |
| g6e | 8 | 48 GB | 50 | 64 | 366 | 1,4,8,16 |

Instance mapping and interconnect:

- **p5.48xlarge** — H100 SXM, 3200 Gbps EFA, NVSwitch intra-node
- **p5en.48xlarge** — H200 SXM, 3200 Gbps EFAv3
- **p6-b200.48xlarge** — B200, 3200 Gbps EFAv4, NVLink5
- **p6-b300.48xlarge** — B300 Ultra, 6400 Gbps EFAv4, NVLink5
- **g5.48xlarge** — A10G, NVLink pairs only, not a full mesh, ENA networking
- **g6.48xlarge** / **g6e.48xlarge** — L4 / L40S, PCIe-connected with no NVLink, ENA networking

The G family is for inference and small-scale fine-tuning, not large distributed training. Its low intra-node bandwidth makes tensor parallelism expensive.

## Not built in

`p6e-gb200.36xlarge` (GB200 NVL, 185 GB per GPU, 2500 peak BF16 TFLOPS, **4 GPUs per node**) is **not** a preset, despite appearing in some project documentation. Supply it via `--hardware`.

Two cautions when you do:

1. The Phase 5 and Phase 6 communication models **assume 8 GPUs per node** and do not read GPU density from hardware entries. At GB200's 4 GPUs per node, the intra-node versus inter-node boundary — and therefore the NVLink-versus-EFA decision — is computed wrongly. Report that caveat alongside any GB200 result.
2. GB200 is sold as UltraServers. A single UltraServer is a 36 or 72 GPU configuration; larger counts are multiple UltraServers joined over EFA, which changes the bandwidth picture the model does not capture.

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
| `peak_tflops_bf16` | Per-GPU peak BF16 TFLOPS |

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
      "peak_tflops_bf16": 989
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
