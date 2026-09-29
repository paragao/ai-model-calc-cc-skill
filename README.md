# AI Model Training Calculator — Claude Code Skill

A [Claude Code Agent Skill](https://docs.claude.com/en/docs/claude-code/skills) that drives the [ai-model-calculator](https://github.com/paragao/ai-model-calculator) Python tool to plan LLM training infrastructure on AWS GPU clusters.

Ask about GPU counts, training duration, memory fit, or parallelism trade-offs and Claude will gather the inputs, run the calculator, and interpret the output.

## Installation

Clone into your personal skills directory:

```bash
git clone https://github.com/paragao/ai-model-calc-cc-skill.git ~/.claude/skills/ai-model-calc
```

Or scope it to a single project:

```bash
git clone https://github.com/paragao/ai-model-calc-cc-skill.git .claude/skills/ai-model-calc
```

Then clone the calculator itself, which the skill shells out to:

```bash
git clone https://github.com/paragao/ai-model-calculator.git
```

Python 3.8 or newer is required. The calculator uses only the standard library, so there is nothing to install.

Verify Claude picked the skill up by running `/skills` in Claude Code.

## Usage

Just ask. The skill loads automatically when the topic matches:

- "How many p5en nodes do I need to train Llama-3.1-70B on 15T tokens in two months?"
- "Will DeepSeek-V3 fit on 64 B300 GPUs?"
- "Compare Qwen3-235B-A22B on p6-b200 versus p6-b300."
- "What does raising PP from 1 to 4 cost me on a 405B model?"
- "Estimate training time for my custom 30B MoE at 32K context."

The skill will ask which of three questions you are really asking — how many instances for a target time, how long on a fixed cluster, or the minimum to fit — because each needs different inputs.

## What it covers

Six analysis phases: per-GPU memory across ZeRO stages including ZeRO-3's transient all-gather buffer, batch configuration, training time at BF16 or FP8 throughput, ZeRO-1/ZeRO-2 communication, ZeRO-3 communication at 1.5x baseline, MoE all-to-all routing, and pipeline send/recv.

22 built-in model variants across Llama 3.1/3.2/3.3, Llama 4, Qwen 2.5/3/3.5 and DeepSeek V3–V4. 32 built-in hardware presets across p5, p5en, p6-b200, p6-b300, p6e-gb200, g5, g6 and g6e. Custom models and clusters are supplied as JSON.

## Structure

```
SKILL.md                     # Entry point: prerequisites, modes, CLI, interpretation
reference/
  models.md                  # Built-in variants and custom model JSON schema
  hardware.md                # Built-in presets and custom hardware JSON schema
  configuration.md           # project_config.py and advanced_config.py parameters
  formulas.md                # Per-phase formulas, verified against engine source
```

`SKILL.md` is loaded when the skill triggers; the reference files are pulled in only as needed.

## Caveats

The calculator is a planning instrument, not a benchmark. Numbers are estimates for ranking configurations against each other, not predictions of measured throughput.

Two specifics worth knowing:

- GPUs per node is derived from each hardware entry as `gpus / nodes`, and that ratio decides whether pipeline and expert traffic is priced as NVLink or EFA — about three orders of magnitude apart per send. Set `nodes` correctly in custom hardware files.
- `p6e-gb200`/GB200 NVL72 **is** a built-in preset (`1`, `2`, `4`, `8 p6e-gb200` = 72, 144, 288, 576 GPUs). A node is one UltraServer: 72 GPUs in a single NVLink domain.

## Related

- [ai-model-calculator](https://github.com/paragao/ai-model-calculator) — the Python CLI this skill drives
- [ai-model-calc-power](https://github.com/paragao/ai-model-calc-power) — the same capability as a Kiro Power
- [ai-model-calc-skill](https://github.com/paragao/ai-model-calc-skill) — the same capability as an Amazon Quick Skill

## License

See [LICENSE](LICENSE).
