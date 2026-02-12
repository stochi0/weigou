# weigou

<p align="center">
  <svg width="140" height="140" viewBox="0 0 140 140" xmlns="http://www.w3.org/2000/svg" role="img" aria-labelledby="title desc">
    <title id="title">Weigou logo</title>
    <desc id="desc">Minimal logo representing distributed parallelism</desc>
    <defs>
      <linearGradient id="wg-gradient" x1="0%" y1="0%" x2="100%" y2="100%">
        <stop offset="0%" stop-color="#5B8DEF"/>
        <stop offset="100%" stop-color="#7F53AC"/>
      </linearGradient>
    </defs>
    <!-- Outer ring -->
    <circle cx="70" cy="70" r="56" fill="none" stroke="url(#wg-gradient)" stroke-width="6"/>
    <!-- Axes lines (4D parallelism vibe) -->
    <line x1="18" y1="70" x2="122" y2="70" stroke="#A5B4FC" stroke-width="2.5" stroke-linecap="round" />
    <line x1="70" y1="18" x2="70" y2="122" stroke="#A5B4FC" stroke-width="2.5" stroke-linecap="round" />
    <!-- Four parallelism nodes -->
    <circle cx="35" cy="70" r="8" fill="#111827" stroke="#60A5FA" stroke-width="3"/>
    <circle cx="105" cy="70" r="8" fill="#111827" stroke="#60A5FA" stroke-width="3"/>
    <circle cx="70" cy="35" r="8" fill="#111827" stroke="#A855F7" stroke-width="3"/>
    <circle cx="70" cy="105" r="8" fill="#111827" stroke="#A855F7" stroke-width="3"/>
    <!-- Center core -->
    <circle cx="70" cy="70" r="10" fill="#111827" stroke="url(#wg-gradient)" stroke-width="4"/>
  </svg>
</p>

**Minimal distributed training for LLaMA-like models.** No bloat. Just the essentials.

Weigou strips away the complexity and gives you a lean framework with full 4D parallelism and HuggingFace compatibility. Inspired by Pictron and Megatron-LM!

---

## Features

| Parallelism | Description |
|------------|-------------|
| **TP** | Tensor parallelism — split layers across GPUs |
| **CP** | Context parallelism — ring attention for long sequences |
| **PP** | Pipeline parallelism — 1F1B or AFAB schedules |
| **DP** | Data parallelism — replicate model, split data |

- Flash Attention & Triton RMSNorm (Linux)
- HuggingFace models & datasets
- SafeTensors checkpointing
- SLURM job management
- WandB logging

---

## Quick Start

```bash
# Install (uv)
uv pip install -e .

# Create config
weigou-config --exp_name my_run --out_dir runs

# Train (single node, 1 GPU)
torchrun --nproc_per_node=1 scripts/train.py --config runs/my_run/config.json

# Train (multi-GPU: 2 DP)
torchrun --nproc_per_node=2 scripts/train.py --config runs/my_run/config.json
```

---

## Config

Create experiments with `weigou-config`:

```bash
weigou-config --exp_name tp2_pp2 \
  --tp 2 --pp 2 \
  --model_name HuggingFaceTB/SmolLM-360M-Instruct \
  --seq_len 2048 --mbs 8
```

Parallelism must satisfy: `world_size = tp × cp × pp × dp`

---

## CLI

| Command | Purpose |
|---------|---------|
| `weigou-train` | Run training |
| `weigou-config` | Generate experiment configs |
| `weigou-slurm` | Submit & manage SLURM jobs |
| `weigou-metrics` | Parse training metrics |

---

## Requirements

- Python ≥ 3.11
- PyTorch
- CUDA (for Flash Attention; CPU mode supported for debugging)
- `HF_TOKEN` for HuggingFace model access

