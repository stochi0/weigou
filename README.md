# weigou

**A small distributed training for LLaMA-like models.** No bloat. Just the essentials.

Weigou strips away the complexity and gives you a lean framework with full 4D parallelism and HuggingFace compatibility. Inspired by Pictron and Megatron-LM!

![Weigou Mascot](https://www.svgrepo.com/show/380278/cybernetic-cyborg-halloween-machine-robot.svg)

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

# Create config (1 GPU, no parallelism)
weigou-config --exp_name my_run --out_dir runs

# Train (single node, 1 GPU)
torchrun --nproc_per_node=1 scripts/train.py --config runs/my_run/config.json

# Train (multi-GPU: 2 DP)
torchrun --nproc_per_node=2 scripts/train.py --config runs/my_run/config.json

# Train with full 4D parallelism (example: tp=2, cp=2, pp=2, dp=2 → world_size=16)
weigou-config --exp_name tp2_cp2_pp2_dp2 --out_dir runs \
  --tp 2 --cp 2 --pp 2 --dp 2 \
  --seq_len 2048 --mbs 1 \
  --model_name HuggingFaceTB/SmolLM-360M-Instruct

HF_TOKEN=... torchrun --nproc_per_node=16 scripts/train.py \
  --config runs/tp2_cp2_pp2_dp2/config.json
```

---

## Config

Create experiments with `weigou-config`:

```bash
weigou-config --exp_name tp2_cp2_pp2_dp2 \
  --tp 2 --cp 2 --pp 2 --dp 2 \
  --model_name HuggingFaceTB/SmolLM-360M-Instruct \
  --seq_len 2048 --mbs 8
```

Parallelism must satisfy: `world_size = tp × cp × pp × dp`

- `WORLD_SIZE` (total processes across all nodes) must equal `tp * cp * pp * dp`.
- `seq_len` must be divisible by `cp`.
- `num_attention_heads` and `num_key_value_heads` must be divisible by `tp` (in the underlying HF config or your overrides).

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

