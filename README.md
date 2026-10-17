# Weigou

Distributed training experiments for LLaMA-style language models.
The implementation separates tensor, context, pipeline, and data parallelism
so their behavior can be studied together.

## What is included

| Parallelism | Operation |
| --- | --- |
| Tensor (TP) | Split model layers across processes |
| Context (CP) | Partition sequences and use ring attention |
| Pipeline (PP) | Schedule model stages with 1F1B or AFAB execution |
| Data (DP) | Replicate the model and distribute training data |

The repository also includes Hugging Face model and dataset loading,
SafeTensors checkpoints, SLURM helpers, and optional Weights & Biases logging.
Flash Attention and Triton dependencies apply on Linux.

## Installation

Use Python 3.11 or later and [uv](https://docs.astral.sh/uv/):

```sh
git clone https://github.com/stochi0/weigou.git
cd weigou
uv sync --locked
```

CUDA is required for the GPU paths. CPU mode is available for debugging.
Set `HF_TOKEN` when the chosen Hugging Face model requires authentication;
do not commit access tokens or generated configs containing them.

## First run

Create a configuration for one process, then launch training:

```sh
uv run weigou-config --exp_name first-run --out_dir runs
uv run torchrun --nproc_per_node=1 scripts/train.py \
  --config runs/first-run/config.json
```

Configuration generation reads the selected model's Hugging Face configuration.
The default model is `HuggingFaceTB/SmolLM-360M-Instruct`; the generated training
configuration uses TinyStories. Runs may download model assets and data.

## Distributed runs

For a two-process data-parallel run, create a matching configuration:

```sh
uv run weigou-config --exp_name dp2 --out_dir runs --dp 2
uv run torchrun --nproc_per_node=2 scripts/train.py \
  --config runs/dp2/config.json
```

The configuration must satisfy:

- `world_size = tp × cp × pp × dp`.
- Sequence length is divisible by `cp`.
- Attention-head and key/value-head counts are divisible by `tp`.

Use `uv run weigou-config --help` for model dimensions, parallelism, sequence
length, microbatch size, and CPU/debug options. Configuration generation replaces
the selected experiment directory; use a new `--exp_name` to preserve an old run.

## Checks and tools

Tests are organized by [parallelism mode](tests/). GPU and distributed tests
require the corresponding hardware and process launch; test presence alone does
not establish correctness for every combination or a throughput advantage.

```sh
uv sync --locked --extra dev
uv run pytest tests/test_data_unit.py
```

| Command | Purpose |
| --- | --- |
| `weigou-train` | Launch the training entrypoint |
| `weigou-config` | Generate an experiment configuration |
| `weigou-slurm` | Prepare and manage SLURM jobs |
| `weigou-metrics` | Read training metrics |

## Repository guide and references

Model, data, training, and checkpoint code live in [weigou/](weigou/).
Parallel operations live in [weigou/parallel/](weigou/parallel/), and job templates
are under [slurm/](slurm/).

The project takes inspiration from
[Picotron](https://github.com/huggingface/picotron) and
[Megatron-LM](https://github.com/NVIDIA/Megatron-LM).
