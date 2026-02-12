"""Create experiment config files."""
import os
import shutil
import argparse
from typing import Optional

from transformers import AutoConfig

from weigou.config import (
    WeigouConfig,
    EnvironmentConfig,
    DistributedConfig,
    ModelConfig,
    DatasetConfig,
    TrainingConfig,
    LoggingConfig,
    CheckpointConfig,
)
from weigou.utils import download_model


def create_config(
    out_dir: str,
    exp_name: str,
    tp: int = 1,
    cp: int = 1,
    dp: int = 1,
    pp: int = 1,
    pp_engine: str = "1f1b",
    model_name: str = "HuggingFaceTB/SmolLM-360M-Instruct",
    num_hidden_layers: Optional[int] = None,
    num_attention_heads: Optional[int] = None,
    num_key_value_heads: Optional[int] = None,
    grad_acc_steps: int = 1,
    mbs: int = 1,
    seq_len: int = 1024,
    subset_name: Optional[str] = None,
    use_wandb: bool = False,
    use_cpu: bool = False,
    use_fused_adam: bool = False,
    hf_token: Optional[str] = None,
) -> None:
    run_path = os.path.join(out_dir, exp_name)
    os.makedirs(run_path, exist_ok=True)

    hf_config = AutoConfig.from_pretrained(model_name, token=hf_token)
    num_layers = num_hidden_layers or hf_config.num_hidden_layers
    num_heads = num_attention_heads or hf_config.num_attention_heads
    num_kv_heads = num_key_value_heads or hf_config.num_key_value_heads

    config = WeigouConfig(
        environment=EnvironmentConfig(
            HF_TOKEN=hf_token,
            FLASH_ATTEN="0" if use_cpu else "1",
        ),
        distributed=DistributedConfig(
            tp_size=tp,
            cp_size=cp,
            pp_size=pp,
            dp_size=dp,
            pp_engine=pp_engine,
            use_cpu=use_cpu,
        ),
        model=ModelConfig(
            name=model_name,
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            num_key_value_heads=num_kv_heads,
            use_fused_adam=use_fused_adam,
        ),
        dataset=DatasetConfig(
            name="roneneldan/TinyStories",
            subset_name=subset_name,
        ),
        training=TrainingConfig(
            seq_length=seq_len,
            micro_batch_size=mbs,
            gradient_accumulation_steps=grad_acc_steps,
            learning_rate=3e-4,
            max_tokens=0,
            total_train_steps=200,
            warmup_steps=10,
            seed=42,
            num_samples=400000,
        ),
        logging=LoggingConfig(
            use_wandb=use_wandb,
            run_name=exp_name,
        ),
        checkpoint=CheckpointConfig(
            save_dir=run_path,
            save_frequency=300,
            load_path="",
        ),
    )

    if os.path.exists(run_path):
        shutil.rmtree(run_path)
    os.makedirs(run_path)
    config.save(os.path.join(run_path, "config.json"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", type=str, default="runs", help="Output directory")
    parser.add_argument("--exp_name", type=str, required=True, help="Experiment name")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallelism")
    parser.add_argument("--cp", type=int, default=1, help="Context parallelism")
    parser.add_argument("--dp", type=int, default=1, help="Data parallelism")
    parser.add_argument("--pp", type=int, default=1, help="Pipeline parallelism")
    parser.add_argument("--pp_engine", type=str, default="1f1b", help="Pipeline engine")
    parser.add_argument("--model_name", type=str, default="HuggingFaceTB/SmolLM-360M-Instruct")
    parser.add_argument("--num_hidden_layers", type=int, default=None)
    parser.add_argument("--num_attention_heads", type=int, default=None)
    parser.add_argument("--num_key_value_heads", type=int, default=None)
    parser.add_argument("--grad_acc_steps", type=int, default=1)
    parser.add_argument("--mbs", type=int, default=1, help="Micro batch size")
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--subset_name", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--use_fused_adam", action="store_true")
    parser.add_argument("--hf_token", type=str, default=None)

    args = parser.parse_args()
    create_config(
        out_dir=args.out_dir,
        exp_name=args.exp_name,
        tp=args.tp,
        cp=args.cp,
        dp=args.dp,
        pp=args.pp,
        pp_engine=args.pp_engine,
        model_name=args.model_name,
        num_hidden_layers=args.num_hidden_layers,
        num_attention_heads=args.num_attention_heads,
        num_key_value_heads=args.num_key_value_heads,
        grad_acc_steps=args.grad_acc_steps,
        mbs=args.mbs,
        seq_len=args.seq_len,
        subset_name=args.subset_name,
        use_wandb=args.use_wandb,
        use_cpu=args.use_cpu,
        use_fused_adam=args.use_fused_adam,
        hf_token=args.hf_token,
    )
    print("Config created.")

    from weigou.utils import MODELS_DIR

    download_model(args.model_name, args.hf_token, MODELS_DIR)
    print("Model downloaded.")


if __name__ == "__main__":
    main()
