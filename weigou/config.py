from dataclasses import dataclass, field, asdict
from typing import Optional
import json
import os

@dataclass
class EnvironmentConfig:
    OMP_NUM_THREADS: str = "1"
    TOKENIZERS_PARALLELISM: str = "false"
    FLASH_ATTEN: str = "1"
    HF_TOKEN: Optional[str] = None
    
    def __post_init__(self):
        if self.HF_TOKEN is None:
            self.HF_TOKEN = os.environ.get("HF_TOKEN")

@dataclass
class DistributedConfig:
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    dp_size: int = 1
    pp_engine: str = "1f1b"
    use_cpu: bool = False
    backend: str = "nccl"

    def __post_init__(self):
        if self.use_cpu:
            self.backend = "gloo"

@dataclass
class ModelConfig:
    name: str
    num_hidden_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_key_value_heads: Optional[int] = None
    use_fused_adam: bool = False

@dataclass
class DatasetConfig:
    name: str
    subset_name: Optional[str] = None
    split: str = "train"
    num_workers: int = 8
    num_proc: int = 8

@dataclass
class TrainingConfig:
    seq_length: int
    micro_batch_size: int
    gradient_accumulation_steps: int
    learning_rate: float
    total_train_steps: int
    warmup_steps: int = 10
    seed: int = 42
    max_tokens: Optional[int] = None
    num_samples: Optional[int] = None

@dataclass
class LoggingConfig:
    use_wandb: bool = False
    run_name: str = "default_run"
    log_frequency: int = 1

@dataclass
class CheckpointConfig:
    save_dir: str
    save_frequency: int
    load_path: Optional[str] = None

@dataclass
class WeigouConfig:
    environment: EnvironmentConfig
    distributed: DistributedConfig
    model: ModelConfig
    dataset: DatasetConfig
    training: TrainingConfig
    logging: LoggingConfig
    checkpoint: CheckpointConfig

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            environment=EnvironmentConfig(**data.get("environment", {})),
            distributed=DistributedConfig(**data.get("distributed", {})),
            model=ModelConfig(**data.get("model", {})),
            dataset=DatasetConfig(**data.get("dataset", {})),
            training=TrainingConfig(**data.get("training", {})),
            logging=LoggingConfig(**data.get("logging", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {}))
        )

    def to_dict(self):
        return asdict(self)

    def save(self, path: str):
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=4)

    @classmethod
    def load(cls, path: str):
        with open(path, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)
