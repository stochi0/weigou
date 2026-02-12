import os
import time
import datetime
import torch
import torch.distributed as dist
from torch.optim import AdamW
import torch.nn.functional as F
from transformers import AutoConfig
import wandb
import inspect

import weigou.parallel.manager as pgm
from weigou.parallel import (
    setup_process_group_manager,
    apply_tensor_parallel,
    apply_context_parallel,
    PipelineParallel,
    DataParallelBucket,
    train_step_pipeline_1f1b,
    train_step_pipeline_afab
)
from weigou.utils import (
    MODELS_DIR,
    average_loss_across_dp_cp_ranks,
    download_model,
    get_mfu,
    get_num_params,
    print,
    set_all_seed,
    to_readable_format,
)
from weigou.checkpoint import (
    CheckpointManager,
    init_model_with_dematerialized_weights,
    init_model_with_materialized_weights
)
from weigou.data import MicroBatchDataLoader
from weigou.model import Llama
from weigou.config import WeigouConfig

class Trainer:
    def __init__(self, config: WeigouConfig):
        self.config = config
        self.setup_environment()
        self.setup_distributed()
        self.setup_seed()
        
    def setup_environment(self):
        os.environ["OMP_NUM_THREADS"] = str(self.config.environment.OMP_NUM_THREADS)
        os.environ["TOKENIZERS_PARALLELISM"] = self.config.environment.TOKENIZERS_PARALLELISM
        os.environ["FLASH_ATTEN"] = self.config.environment.FLASH_ATTEN
        os.environ["DEVICE"] = "cpu" if self.config.distributed.use_cpu else "cuda"
        
        if self.config.environment.HF_TOKEN:
            os.environ["HF_TOKEN"] = self.config.environment.HF_TOKEN
        elif "HF_TOKEN" not in os.environ:
             raise ValueError("HF_TOKEN is neither set in the config file nor in the environment")
             
        self.dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() and not self.config.distributed.use_cpu else torch.float32
        assert (self.dtype == torch.bfloat16 and os.getenv("FLASH_ATTEN") == "1") or os.getenv("FLASH_ATTEN") != "1", "Kernel operations requires dtype=torch.bfloat16"

    def setup_distributed(self):
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.global_rank = int(os.environ["RANK"])
        self.world_size = int(os.environ["WORLD_SIZE"])

        backend = self.config.distributed.backend
        
        assert self.config.training.seq_length % self.config.distributed.cp_size == 0, "seq_length must be divisible by cp_size for Context Parallelism"
        expected_world_size = (self.config.distributed.tp_size * 
                               self.config.distributed.pp_size * 
                               self.config.distributed.dp_size * 
                               self.config.distributed.cp_size)
        assert self.world_size == expected_world_size, f"world_size ({self.world_size}) must be equal to tp*pp*dp*cp ({expected_world_size})"

        if backend == "nccl":
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
        else:
            self.device = torch.device("cpu")

        dist.init_process_group(rank=self.global_rank, world_size=self.world_size, backend=backend, init_method=f"env://", timeout=datetime.timedelta(minutes=3))
        setup_process_group_manager(
            tp_size=self.config.distributed.tp_size,
            cp_size=self.config.distributed.cp_size,
            pp_size=self.config.distributed.pp_size,
            dp_size=self.config.distributed.dp_size
        )
        self.is_wandb_rank = (pgm.process_group_manager.tp_rank == 0 and 
                              pgm.process_group_manager.dp_rank == 0 and 
                              pgm.process_group_manager.cp_rank == 0 and 
                              pgm.process_group_manager.pp_is_last_stage)

    def setup_seed(self):
        set_all_seed(self.config.training.seed)

    def train_step(self, model, data_loader):
        acc_loss = 0.0
        requires_grad_sync = pgm.process_group_manager.cp_dp_world_size > 1
        
        for i in range(data_loader.grad_acc_steps):
            batch = next(data_loader)
            input_ids = batch["input_ids"].to(self.device)
            target_ids = batch["target_ids"].to(self.device)

            if requires_grad_sync:
                model.require_backward_grad_sync = (i == data_loader.grad_acc_steps - 1)

            outputs = model(input_ids=input_ids)

            batch_size, seq_len = input_ids.shape
            target_ids = target_ids.reshape(-1)
            outputs = outputs.view(seq_len*batch_size, -1)
            loss = F.cross_entropy(outputs, target_ids, reduction='mean') / data_loader.grad_acc_steps
            
            loss.backward()
            acc_loss += loss.item()

        return acc_loss

    def run(self):
        start_time = time.time()
        
        # DataLoader
        data_loader = MicroBatchDataLoader(
            micro_batch_size=self.config.training.micro_batch_size,
            seq_length=self.config.training.seq_length,
            dataset_name=self.config.dataset.name,
            tokenizer_name=self.config.model.name,
            grad_acc_steps=self.config.training.gradient_accumulation_steps,
            device=self.device,
            num_workers=self.config.dataset.num_workers,
            num_proc=self.config.dataset.num_proc,
            num_samples=self.config.training.num_samples,
            subset_name=self.config.dataset.subset_name,
            split=self.config.dataset.split
        )

        if pgm.process_group_manager.global_rank == 0:
            download_model(self.config.model.name, os.environ["HF_TOKEN"], MODELS_DIR)

        dist.barrier()
        print(f"init dataloader time: {time.time()-start_time:.2f}s", is_print_rank=self.is_wandb_rank)
        
        tokens_per_step = data_loader.global_batch_size * self.config.training.seq_length
        if pgm.process_group_manager.global_rank == 0:
            print("Tokens per step:", to_readable_format(tokens_per_step), is_print_rank=self.is_wandb_rank)

        # WandB
        if self.is_wandb_rank and self.config.logging.use_wandb:
            wandb.init(
                project="weigou",
                name=f"{self.config.logging.run_name}_{to_readable_format(tokens_per_step)}_{pgm.process_group_manager}",
                config=self.config.to_dict()
            )

        # Model Config
        if pgm.process_group_manager.global_rank == 0:
            print(f"rank {pgm.process_group_manager.global_rank}: Creating model config")
            model_config = AutoConfig.from_pretrained(self.config.model.name)
            if self.config.model.num_hidden_layers is not None:
                model_config.num_hidden_layers = self.config.model.num_hidden_layers
            if self.config.model.num_attention_heads is not None:
                model_config.num_attention_heads = self.config.model.num_attention_heads
            if self.config.model.num_key_value_heads is not None:
                model_config.num_key_value_heads = self.config.model.num_key_value_heads
            model_config.max_position_embeddings = self.config.training.seq_length
            objects = [model_config]
        else:
            objects = [None]

        dist.broadcast_object_list(objects, src=0, device=self.device)
        model_config = objects[0]
        print(f"rank {pgm.process_group_manager.global_rank}: Broadcasting model_config to all ranks", is_print_rank=pgm.process_group_manager.global_rank==0)
        dist.barrier()
        
        # Model Init
        print(f"rank {pgm.process_group_manager.global_rank}: Initializing model meta device", is_print_rank=self.is_wandb_rank)
        start_time = time.time()
        
        with init_model_with_dematerialized_weights():
            model = Llama(config=model_config)
            if pgm.process_group_manager.tp_world_size > 1:
                model = apply_tensor_parallel(model)
            if pgm.process_group_manager.pp_world_size > 1:
                model = PipelineParallel(model, model_config)

        model_dir = os.path.join(MODELS_DIR, self.config.model.name.replace("/", "__"))
        model = init_model_with_materialized_weights(model, model_config, save_dir=model_dir)
        
        # Load Checkpoint (to continue pre-training) - TODO: Implement logic to load from existing run if needed
        
        if pgm.process_group_manager.cp_world_size > 1:
            model = apply_context_parallel(model)

        model.to(self.dtype).to(self.device)
        
        if pgm.process_group_manager.dp_world_size > 1:
            model = DataParallelBucket(model)
        
        print(f"init model parallel time: {time.time()-start_time:.2f}s", is_print_rank=self.is_wandb_rank)
        model.train()
        num_params = get_num_params(model)
        print(f"Number of parameters: {to_readable_format(num_params)}", is_print_rank=self.is_wandb_rank)

        # Optimizer
        tensor_shapes = (data_loader.micro_batch_size, data_loader.seq_length_per_gpu, model_config.hidden_size)
        extra_args = dict()
        if self.config.model.use_fused_adam:
            fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
            use_fused = fused_available and self.device.type == 'cuda'
            extra_args = dict(fused=True) if use_fused else dict()

        optimizer = AdamW(model.parameters(), lr=self.config.training.learning_rate, **extra_args)
        checkpoint_manager = CheckpointManager()
        
        trained_tokens, step = 0, 0
        if self.config.checkpoint.load_path:
             step, trained_tokens = checkpoint_manager.load_checkpoint(model, optimizer, self.config.checkpoint.load_path)

        dist.barrier()

        # Training Loop
        while self.config.training.max_tokens is None or trained_tokens < self.config.training.max_tokens:
            step_start_time = time.time()
            optimizer.zero_grad()
            
            if pgm.process_group_manager.pp_world_size > 1:
                if self.config.distributed.pp_engine == "afab":
                    loss = train_step_pipeline_afab(model, data_loader, tensor_shapes, self.device, self.dtype)
                elif self.config.distributed.pp_engine == "1f1b":
                    loss = train_step_pipeline_1f1b(model, data_loader, tensor_shapes, self.device, self.dtype)
                else:
                    raise ValueError(f"Invalid pipeline parallel engine: {self.config.distributed.pp_engine}")
            else:
                loss = self.train_step(model, data_loader)
                
            loss = average_loss_across_dp_cp_ranks(loss, self.device)
            optimizer.step()
            
            trained_tokens += tokens_per_step
            step += 1
            
            if hasattr(model, 'reset'):
                model.reset()
                
            step_duration = time.time() - step_start_time
            tokens_per_second = tokens_per_step / step_duration
            tokens_per_second_per_gpu = tokens_per_second / self.world_size
            mfu = get_mfu(tokens_per_second_per_gpu, num_params, model_config)
            
            if self.is_wandb_rank:
                 print(
                    f"[rank {pgm.process_group_manager.global_rank}] "
                    f"Step: {step:<5d} | "
                    f"Loss: {loss:6.4f} | "
                    f"Global batch size: {to_readable_format(tokens_per_step):>7s} | "
                    f"Tokens/s: {to_readable_format(tokens_per_second):>7s} | "
                    f"Tokens/s/GPU: {to_readable_format(tokens_per_second_per_gpu):>7s} | "
                    f"Tokens: {to_readable_format(trained_tokens):>7s}{('/' + to_readable_format(self.config.training.max_tokens)) if self.config.training.max_tokens else ''} | "
                    f"MFU: {mfu:5.2f}% | "
                    f"Memory usage: {torch.cuda.memory_reserved() / 1e9:6.2f}GB",
                    is_print_rank=self.is_wandb_rank
                )
                 if self.config.logging.use_wandb:
                    wandb.log({
                        "loss": loss,
                        "tokens_per_step": tokens_per_step,
                        "tokens_per_second": tokens_per_step / step_duration,
                        "mfu": mfu,
                        "tokens_per_second_per_gpu": tokens_per_second_per_gpu,
                        "memory_usage": torch.cuda.memory_reserved() / 1e9,
                        "trained_tokens": trained_tokens
                    })

            if step % self.config.checkpoint.save_frequency == 0:
                 checkpoint_manager.save_checkpoint(model, optimizer, step, trained_tokens, self.config.checkpoint.save_dir + f"/{step}")

            if step >= self.config.training.total_train_steps:
                break
        
        if self.is_wandb_rank and self.config.logging.use_wandb:
            wandb.finish()
            
        dist.destroy_process_group()
