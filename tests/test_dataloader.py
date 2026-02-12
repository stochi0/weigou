"""
Dataloader tests. Run with:
  torchrun --nproc_per_node 2 tests/test_dataloader.py cp    # CP split
  torchrun --nproc_per_node 1 tests/test_dataloader.py loop  # infinite cycling
"""
import os
import sys
from functools import partial

import torch
from datasets import Features, Sequence, Value, load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader, DistributedSampler

import weigou.parallel.manager as pgm
from weigou.data import MicroBatchDataLoader
from tests.conftest import init_dist

# Minimal config
DS, TK = "roneneldan/TinyStories", "HuggingFaceTB/SmolLM-135M"
SEQ, MB = 8, 2
N = 4


def _device():
    r = int(os.environ.get("LOCAL_RANK", 0))
    return f"cuda:{r}" if torch.cuda.is_available() else "cpu"


def _loader_kw(seq=SEQ, num_samples=N):
    return dict(
        micro_batch_size=MB,
        seq_length=seq,
        dataset_name=DS,
        tokenizer_name=TK,
        grad_acc_steps=1,
        device=_device(),
        num_workers=0,
        num_proc=1,
        num_samples=num_samples,
    )


def _full_collate(batch, seq_length):
    ids = torch.stack([torch.tensor(x["input_ids"]) for x in batch])
    return {
        "input_ids": ids[:, :seq_length],
        "target_ids": ids[:, 1 : seq_length + 1],
        "position_ids": torch.arange(seq_length, dtype=torch.long).unsqueeze(0).expand(ids.size(0), -1),
    }


def _ref_batch(seq_length, micro_batch):
    """Full-sequence batch (no CP split) for reference."""
    tokenizer = AutoTokenizer.from_pretrained(TK)
    ds = load_dataset(DS, split="train").select(range(N))
    fn = partial(MicroBatchDataLoader.tokenizer_group_text, tokenizer=tokenizer, sequence_length=seq_length)
    mapped = ds.map(
        fn,
        input_columns="text",
        remove_columns=ds.column_names,
        features=Features({"input_ids": Sequence(Value("int64"), length=seq_length + 1)}),
        batched=True,
    )
    m = pgm.process_group_manager
    sampler = DistributedSampler(mapped, num_replicas=m.dp_world_size, rank=m.dp_rank, shuffle=False)
    loader = DataLoader(mapped, batch_size=micro_batch, sampler=sampler, collate_fn=lambda b: _full_collate(b, seq_length))
    return next(iter(loader))


def test_cp_split():
    """CP slices must match reference."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node 2 tests/test_dataloader.py cp")
    CP = 2
    init_dist(tp=1, cp=CP, pp=1, dp=1)

    loader = MicroBatchDataLoader(**{**_loader_kw(), "pin_memory": False})
    ref = _ref_batch(SEQ, MB)
    batch = next(loader)

    r, size = pgm.process_group_manager.cp_rank, SEQ // CP
    lo, hi = r * size, (r + 1) * size
    assert torch.equal(ref["input_ids"][:, lo:hi], batch["input_ids"]), "CP slice mismatch"


def test_infinite_cycle():
    """Loader must cycle on epoch end."""
    from tests.conftest import skip_if_not_distributed
    skip_if_not_distributed("run with: torchrun --nproc_per_node 1 tests/test_dataloader.py loop")
    init_dist(tp=1, cp=1, pp=1, dp=1)

    loader = MicroBatchDataLoader(**{**_loader_kw(seq=32, num_samples=N), "pin_memory": False})
    seen = set()
    for _ in range(15):
        batch = next(loader)
        k = tuple(batch["input_ids"].flatten().tolist())
        if k in seen:
            return
        seen.add(k)
    assert False, "No repeat in 15 iters"


if __name__ == "__main__":
    mode = (sys.argv[1] if len(sys.argv) > 1 else "cp").lower()
    if mode == "cp":
        test_cp_split()
    else:
        test_infinite_cycle()
    print("OK")
