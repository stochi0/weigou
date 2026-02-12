"""Unit tests (no torchrun). Run with: pytest tests/test_data_unit.py"""
from transformers import AutoTokenizer

from weigou.data import MicroBatchDataLoader


def test_tokenizer_group_text_chunks():
    """tokenizer_group_text produces correct-length chunks."""
    tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    seq_len = 8
    examples = {"text": ["The cat sat on the mat. The dog ran.", "A bird flew high."]}

    out = MicroBatchDataLoader.tokenizer_group_text(examples, tok, seq_len)

    for chunk in out["input_ids"]:
        assert len(chunk) == seq_len + 1
    assert len(out["input_ids"]) >= 1
