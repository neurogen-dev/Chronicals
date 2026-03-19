import torch

from chronicals.data import ChunkedPackingCollator, DynamicBucketingPacker, create_optimized_collator


def test_dynamic_bucketing_packer_respects_max_length():
    packer = DynamicBucketingPacker(max_length=16, pad_token_id=0, num_buckets=4)
    items = [
        {"input_ids": torch.arange(5), "labels": torch.arange(5)},
        {"input_ids": torch.arange(6), "labels": torch.arange(6)},
        {"input_ids": torch.arange(4), "labels": torch.arange(4)},
    ]
    packed = packer.pack_batch(items)
    assert packed
    assert all(batch["input_ids"].shape[0] == 16 for batch in packed)
    assert all(batch["labels"].shape[0] == 16 for batch in packed)


def test_chunked_packing_collator_returns_attention_mask():
    collator = ChunkedPackingCollator(max_length=16, pad_token_id=0, chunk_overlap=4)
    batch = collator(
        [
            {"input_ids": list(range(10)), "labels": list(range(10))},
            {"input_ids": list(range(8)), "labels": list(range(8))},
        ]
    )
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape[0] == 16
    assert batch["attention_mask"].shape == batch["input_ids"].shape


def test_create_optimized_collator_returns_chunked_collator():
    collator = create_optimized_collator(max_length=32, pad_token_id=0)
    assert isinstance(collator, ChunkedPackingCollator)
