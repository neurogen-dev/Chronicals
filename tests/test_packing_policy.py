from torch.utils.data import Dataset

from chronicals.data import resolve_packing_plan
from chronicals.data.data_loader import create_optimized_dataloader


class DummyTokenizer:
    pad_token_id = 0


class DummyDataset(Dataset):
    def __init__(self, size: int = 8, length: int = 12):
        self.size = size
        self.length = length

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        values = list(range(self.length))
        return {
            "input_ids": values,
            "labels": values,
        }


def test_resolve_packing_plan_returns_lazy_for_large_long_auto_mode():
    plan = resolve_packing_plan(
        use_packing=True,
        packing_runtime="auto",
        dataset_size=60_000,
        max_length=4096,
        lazy_min_samples=50_000,
        lazy_min_length=4096,
    )
    assert plan.runtime == "lazy_collator"


def test_resolve_packing_plan_returns_fixed_shape_for_small_auto_mode():
    plan = resolve_packing_plan(
        use_packing=True,
        packing_runtime="auto",
        dataset_size=1_000,
        max_length=2048,
        lazy_min_samples=50_000,
        lazy_min_length=4096,
    )
    assert plan.runtime == "fixed_shape"


def test_create_optimized_dataloader_can_use_lazy_collator_runtime():
    dataloader = create_optimized_dataloader(
        dataset=DummyDataset(),
        tokenizer=DummyTokenizer(),
        max_length=16,
        batch_size=2,
        use_packing=True,
        packing_runtime="lazy_collator",
        use_prefetching=False,
    )
    batch = next(iter(dataloader))
    assert set(batch) == {"input_ids", "labels", "attention_mask"}
    assert batch["input_ids"].shape[-1] == 16
