"""
Packing policy helpers for Chronicals data loading.

These helpers keep runtime selection logic separate from trainer and dataloader
implementations so packing behavior remains explicit and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional


PackingRuntime = Literal["fixed_shape", "lazy_collator", "auto"]


@dataclass(frozen=True)
class PackingPlan:
    """Resolved runtime plan for sequence packing."""

    use_packing: bool
    runtime: PackingRuntime | Literal["disabled"]
    reason: str


def resolve_packing_plan(
    *,
    use_packing: bool,
    packing_runtime: PackingRuntime = "fixed_shape",
    dataset_size: Optional[int] = None,
    max_length: int = 4096,
    lazy_min_samples: int = 50_000,
    lazy_min_length: int = 4096,
) -> PackingPlan:
    """
    Resolve whether packing should use fixed-shape batching or lazy collator mode.

    `fixed_shape` preserves the existing Chronicals behavior and remains the
    default. `auto` chooses lazy collator mode only for sufficiently large,
    long-context datasets.
    """
    if not use_packing:
        return PackingPlan(
            use_packing=False,
            runtime="disabled",
            reason="sequence packing disabled",
        )

    if packing_runtime == "fixed_shape":
        return PackingPlan(
            use_packing=True,
            runtime="fixed_shape",
            reason="explicit fixed-shape packing runtime",
        )

    if packing_runtime == "lazy_collator":
        return PackingPlan(
            use_packing=True,
            runtime="lazy_collator",
            reason="explicit lazy collator packing runtime",
        )

    large_dataset = dataset_size is not None and dataset_size >= lazy_min_samples
    long_context = max_length >= lazy_min_length
    if large_dataset and long_context:
        return PackingPlan(
            use_packing=True,
            runtime="lazy_collator",
            reason="auto-selected lazy collator for large long-context dataset",
        )

    return PackingPlan(
        use_packing=True,
        runtime="fixed_shape",
        reason="auto-selected fixed-shape packing runtime",
    )
