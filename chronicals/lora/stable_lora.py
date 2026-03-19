"""
Stable-LoRA: Weight shrinkage for lora_A during early training steps.

Reference: arXiv:2603.05204 — Stabilizing Feature Learning of Low-Rank Adaptation (ICLR 2026).
Progressive shrinkage of lora_A in the first steps eliminates instability with non-zero init.
"""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn


def apply_stable_lora_shrinkage(
    model: nn.Module,
    step: int,
    total_shrinkage_steps: int,
    max_shrinkage: float = 0.1,
) -> None:
    """
    Apply weight shrinkage to lora_A in all LoRA layers.

    For the first total_shrinkage_steps steps, each step multiplies lora_A by (1 - decay),
    where decay = max_shrinkage / total_shrinkage_steps.

    Args:
        model: PEFT model with LoRA adapters
        step: Current global training step
        total_shrinkage_steps: Number of steps for progressive shrinkage
        max_shrinkage: Maximum shrinkage fraction (0.1 = 10%)
    """
    if step >= total_shrinkage_steps:
        return
    decay = max_shrinkage / total_shrinkage_steps
    multiplier = 1.0 - decay

    for _, module in model.named_modules():
        if hasattr(module, "lora_A"):
            for _, lora_a_layer in module.lora_A.items():
                if hasattr(lora_a_layer, "weight") and lora_a_layer.weight.requires_grad:
                    with torch.no_grad():
                        lora_a_layer.weight.data.mul_(multiplier)


def create_stable_lora_callback(
    total_shrinkage_steps: int = 100,
    max_shrinkage: float = 0.1,
) -> Callable[[nn.Module, int], None]:
    """
    Create a callback that applies Stable-LoRA after each optimizer step.

    Use by wrapping optimizer.step():
        callback = create_stable_lora_callback(total_shrinkage_steps=100)
        def wrapped_step():
            optimizer.step()
            callback(model, global_step)

    Args:
        total_shrinkage_steps: Number of steps for progressive shrinkage
        max_shrinkage: Maximum shrinkage fraction (0.1 = 10%)

    Returns:
        Callback function (model, step) -> None
    """
    def _on_step(model: nn.Module, step: int) -> None:
        apply_stable_lora_shrinkage(model, step, total_shrinkage_steps, max_shrinkage)

    return _on_step
