"""
Optimizer-step hook utilities for ChronicalsTrainer.

Hooks are executed only after a successful optimizer update and can be used for
training-time methods such as Stable-LoRA.
"""

from __future__ import annotations

from typing import Callable, Iterable, Protocol

import torch.nn as nn


class OptimizerStepHook(Protocol):
    """Callable contract for post-optimizer-step hooks."""

    def __call__(self, model: nn.Module, step: int) -> None:
        ...


def run_optimizer_step_hooks(
    hooks: Iterable[Callable[[nn.Module, int], None]],
    *,
    model: nn.Module,
    step: int,
) -> None:
    """Run all registered post-step hooks in order."""
    for hook in hooks:
        hook(model, step)
