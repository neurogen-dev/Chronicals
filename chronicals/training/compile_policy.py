"""
Compile policy helpers for ChronicalsTrainer.

This module keeps compatibility heuristics out of the trainer hot path and
turns them into a small, testable contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass(frozen=True)
class CompileDecision:
    """Resolved torch.compile decision for a training run."""

    enabled: bool
    reason: str
    mode: Optional[str] = None
    backend: Optional[str] = None
    fullgraph: bool = False
    dynamic: Optional[bool] = None
    use_regional: bool = True
    adjusted: bool = False


def resolve_compile_decision(
    args: Any,
    *,
    torch_compile_available: bool,
    use_liger: bool = False,
) -> CompileDecision:
    """
    Resolve whether torch.compile should be used for the current run.

    The result is intentionally small and serializable so the trainer can apply
    it without embedding model-specific compatibility branches in the setup code.
    """
    if not getattr(args, "use_torch_compile", False) or not torch_compile_available:
        return CompileDecision(
            enabled=False,
            reason="torch.compile unavailable or disabled",
        )

    if getattr(args, "torch_compile_disable", False):
        return CompileDecision(
            enabled=False,
            reason="torch.compile disabled via config",
        )

    mode = getattr(args, "torch_compile_mode", "default")
    if use_liger and getattr(args, "use_torch_compile_disable_for_liger", True):
        return CompileDecision(
            enabled=False,
            reason="torch.compile disabled for Liger compatibility",
        )

    adjusted = False
    reason = "torch.compile enabled"
    if use_liger and mode == "reduce-overhead":
        mode = "default"
        adjusted = True
        reason = "torch.compile mode downgraded for Liger compatibility"

    return CompileDecision(
        enabled=True,
        reason=reason,
        mode=mode,
        backend=getattr(args, "torch_compile_backend", "inductor"),
        fullgraph=getattr(args, "torch_compile_fullgraph", False),
        dynamic=getattr(args, "torch_compile_dynamic", None),
        use_regional=getattr(args, "torch_compile_regional", True),
        adjusted=adjusted,
    )
