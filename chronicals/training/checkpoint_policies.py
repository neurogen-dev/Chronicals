"""
Checkpoint helpers for ChronicalsTrainer.

The goal is to keep adapter-only resume logic reusable and testable without
hardcoding PEFT-specific details deep inside the trainer flow.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple

import torch

ADAPTER_KEY_MARKERS: Tuple[str, ...] = (
    "lora_A",
    "lora_B",
    "lora_embedding_A",
    "lora_embedding_B",
    "modules_to_save",
)


@dataclass(frozen=True)
class AdapterCheckpointLoadResult:
    """Summary of an adapter-only checkpoint restore."""

    loaded_keys: int
    missing_keys: int
    unexpected_keys: int


def filter_adapter_state_dict(
    state_dict: Dict[str, Any],
    adapter_key_markers: Iterable[str] = ADAPTER_KEY_MARKERS,
) -> Dict[str, Any]:
    """Return only adapter-related keys from a checkpoint state dict."""
    markers = tuple(adapter_key_markers)
    return {
        key: value
        for key, value in state_dict.items()
        if any(marker in key for marker in markers)
    }


def load_adapter_checkpoint(
    model: torch.nn.Module,
    checkpoint_dir: str,
    *,
    map_location: str | torch.device = "cpu",
) -> AdapterCheckpointLoadResult:
    """Load only adapter keys from `model.pt` in a checkpoint directory."""
    model_path = os.path.join(checkpoint_dir, "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"model.pt not found in checkpoint: {checkpoint_dir}")

    raw_state = torch.load(model_path, map_location=map_location)
    filtered_state = filter_adapter_state_dict(raw_state)
    missing, unexpected = model.load_state_dict(filtered_state, strict=False)
    return AdapterCheckpointLoadResult(
        loaded_keys=len(filtered_state),
        missing_keys=len(missing),
        unexpected_keys=len(unexpected),
    )


def restore_trainer_state(trainer: Any, checkpoint_dir: str) -> None:
    """Best-effort restore of optimizer, scheduler and trainer metadata."""
    optimizer_path = os.path.join(checkpoint_dir, "optimizer.pt")
    if os.path.exists(optimizer_path):
        trainer.optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))

    scheduler_path = os.path.join(checkpoint_dir, "scheduler.pt")
    if os.path.exists(scheduler_path) and getattr(trainer, "scheduler", None) is not None:
        trainer.scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    trainer_state_path = os.path.join(checkpoint_dir, "trainer_state.pt")
    if os.path.exists(trainer_state_path):
        state = torch.load(trainer_state_path, map_location="cpu")
        if isinstance(state, tuple):
            state = state[0]
        trainer.state.global_step = state.get("global_step", trainer.state.global_step)
        trainer.state.epoch = state.get("epoch", trainer.state.epoch)
        trainer.state.best_loss = state.get("best_loss", trainer.state.best_loss)
        trainer.state.total_tokens = state.get("total_tokens", trainer.state.total_tokens)
        trainer.state.total_time = state.get("total_time", trainer.state.total_time)

    history_path = os.path.join(checkpoint_dir, "training_history.pt")
    if os.path.exists(history_path):
        history = torch.load(history_path, map_location="cpu")
        if isinstance(history, tuple):
            history = history[0]
        trainer.state.log_history = history


def write_checkpoint_manifest(
    checkpoint_dir: str,
    *,
    strategy: str,
    artifacts: List[str],
    metadata: Dict[str, Any] | None = None,
) -> None:
    """Write a small JSON manifest describing checkpoint contents."""
    manifest = {
        "version": 1,
        "strategy": strategy,
        "artifacts": artifacts,
        "metadata": metadata or {},
    }
    with open(os.path.join(checkpoint_dir, "manifest.json"), "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=True, indent=2, sort_keys=True)
