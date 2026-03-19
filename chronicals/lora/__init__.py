"""Chronicals LoRA Module."""

from .adapter_setup import (
    AdapterInitPlan,
    apply_qwen_loftq_fallback,
    get_qwen_loftq_weight_candidates,
    resolve_adapter_init_plan,
)
from .stable_lora import apply_stable_lora_shrinkage, create_stable_lora_callback

__all__ = [
    "lora_optimized",
    "dora_adapter",
    "gradient_checkpointing_lora",
    "lora_presets",
    "AdapterInitPlan",
    "resolve_adapter_init_plan",
    "get_qwen_loftq_weight_candidates",
    "apply_qwen_loftq_fallback",
    "apply_stable_lora_shrinkage",
    "create_stable_lora_callback",
]
