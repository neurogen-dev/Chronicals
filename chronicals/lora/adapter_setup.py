"""
Adapter setup helpers for LoRA/QLoRA style training.

This module keeps model-family-specific adapter initialization rules outside the
trainer loop while still exposing reusable, testable helpers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class AdapterInitPlan:
    """Resolved adapter initialization plan for a training run."""

    quantized: bool
    effective_init_method: Optional[str]
    use_loftq: bool
    use_rslora: bool
    notes: Tuple[str, ...] = ()


def resolve_adapter_init_plan(
    *,
    init_method: Optional[str] = None,
    load_in_4bit: bool = False,
    load_in_8bit: bool = False,
    prefer_loftq_for_quantized: bool = True,
    use_rslora: bool = True,
) -> AdapterInitPlan:
    """
    Resolve a safe adapter initialization strategy for quantized and non-quantized runs.

    Current rules:
    - PiSSA is disabled for quantized runs because 4-bit/8-bit adapter init paths are
      not compatible with standard PiSSA initialization.
    - LoftQ is preferred for 4-bit runs when explicitly allowed.
    - RSLoRA stays as an orthogonal toggle and is returned unchanged.
    """
    quantized = load_in_4bit or load_in_8bit
    effective_init_method = init_method
    use_loftq = False
    notes = []

    if quantized and init_method and init_method.lower() == "pissa":
        effective_init_method = None
        notes.append("PiSSA disabled for quantized adapter initialization")

    if load_in_4bit and prefer_loftq_for_quantized:
        use_loftq = True
        notes.append("LoftQ enabled for 4-bit adapter initialization")
    elif load_in_8bit and prefer_loftq_for_quantized:
        notes.append("LoftQ not applied for 8-bit adapter initialization")

    return AdapterInitPlan(
        quantized=quantized,
        effective_init_method=effective_init_method,
        use_loftq=use_loftq,
        use_rslora=use_rslora,
        notes=tuple(notes),
    )


def get_qwen_loftq_weight_candidates(module_name: str) -> Tuple[str, ...]:
    """
    Return candidate safetensor weight keys for a Qwen-style PEFT module name.

    PEFT LoRA modules are often addressed as `model.*`, while some Qwen-family
    checkpoints store the corresponding weight tensors under
    `model.language_model.*`.
    """
    candidates = [f"{module_name}.weight"]
    if module_name.startswith("model."):
        candidates.insert(0, f"model.language_model.{module_name[len('model.'):]}.weight")
    return tuple(candidates)


def apply_qwen_loftq_fallback(peft_model, model_path: str, adapter_name: str = "default") -> int:
    """
    Apply LoftQ weights for Qwen-style checkpoints whose tensor keys live under
    `model.language_model.*` while PEFT module names live under `model.*`.

    Returns the number of patched Linear4bit LoRA modules.
    """
    from peft.tuners.lora import Linear4bit
    from peft.utils.loftq_utils import _SafetensorLoader, _loftq_init_new

    loader = _SafetensorLoader(peft_model, model_path)
    prefix = "base_model.model."
    patched = 0

    for name, module in peft_model.named_modules():
        if not isinstance(module, Linear4bit):
            continue
        if not name.startswith(prefix):
            continue

        short_name = name[len(prefix):]
        tensor = None
        for candidate in get_qwen_loftq_weight_candidates(short_name):
            if candidate in loader.weight_map:
                tensor = loader.get_tensor(candidate)
                break

        if tensor is None:
            raise KeyError(f"LoftQ fallback: weight key not found for {short_name}")

        reduced_rank = module.r[adapter_name]
        lora_A, lora_B = _loftq_init_new(module.weight, tensor, num_bits=4, reduced_rank=reduced_rank)
        module.lora_A[adapter_name].weight.data = lora_A
        module.lora_B[adapter_name].weight.data = lora_B
        patched += 1

    if patched == 0:
        raise ValueError("LoftQ fallback: no Linear4bit LoRA modules found")

    return patched
