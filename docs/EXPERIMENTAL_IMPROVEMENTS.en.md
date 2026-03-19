# Experimental Training Methods

This guide documents the experimental training and optimization methods exposed by the `experimental` branch as library-facing APIs.

These features are intended for model adaptation and systems optimization only. They do not depend on any project-specific dataset format or data collection pipeline.

## Public API

| API | Default | Scope | Purpose |
| --- | --- | --- | --- |
| `TrainingConfig.use_torch_compile_disable_for_liger` | `True` | trainer setup | Disables `torch.compile` when Liger is active |
| `TrainingConfig.resume_quantized_lora_only` | `False` | checkpoint loading | Restores adapter-only state for quantized LoRA runs |
| `TrainingConfig.log_progress_pct` | `True` | logging | Shows `step/max_steps` progress |
| `TrainingConfig.log_progress_flush` | `True` | logging | Forces progress output to flush immediately |
| `TrainingConfig.use_lora_plus` | `False` | optimizer setup | Enables LoRA+ parameter groups |
| `TrainingConfig.lora_plus_lr_ratio` | `16.0` | optimizer setup | Sets the `lr_B / lr_A` ratio for LoRA+ |
| `TrainingConfig.use_stable_lora` | `False` | optimizer-step hooks | Enables Stable-LoRA shrinkage during early steps |
| `TrainingConfig.stable_lora_steps` | `100` | optimizer-step hooks | Number of steps to apply Stable-LoRA |
| `TrainingConfig.stable_lora_max_shrinkage` | `0.1` | optimizer-step hooks | Total shrinkage budget for `lora_A` |
| `chronicals.training.resolve_compile_decision()` | n/a | policy helper | Resolves compile compatibility decisions |
| `chronicals.training.load_adapter_checkpoint()` | n/a | checkpoint helper | Loads only adapter weights from `model.pt` |
| `chronicals.lora.create_stable_lora_callback()` | n/a | hook factory | Creates a post-step Stable-LoRA hook |
| `chronicals.data.DynamicBucketingPacker` | n/a | packing | Length-aware packing utility |
| `chronicals.data.ChunkedPackingCollator` | n/a | packing | Lazy chunking + packing collator |

## Recommended Usage

### Liger + torch.compile

Use the default compatibility policy unless you have already profiled your specific model stack.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_torch_compile=True,
    use_torch_compile_disable_for_liger=True,
)
```

If Liger is active, Chronicals will skip compilation instead of silently entering a slow path.

### Quantized LoRA Resume

For 4-bit or 8-bit LoRA runs, resume from adapter state rather than reloading a full quantized base-model state dict.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    resume_quantized_lora_only=True,
)
```

This path restores:

- adapter weights from `model.pt`
- optimizer state when available
- scheduler state when available
- trainer metadata and training history

### Stable-LoRA

Stable-LoRA is exposed as an optimizer-step hook and can be enabled through `TrainingConfig`.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_stable_lora=True,
    stable_lora_steps=100,
    stable_lora_max_shrinkage=0.1,
)
```

The trainer applies the hook only after successful optimizer updates.

### LoRA+

LoRA+ can be enabled directly in the trainer path:

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_lora_plus=True,
    lora_plus_lr_ratio=16.0,
)
```

This keeps the library contract explicit instead of relying on project-local optimizer replacement code.

### Optimized Packing

Use the packing utilities directly when you need lazy length-aware packing outside the built-in fixed-shape `SequencePacker`.

```python
from chronicals.data import create_optimized_collator

collator = create_optimized_collator(
    max_length=4096,
    pad_token_id=0,
)
```

## Compatibility Notes

### Qwen-family LoRA runs

- `apply_liger_kernel_to_qwen2` is still the relevant Liger entry point.
- `use_torch_compile_disable_for_liger=True` remains the safest default.
- Adapter-only resume is recommended for quantized LoRA checkpoints.

### LLaMA-family LoRA runs

- The same adapter-only resume path works for quantized LoRA checkpoints.
- Stable-LoRA and LoRA+ are model-agnostic as long as LoRA modules follow standard PEFT naming.

### FP8 + LoRA

`convert_linear_to_fp8()` now preserves device placement when converting `nn.Linear` modules. This avoids FP8 layers remaining on CPU while LoRA tensors are already on GPU.

## Design Notes

The `experimental` branch now exposes these features through small policy/helper modules:

- `chronicals.training.compile_policy`
- `chronicals.training.checkpoint_policies`
- `chronicals.training.step_hooks`

This keeps trainer internals smaller and makes the contracts testable outside a full training run.
