# Experimental Branch

This branch contains training and optimization improvements for specific hardware and model configurations.

## Improvements Included

| Feature | Description |
|---------|-------------|
| **FP8 device fix** | Fixes mat2-on-CPU when using FP8 + LoRA |
| **torch.compile + Liger** | Option to disable compile when Liger is enabled (avoids 3–4× slowdown) |
| **Quantized resume** | Load only LoRA adapters when resuming 4-bit training |
| **Progress logging** | Human-readable `[step/max (pct%)]` in logs |
| **Stable-LoRA** | Weight shrinkage for lora_A (arXiv:2603.05204) |
| **Dynamic bucketing** | `ChunkedPackingCollator`, `DynamicBucketingPacker` for +15–20% throughput |
| **autocast** | Replaced deprecated `torch.cuda.amp.autocast` with `torch.amp.autocast('cuda')` |

## Documentation

See [docs/EXPERIMENTAL_IMPROVEMENTS.md](docs/EXPERIMENTAL_IMPROVEMENTS.md) for full bilingual (EN/RU) documentation.

## Installation

```bash
pip install -e .
# or with extras
pip install -e ".[all]"
```

## Config Options

Add to your `TrainingConfig`:

- `use_torch_compile_disable_for_liger=True` — skip compile when Liger enabled
- `log_progress_pct=True` — show progress percentage in logs
- `resume_quantized_lora_only=True` — load only LoRA keys when resuming quantized training
