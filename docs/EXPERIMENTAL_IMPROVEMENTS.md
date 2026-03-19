# Experimental Improvements

This document describes experimental training and optimization improvements that may be enabled or configured for specific hardware and model setups.

---

# Экспериментальные улучшения

В этом документе описаны экспериментальные улучшения обучения и оптимизации, которые могут быть включены или настроены для конкретных конфигураций железа и моделей.

---

## 1. FP8 Device Fix

`convert_linear_to_fp8` now explicitly moves `fp8_linear` to `module.weight.device` before returning. This fixes the "mat2 on CPU" error that occurs when FP8 quantization is used together with LoRA adapters — previously, FP8 linear layers could remain on CPU while LoRA tensors were on GPU, causing device mismatch during forward pass.

**When to use:** Enable when training with FP8 + LoRA (e.g., `--fp8` or similar flags). If you see `RuntimeError: mat2 must be on the same device as mat1`, this fix addresses it.

---

`convert_linear_to_fp8` теперь явно перемещает `fp8_linear` на `module.weight.device` перед возвратом. Это устраняет ошибку «mat2 on CPU», возникающую при совместном использовании FP8-квантизации и LoRA-адаптеров — ранее FP8 linear-слои могли оставаться на CPU, пока тензоры LoRA были на GPU, что вызывало несовпадение устройств при forward pass.

**Когда использовать:** Включать при обучении с FP8 + LoRA (например, флаг `--fp8`). При ошибке `RuntimeError: mat2 must be on the same device as mat1` этот фикс её устраняет.

---

## 2. torch.compile + Liger

When both Liger kernel and `torch.compile` are enabled, a severe slowdown (3–4×) can occur. The option `use_torch_compile_disable_for_liger` addresses this: when `True` and Liger is enabled, `torch.compile` is skipped for the model. Liger provides its own optimizations, so disabling compile in this case avoids the regression without losing performance.

**When to use:** Set `use_torch_compile_disable_for_liger=True` when training with Liger (e.g., Qwen3.5 with Liger kernel). If you observe 3–4× slower training with both enabled, this option resolves it.

---

При одновременном включении ядра Liger и `torch.compile` возможен сильный замедление (в 3–4 раза). Опция `use_torch_compile_disable_for_liger` решает эту проблему: при `True` и включённом Liger `torch.compile` для модели не применяется. Liger даёт собственные оптимизации, поэтому отключение compile в этом случае устраняет регрессию без потери производительности.

**Когда использовать:** Устанавливать `use_torch_compile_disable_for_liger=True` при обучении с Liger (например, Qwen3.5 с ядром Liger). При замедлении обучения в 3–4 раза при одновременном включении обоих — эта опция устраняет проблему.

---

## 3. Quantized Resume

For 4-bit PEFT training, full checkpoint resume can fail or be inefficient because loading the quantized base model state dict is heavy and sometimes incompatible. The option `resume_quantized_lora_only` changes resume behavior: only LoRA-related keys (`lora_A`, `lora_B`, `lora_embedding`, etc.) are loaded from the checkpoint. The base model is loaded fresh from the original pretrained weights, and only the adapter state is restored.

**When to use:** Enable when resuming 4-bit LoRA training and encountering OOM, device mismatch, or bnb quantization errors during full checkpoint load. This is a best-effort resume path for quantized PEFT.

---

При 4-bit PEFT обучении полный resume из checkpoint может падать или быть неэффективным, так как загрузка quantized base model state dict тяжёлая и иногда несовместима. Опция `resume_quantized_lora_only` меняет поведение resume: из checkpoint загружаются только LoRA-ключи (`lora_A`, `lora_B`, `lora_embedding` и т.д.). Базовая модель загружается заново из оригинальных pretrained весов, восстанавливается только состояние адаптера.

**Когда использовать:** Включать при resume 4-bit LoRA обучения и появлении OOM, device mismatch или ошибок bnb quantization при полной загрузке checkpoint. Это best-effort путь resume для quantized PEFT.

---

## 4. Progress Logging

The option `log_progress_pct` adds human-readable progress to training logs. When enabled, logs include `[step/max_steps (pct%)]` (e.g., `[500/2000 (25%)]`), making it easy to see completion percentage at a glance without parsing step numbers.

**When to use:** Enable for long runs where you want quick visual feedback on progress. Does not affect training behavior.

---

Опция `log_progress_pct` добавляет в логи обучения понятный прогресс. При включении в логах появляется `[step/max_steps (pct%)]` (например, `[500/2000 (25%)]`), что позволяет быстро оценить процент выполнения без разбора номеров шагов.

**Когда использовать:** Включать при длинных прогонах, когда нужна быстрая визуальная обратная связь по прогрессу. На поведение обучения не влияет.

---

## 5. Stable-LoRA

Stable-LoRA (arXiv:2603.05204) applies progressive weight shrinkage to lora_A during early training steps. This stabilizes feature learning when using non-zero LoRA initialization (e.g., PiSSA, LoftQ). Use `apply_stable_lora_shrinkage(model, step, total_steps, max_shrinkage)` after each optimizer step, or wrap with `create_stable_lora_callback()`.

**When to use:** For 4-bit/8-bit LoRA with LoftQ or PiSSA init, especially on long-context or hybrid datasets. Typical: `total_shrinkage_steps=100`, `max_shrinkage=0.1`.

---

Stable-LoRA (arXiv:2603.05204) применяет прогрессивное сжатие весов lora_A в первые шаги обучения. Стабилизирует обучение признаков при non-zero LoRA инициализации (PiSSA, LoftQ). Использовать `apply_stable_lora_shrinkage(model, step, total_steps, max_shrinkage)` после каждого optimizer.step() или обернуть через `create_stable_lora_callback()`.

**Когда использовать:** Для 4-bit/8-bit LoRA с LoftQ или PiSSA init, особенно на long-context или hybrid датасетах. Типично: `total_shrinkage_steps=100`, `max_shrinkage=0.1`.

---

## 6. Dynamic Bucketing

### DynamicBucketingPacker

A packer that groups sequences by length into buckets and packs them into fixed-length blocks. Reduces padding and improves GPU utilization compared to naive padding to max length.

---

Пакер, который группирует последовательности по длине в бакеты и упаковывает их в блоки фиксированной длины. Уменьшает padding и улучшает загрузку GPU по сравнению с наивным дополнением до максимальной длины.

---

### ChunkedPackingCollator

A data collator that works with chunked or streamed data, packing sequences on-the-fly. Suitable for large datasets where eager full-dataset prepacking would consume too much CPU memory.

---

Коллатор данных, работающий с чанкованными или стриминговыми данными, упаковывающий последовательности на лету. Подходит для больших датасетов, когда eager prepack всего датасета потребляет слишком много CPU-памяти.

---

### create_optimized_collator

A factory that returns an optimized packing collator based on dataset size, sequence length distribution, and available memory. Chooses between eager prepacking and lazy packing strategies.

**When to use:** For large long-context datasets (e.g., 50k+ samples, 4k+ tokens), prefer lazy packing via `ChunkedPackingCollator` or `create_optimized_collator` to avoid CPU prepack stalls. For smaller datasets, eager prepacking may be faster.

---

Фабрика, возвращающая оптимизированный packing collator на основе размера датасета, распределения длин последовательностей и доступной памяти. Выбирает между eager prepacking и lazy packing стратегиями.

**Когда использовать:** Для больших long-context датасетов (например, 50k+ сэмплов, 4k+ токенов) предпочтительнее lazy packing через `ChunkedPackingCollator` или `create_optimized_collator`, чтобы избежать зависаний на CPU prepack. Для меньших датасетов eager prepacking может быть быстрее.

---

## 7. Model-Specific Notes

### Qwen3.5

Liger applies `apply_liger_kernel_to_qwen2` when `model_type` contains `"qwen"`. If this fails, training continues without Liger. **FP8:** Can cause device mismatch with LoRA — use the FP8 device fix (Section 1) or disable FP8. **torch.compile:** Disable when using Liger (`use_torch_compile_disable_for_liger=True`) to avoid 3–4× slowdown.

---

Liger применяет `apply_liger_kernel_to_qwen2`, когда `model_type` содержит `"qwen"`. При ошибке обучение продолжается без Liger. **FP8:** Может вызывать device mismatch с LoRA — использовать FP8 device fix (раздел 1) или отключить FP8. **torch.compile:** Отключать при использовании Liger (`use_torch_compile_disable_for_liger=True`), чтобы избежать замедления в 3–4 раза.

---

### LLaMA 8B (Saiga / YandexGPT)

FP8 + LoRA causes device mismatch (mat2 on CPU). **Recommendation:** Disable FP8 for Saiga/8B LLaMA variants. Use bf16 or 8-bit quantization instead. LoRA+ is supported by default.

---

FP8 + LoRA вызывает device mismatch (mat2 on CPU). **Рекомендация:** Отключать FP8 для Saiga и 8B LLaMA-вариантов. Использовать bf16 или 8-bit квантизацию. LoRA+ поддерживается по умолчанию.

---

### 9B on RTX 4090

**torch.compile:** Disable for stability — on 4090 with 9B, compile can cause OOM on backward or unstable behavior. **Quantization:** 4-bit + LoftQ + RSLoRA is the recommended profile. PiSSA is not compatible with bnb 4-bit; use LoftQ instead. **Batch size:** Start with `batch=1` and high `grad_accum`; increase batch only if VRAM allows.

---

**torch.compile:** Отключать для стабильности — на 4090 с 9B compile может вызывать OOM на backward или нестабильное поведение. **Квантизация:** Рекомендуемый профиль — 4-bit + LoftQ + RSLoRA. PiSSA несовместим с bnb 4-bit; использовать LoftQ. **Batch size:** Начинать с `batch=1` и высоким `grad_accum`; увеличивать batch только при достаточном VRAM.
