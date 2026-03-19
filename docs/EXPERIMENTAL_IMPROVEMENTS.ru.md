# Экспериментальные методы обучения

Этот документ описывает экспериментальные методы обучения и оптимизации, которые ветка `experimental` предоставляет как библиотечный API.

Описаны только методы тренировки и оптимизации моделей. Проектно-специфичные форматы датасетов и пайплайны сбора данных сюда не входят.

## Публичный API

| API | Значение по умолчанию | Область | Назначение |
| --- | --- | --- | --- |
| `TrainingConfig.use_torch_compile_disable_for_liger` | `True` | setup trainer | Отключает `torch.compile`, если активен Liger |
| `TrainingConfig.resume_quantized_lora_only` | `False` | загрузка checkpoint | Восстанавливает только adapter state для quantized LoRA |
| `TrainingConfig.log_progress_pct` | `True` | логирование | Показывает прогресс `step/max_steps` |
| `TrainingConfig.log_progress_flush` | `True` | логирование | Немедленно сбрасывает progress-лог в stdout |
| `TrainingConfig.use_lora_plus` | `False` | setup optimizer | Включает LoRA+ parameter groups |
| `TrainingConfig.lora_plus_lr_ratio` | `16.0` | setup optimizer | Задаёт отношение `lr_B / lr_A` для LoRA+ |
| `TrainingConfig.use_stable_lora` | `False` | post-step hooks | Включает Stable-LoRA shrinkage на ранних шагах |
| `TrainingConfig.stable_lora_steps` | `100` | post-step hooks | Число шагов применения Stable-LoRA |
| `TrainingConfig.stable_lora_max_shrinkage` | `0.1` | post-step hooks | Общий бюджет shrinkage для `lora_A` |
| `chronicals.lora.resolve_adapter_init_plan()` | n/a | adapter setup | Разрешает безопасную policy для PiSSA/LoftQ/RSLoRA |
| `chronicals.lora.apply_qwen_loftq_fallback()` | n/a | adapter setup | Применяет LoftQ-веса для Qwen-style layout checkpoint |
| `chronicals.training.resolve_compile_decision()` | n/a | policy helper | Разрешает политику совместимости для compile |
| `chronicals.training.load_adapter_checkpoint()` | n/a | checkpoint helper | Загружает только adapter-веса из `model.pt` |
| `chronicals.lora.create_stable_lora_callback()` | n/a | factory hook | Создаёт post-step hook для Stable-LoRA |
| `chronicals.data.DynamicBucketingPacker` | n/a | packing | Length-aware packing utility |
| `chronicals.data.ChunkedPackingCollator` | n/a | packing | Lazy chunking + packing collator |

## Рекомендуемое использование

### Liger + torch.compile

Если вы отдельно не профилировали конкретный стек модели, оставляйте дефолтную policy-конфигурацию.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_torch_compile=True,
    use_torch_compile_disable_for_liger=True,
)
```

Если активен Liger, Chronicals пропустит compilation вместо входа в медленный path.

### Resume для quantized LoRA

Для 4-bit и 8-bit LoRA запусков лучше восстанавливать adapter state, а не полный state dict квантованной базовой модели.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    resume_quantized_lora_only=True,
)
```

Этот путь восстанавливает:

- adapter-веса из `model.pt`
- состояние optimizer при наличии
- состояние scheduler при наличии
- trainer metadata и history

### Stable-LoRA

Stable-LoRA предоставляется как optimizer-step hook и может быть включён через `TrainingConfig`.

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_stable_lora=True,
    stable_lora_steps=100,
    stable_lora_max_shrinkage=0.1,
)
```

Trainer применяет hook только после успешного `optimizer.step()`.

### LoRA+

LoRA+ можно включить прямо в основном trainer path:

```python
from chronicals.config import TrainingConfig

config = TrainingConfig(
    use_lora_plus=True,
    lora_plus_lr_ratio=16.0,
)
```

Так поведение остаётся явной частью библиотечного API, а не локальным project-specific патчем optimizer.

### Policy для quantized adapter init

Для этапа создания адаптеров, который живёт вне `ChronicalsTrainer`, используйте отдельный helper:

```python
from chronicals.lora import resolve_adapter_init_plan

plan = resolve_adapter_init_plan(
    init_method="pissa",
    load_in_4bit=True,
    prefer_loftq_for_quantized=True,
    use_rslora=True,
)
```

Так библиотека явно разрешает безопасный init-plan: PiSSA отключается для quantized adapters, а для 4-bit пути предпочтителен LoftQ.

### Optimized Packing

Используйте packing utilities напрямую, если нужен lazy length-aware packing вне встроенного fixed-shape `SequencePacker`.

```python
from chronicals.data import create_optimized_collator

collator = create_optimized_collator(
    max_length=4096,
    pad_token_id=0,
)
```

## Заметки по совместимости

### LoRA-запуски на моделях семейства Qwen

- Для Liger по-прежнему используется `apply_liger_kernel_to_qwen2`.
- `use_torch_compile_disable_for_liger=True` остаётся наиболее безопасным дефолтом.
- Для quantized LoRA checkpoint предпочтителен adapter-only resume.
- Для checkpoint layout с ключами `model.language_model.*` доступен `apply_qwen_loftq_fallback()`.

### LoRA-запуски на моделях семейства LLaMA

- Тот же adapter-only resume path работает и для quantized LoRA checkpoint.
- Stable-LoRA и LoRA+ не завязаны на конкретную модель, если LoRA-модули следуют стандартному PEFT naming.

### FP8 + LoRA

`convert_linear_to_fp8()` теперь сохраняет placement устройства при конверсии `nn.Linear`. Это устраняет ситуацию, где FP8-слои остаются на CPU, а LoRA-тензоры уже находятся на GPU.

## Замечания по дизайну

Ветка `experimental` теперь выносит эти возможности в небольшие policy/helper-модули:

- `chronicals.training.compile_policy`
- `chronicals.training.checkpoint_policies`
- `chronicals.training.step_hooks`

Это уменьшает связанность trainer internals и делает новые контракты тестируемыми отдельно от полного train run.
