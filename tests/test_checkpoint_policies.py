from types import SimpleNamespace

import torch
import torch.nn as nn

from chronicals.training import (
    filter_adapter_state_dict,
    load_adapter_checkpoint,
    restore_trainer_state,
)


class DummyAdapterModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.base = nn.Linear(2, 2, bias=False)
        self.lora_A = nn.Linear(2, 1, bias=False)
        self.lora_B = nn.Linear(1, 2, bias=False)


def test_filter_adapter_state_dict_keeps_only_adapter_keys():
    state_dict = {
        "base.weight": torch.ones(2, 2),
        "lora_A.weight": torch.ones(1, 2),
        "lora_B.weight": torch.ones(2, 1),
        "modules_to_save.default.weight": torch.ones(1),
    }
    filtered = filter_adapter_state_dict(state_dict)
    assert set(filtered) == {
        "lora_A.weight",
        "lora_B.weight",
        "modules_to_save.default.weight",
    }


def test_load_adapter_checkpoint_restores_only_lora_weights(tmp_path):
    model = DummyAdapterModel()
    base_before = model.base.weight.detach().clone()

    checkpoint = {
        "base.weight": torch.full_like(model.base.weight, 9.0),
        "lora_A.weight": torch.full_like(model.lora_A.weight, 3.0),
        "lora_B.weight": torch.full_like(model.lora_B.weight, 5.0),
    }
    torch.save(checkpoint, tmp_path / "model.pt")

    result = load_adapter_checkpoint(model, str(tmp_path))
    assert result.loaded_keys == 2
    assert torch.equal(model.base.weight, base_before)
    assert torch.allclose(model.lora_A.weight, torch.full_like(model.lora_A.weight, 3.0))
    assert torch.allclose(model.lora_B.weight, torch.full_like(model.lora_B.weight, 5.0))


def test_restore_trainer_state_restores_metadata_and_history(tmp_path):
    model = DummyAdapterModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
    trainer = SimpleNamespace(
        optimizer=optimizer,
        scheduler=scheduler,
        state=SimpleNamespace(
            global_step=0,
            epoch=0.0,
            best_loss=float("inf"),
            total_tokens=0,
            total_time=0.0,
            log_history=[],
        ),
    )

    torch.save(optimizer.state_dict(), tmp_path / "optimizer.pt")
    torch.save(scheduler.state_dict(), tmp_path / "scheduler.pt")
    torch.save(
        {
            "global_step": 12,
            "epoch": 1.5,
            "best_loss": 0.42,
            "total_tokens": 1024,
            "total_time": 9.5,
        },
        tmp_path / "trainer_state.pt",
    )
    torch.save([{"step": 12, "loss": 0.42}], tmp_path / "training_history.pt")

    restore_trainer_state(trainer, str(tmp_path))
    assert trainer.state.global_step == 12
    assert trainer.state.epoch == 1.5
    assert trainer.state.best_loss == 0.42
    assert trainer.state.total_tokens == 1024
    assert trainer.state.total_time == 9.5
    assert trainer.state.log_history == [{"step": 12, "loss": 0.42}]
