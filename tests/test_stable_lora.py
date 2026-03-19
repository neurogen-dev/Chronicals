import torch
import torch.nn as nn

from chronicals.lora import apply_stable_lora_shrinkage, create_stable_lora_callback
from chronicals.training import run_optimizer_step_hooks


class DummyStableLoRAModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.ModuleDict({"default": nn.Linear(4, 2, bias=False)})


def test_stable_lora_shrinkage_applies_expected_multiplier():
    model = DummyStableLoRAModule()
    model.lora_A["default"].weight.data.fill_(1.0)

    apply_stable_lora_shrinkage(model, step=0, total_shrinkage_steps=10, max_shrinkage=0.1)
    expected = 1.0 * (1.0 - 0.1 / 10.0)
    assert torch.allclose(model.lora_A["default"].weight, torch.full_like(model.lora_A["default"].weight, expected))


def test_stable_lora_callback_stops_after_configured_steps():
    model = DummyStableLoRAModule()
    model.lora_A["default"].weight.data.fill_(1.0)
    callback = create_stable_lora_callback(total_shrinkage_steps=2, max_shrinkage=0.2)

    run_optimizer_step_hooks([callback], model=model, step=0)
    after_first = model.lora_A["default"].weight.detach().clone()
    run_optimizer_step_hooks([callback], model=model, step=2)

    assert torch.equal(model.lora_A["default"].weight, after_first)
