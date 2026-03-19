import torch
import torch.nn as nn

from chronicals.utils.fp8_deepseek import convert_linear_to_fp8


def test_convert_linear_to_fp8_preserves_device_and_weights():
    linear = nn.Linear(8, 4, bias=True)
    converted = convert_linear_to_fp8(linear)

    assert converted.weight.device == linear.weight.device
    assert torch.allclose(converted.weight, linear.weight)
    assert torch.allclose(converted.bias, linear.bias)
