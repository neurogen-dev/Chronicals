from types import SimpleNamespace

from chronicals.training import resolve_compile_decision


def make_args(**overrides):
    defaults = {
        "use_torch_compile": True,
        "torch_compile_disable": False,
        "torch_compile_mode": "default",
        "torch_compile_backend": "inductor",
        "torch_compile_fullgraph": False,
        "torch_compile_dynamic": None,
        "torch_compile_regional": True,
        "use_torch_compile_disable_for_liger": True,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def test_compile_policy_disables_compile_for_liger_by_default():
    decision = resolve_compile_decision(
        make_args(),
        torch_compile_available=True,
        use_liger=True,
    )
    assert decision.enabled is False
    assert "Liger" in decision.reason


def test_compile_policy_downgrades_reduce_overhead_when_allowed():
    decision = resolve_compile_decision(
        make_args(
            torch_compile_mode="reduce-overhead",
            use_torch_compile_disable_for_liger=False,
        ),
        torch_compile_available=True,
        use_liger=True,
    )
    assert decision.enabled is True
    assert decision.mode == "default"
    assert decision.adjusted is True


def test_compile_policy_respects_global_disable():
    decision = resolve_compile_decision(
        make_args(use_torch_compile=False),
        torch_compile_available=True,
        use_liger=False,
    )
    assert decision.enabled is False
    assert "disabled" in decision.reason
