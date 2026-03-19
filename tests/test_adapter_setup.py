from chronicals.lora import (
    get_qwen_loftq_weight_candidates,
    resolve_adapter_init_plan,
)


def test_quantized_plan_disables_pissa_and_enables_loftq_for_4bit():
    plan = resolve_adapter_init_plan(
        init_method="pissa",
        load_in_4bit=True,
        prefer_loftq_for_quantized=True,
        use_rslora=True,
    )
    assert plan.quantized is True
    assert plan.effective_init_method is None
    assert plan.use_loftq is True
    assert plan.use_rslora is True
    assert any("PiSSA disabled" in note for note in plan.notes)


def test_non_quantized_plan_keeps_requested_init_method():
    plan = resolve_adapter_init_plan(
        init_method="pissa",
        load_in_4bit=False,
        load_in_8bit=False,
        prefer_loftq_for_quantized=True,
        use_rslora=False,
    )
    assert plan.quantized is False
    assert plan.effective_init_method == "pissa"
    assert plan.use_loftq is False
    assert plan.use_rslora is False


def test_qwen_loftq_candidates_include_language_model_prefix():
    candidates = get_qwen_loftq_weight_candidates("model.layers.0.mlp.down_proj")
    assert candidates == (
        "model.language_model.layers.0.mlp.down_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
    )
