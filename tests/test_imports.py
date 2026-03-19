"""Test that all Chronicals modules can be imported correctly."""

import pytest


class TestCoreImports:
    """Test core package imports."""

    def test_import_chronicals(self):
        """Test main package import."""
        import chronicals
        assert hasattr(chronicals, "__version__")
        assert chronicals.__version__

    def test_import_version(self):
        """Test version function."""
        from chronicals import get_version
        version = get_version()
        assert version
        assert isinstance(version, str)
        assert "." in version

    def test_import_device_info(self):
        """Test device info function."""
        from chronicals import get_device_info
        info = get_device_info()
        assert isinstance(info, dict)
        assert "cuda_available" in info
        assert "triton_available" in info


class TestConfigImports:
    """Test configuration module imports."""

    def test_import_chronicals_config(self):
        """Test ChronicalsConfig import."""
        from chronicals.config import ChronicalsConfig
        assert ChronicalsConfig is not None

    def test_import_training_config(self):
        """Test TrainingConfig import."""
        from chronicals.config import TrainingConfig
        assert TrainingConfig is not None

    def test_config_instantiation(self):
        """Test config can be instantiated with defaults."""
        from chronicals.config import ChronicalsConfig
        config = ChronicalsConfig()
        assert config is not None
        assert hasattr(config, "hidden_size")
        assert hasattr(config, "num_attention_heads")


class TestOptimizerImports:
    """Test optimizer module imports."""

    def test_import_lora_plus_optimizer(self):
        """Test LoRAPlusOptimizer import."""
        from chronicals.optimizers import LoRAPlusOptimizer
        assert LoRAPlusOptimizer is not None

    def test_import_lora_plus_adamw(self):
        """Test LoRAPlusAdamW import."""
        from chronicals.optimizers import LoRAPlusAdamW
        assert LoRAPlusAdamW is not None

    def test_optimizer_alias(self):
        """Test that LoRAPlusOptimizer is alias for LoRAPlusAdamW."""
        from chronicals.optimizers import LoRAPlusOptimizer, LoRAPlusAdamW
        assert LoRAPlusOptimizer is LoRAPlusAdamW


class TestDataImports:
    """Test data module imports."""

    def test_import_sequence_packer(self):
        """Test SequencePacker import."""
        from chronicals.data import SequencePacker
        assert SequencePacker is not None

    def test_import_packed_batch(self):
        """Test PackedBatch import."""
        from chronicals.data import PackedBatch
        assert PackedBatch is not None

    def test_import_optimized_dataloader_factory(self):
        """Test optimized dataloader factory import."""
        from chronicals.data import create_optimized_dataloader
        assert create_optimized_dataloader is not None

    def test_import_packing_policy(self):
        """Test packing policy helper import."""
        from chronicals.data import resolve_packing_plan
        assert resolve_packing_plan is not None


class TestLoRAImports:
    """Test LoRA helper imports."""

    def test_import_stable_lora_callback(self):
        """Test Stable-LoRA callback import."""
        from chronicals.lora import create_stable_lora_callback
        assert create_stable_lora_callback is not None

    def test_import_adapter_init_policy(self):
        """Test adapter init policy import."""
        from chronicals.lora import resolve_adapter_init_plan
        assert resolve_adapter_init_plan is not None


class TestTrainingImports:
    """Test training module imports."""

    def test_import_chronicals_trainer(self):
        """Test ChronicalsTrainer import."""
        from chronicals.training import ChronicalsTrainer
        assert ChronicalsTrainer is not None

    def test_import_compile_policy_helper(self):
        """Test compile policy helper import."""
        from chronicals.training import resolve_compile_decision
        assert resolve_compile_decision is not None

    def test_import_checkpoint_helper(self):
        """Test adapter checkpoint helper import."""
        from chronicals.training import load_adapter_checkpoint
        assert load_adapter_checkpoint is not None


class TestLazyImports:
    """Test lazy imports from main package."""

    def test_lazy_import_trainer(self):
        """Test lazy import of ChronicalsTrainer."""
        from chronicals import ChronicalsTrainer
        assert ChronicalsTrainer is not None

    def test_lazy_import_config(self):
        """Test lazy import of ChronicalsConfig."""
        from chronicals import ChronicalsConfig
        assert ChronicalsConfig is not None

    def test_lazy_import_optimizer(self):
        """Test lazy import of LoRAPlusOptimizer."""
        from chronicals import LoRAPlusOptimizer
        assert LoRAPlusOptimizer is not None

    def test_lazy_import_sequence_packer(self):
        """Test lazy import of SequencePacker."""
        from chronicals import SequencePacker
        assert SequencePacker is not None

    def test_lazy_import_optimized_dataloader_factory(self):
        """Test lazy import of create_optimized_dataloader."""
        from chronicals import create_optimized_dataloader
        assert create_optimized_dataloader is not None

    def test_lazy_import_packing_policy(self):
        """Test lazy import of resolve_packing_plan."""
        from chronicals import resolve_packing_plan
        assert resolve_packing_plan is not None

    def test_lazy_import_stable_lora_callback(self):
        """Test lazy import of create_stable_lora_callback."""
        from chronicals import create_stable_lora_callback
        assert create_stable_lora_callback is not None

    def test_lazy_import_adapter_init_policy(self):
        """Test lazy import of resolve_adapter_init_plan."""
        from chronicals import resolve_adapter_init_plan
        assert resolve_adapter_init_plan is not None

    def test_lazy_import_compile_policy(self):
        """Test lazy import of resolve_compile_decision."""
        from chronicals import resolve_compile_decision
        assert resolve_compile_decision is not None

    def test_lazy_import_checkpoint_helper(self):
        """Test lazy import of load_adapter_checkpoint."""
        from chronicals import load_adapter_checkpoint
        assert load_adapter_checkpoint is not None


class TestKernelImports:
    """Test kernel module imports (may fail without GPU/Triton)."""

    @pytest.mark.skipif(True, reason="Triton kernels require GPU")
    def test_import_triton_kernels(self):
        """Test Triton kernels import (requires GPU)."""
        from chronicals.kernels import triton_kernels
        assert triton_kernels is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
