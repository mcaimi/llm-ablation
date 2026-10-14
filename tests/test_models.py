import pytest

from llm_ablation.models import get_accelerator_device

def test_resolve_device_default_is_valid():
    device, dtype = get_accelerator_device()
    assert device in {"cpu", "cuda", "mps"}


@pytest.mark.slow
def test_load_model_exposes_architecture(tiny_bundle):
    assert tiny_bundle.n_layers == 2
    assert tiny_bundle.n_heads == 4
    assert tiny_bundle.head_dim == 2
