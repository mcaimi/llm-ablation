import pytest
import torch

from llm_ablation.hooks import (
    ablate_heads,
    ablate_layer,
    ablate_neurons,
    collect_activations,
    head_hook_point,
)


def _logits(bundle, prompt="hello there, how are you"):
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    with torch.no_grad():
        return bundle.model(**inputs).logits


@pytest.mark.slow
def test_ablate_heads_zero_changes_output_and_restores(tiny_bundle):
    baseline = _logits(tiny_bundle)
    with ablate_heads(tiny_bundle, layer=0, heads=[0]):
        ablated = _logits(tiny_bundle)
    restored = _logits(tiny_bundle)
    assert not torch.allclose(baseline, ablated)
    assert torch.allclose(baseline, restored)


@pytest.mark.slow
def test_ablate_neurons_zero_changes_output_and_restores(tiny_bundle):
    baseline = _logits(tiny_bundle)
    with ablate_neurons(tiny_bundle, layer=0, neuron_idx=[0, 1, 2]):
        ablated = _logits(tiny_bundle)
    restored = _logits(tiny_bundle)
    assert not torch.allclose(baseline, ablated)
    assert torch.allclose(baseline, restored)


@pytest.mark.slow
def test_ablate_layer_zero_changes_output_and_restores(tiny_bundle):
    baseline = _logits(tiny_bundle)
    with ablate_layer(tiny_bundle, layer=0):
        ablated = _logits(tiny_bundle)
    restored = _logits(tiny_bundle)
    assert not torch.allclose(baseline, ablated)
    assert torch.allclose(baseline, restored)


@pytest.mark.slow
def test_mean_ablation_differs_from_zero_ablation(tiny_bundle):
    hp = head_hook_point(0)
    cache = collect_activations(tiny_bundle, ["hello there", "a different sentence here"], [hp])
    with ablate_heads(tiny_bundle, layer=0, heads=[0], mode="zero"):
        zero_logits = _logits(tiny_bundle)
    with ablate_heads(tiny_bundle, layer=0, heads=[0], mode="mean", reference_cache=cache):
        mean_logits = _logits(tiny_bundle)
    assert not torch.allclose(zero_logits, mean_logits)


@pytest.mark.slow
def test_mean_ablation_without_reference_cache_raises(tiny_bundle):
    with pytest.raises(ValueError):
        with ablate_heads(tiny_bundle, layer=0, heads=[0], mode="mean"):
            _logits(tiny_bundle)


@pytest.mark.slow
def test_collect_activations_shapes(tiny_bundle):
    hp = head_hook_point(0)
    cache = collect_activations(tiny_bundle, ["hi", "ok then"], [hp])
    width = tiny_bundle.n_heads * tiny_bundle.head_dim
    assert cache.activations[hp].shape[-1] == width
    assert cache.mean(hp).shape == (width,)
    assert cache.sample(hp).shape == (width,)
