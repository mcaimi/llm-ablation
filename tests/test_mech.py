import numpy as np
import pytest
import torch

from llm_ablation.mech import head_ablation_sweep, logit_diff, run_factual_recall_task, top_heads
from llm_ablation.metrics import cosine_sim, kl_divergence


def test_kl_divergence_of_identical_distributions_is_zero():
    logits = torch.tensor([1.0, 2.0, 3.0])
    assert kl_divergence(logits, logits) == pytest.approx(0.0, abs=1e-6)


def test_kl_divergence_of_different_distributions_is_positive():
    p = torch.tensor([5.0, 0.0, 0.0])
    q = torch.tensor([0.0, 0.0, 5.0])
    assert kl_divergence(p, q) > 0


def test_cosine_sim_of_orthogonal_vectors_is_zero():
    a = torch.tensor([1.0, 0.0])
    b = torch.tensor([0.0, 1.0])
    assert cosine_sim(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_sim_of_identical_vectors_is_one():
    a = torch.tensor([3.0, 4.0])
    assert cosine_sim(a, a) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.slow
def test_logit_diff_returns_a_float(tiny_bundle):
    value = logit_diff(tiny_bundle, "The capital of France is", "Paris", "London")
    assert isinstance(value, float)


@pytest.mark.slow
def test_head_ablation_sweep_shape(tiny_bundle):
    pairs = [("The capital of France is", "Paris", "London")]
    sweep = head_ablation_sweep(tiny_bundle, pairs)
    assert sweep.shape == (tiny_bundle.n_layers, tiny_bundle.n_heads)


@pytest.mark.slow
def test_run_factual_recall_task_smoke(tiny_bundle):
    pairs = [("The capital of France is", "Paris", "London")]
    result = run_factual_recall_task(tiny_bundle, pairs)
    assert result["head_ablation_drop"].shape == (tiny_bundle.n_layers, tiny_bundle.n_heads)
    assert len(result["baseline_logit_diffs"]) == 1


def test_top_heads_orders_by_magnitude():
    matrix = np.array([[0.1, 5.0], [3.0, -2.0]])
    result = top_heads(matrix, k=2)
    assert result == [(0, 1, 5.0), (1, 0, 3.0)]


def test_top_heads_respects_k():
    matrix = np.arange(12, dtype=float).reshape(3, 4)
    assert len(top_heads(matrix, k=3)) == 3
