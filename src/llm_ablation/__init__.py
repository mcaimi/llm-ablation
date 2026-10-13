"""
    helper functions used in jupyter notebooks.
    can also be imported directly in a python interpreder
    or used with 'uv run python' in an uv-managed venv
"""

# model checkpoint stuff - load weights, detect accelerators...
from llm_ablation.models import ModelBundle, load_model

# hook layers and perform ablation at layer/weight level
from llm_ablation.hooks import (
    ActivationCache,
    ablate_heads,
    ablate_layer,
    ablate_neurons,
    collect_activations,
)
# mechanistic ablation helpers
from llm_ablation.mech import (
    head_ablation_sweep,
    logit_diff,
    run_factual_recall_task,
    run_ioi_style_task,
    top_heads,
)

# exposed functions/classes
__all__ = [
    "ModelBundle",
    "load_model",
    "ActivationCache",
    "ablate_heads",
    "ablate_layer",
    "ablate_neurons",
    "collect_activations",
    "head_ablation_sweep",
    "logit_diff",
    "run_factual_recall_task",
    "run_ioi_style_task",
    "top_heads"
]
