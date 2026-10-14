"""
    Low-level ablation primitives: forward hooks that zero, mean-, or
    resample-ablate a slice of a model's activations for the duration of a
    `with` block, then restore the original forward pass.

    These work on any standard decoder-only HF model that exposes
    `model.model.layers[i].self_attn.o_proj` and `.mlp.down_proj` (Qwen2, Llama,
    Gemma2, Mistral, ...). 
    
    Ablating "head h" means zeroing/replacing the slice `[h*head_dim:(h+1)*head_dim]`
    of the tensor fed into `o_proj` — this is correct even for grouped-query attention, 
    because GQA only changes the width of the K/V projections, not the width of the 
    concatenated attention output that `o_proj` consumes.
"""

from collections.abc import Iterable, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Literal
import logging
logger = logging.getLogger(__name__)

try:
    import torch
    from llm_ablation.models import ModelBundle
except Exception as e:
    raise(e)

# weight ablation modes
# ZERO: all head activations set to zero
# MEAN: all activations set to the mean value
AblationMode = Literal["zero", "mean", "resample"]

# generate names for hook points at each of the 3 layers
# 1- Head level -> ablate output of attention heads
# 2- Neuron level -> ablate output of MLP layers
# 3- Block level -> ablate residual output of whole layer
def head_hook_point(layer: int) -> str:
    """Hook point name for the (self-attention o_proj) concatenated attention output of `layer`."""
    return f"layer{layer}.self_attn_o_proj"


def neuron_hook_point(layer: int) -> str:
    """Hook point name for the (mlp down_proj) MLP intermediate activation of `layer`."""
    return f"layer{layer}.mlp_activation_down_proj"


def layer_hook_point(layer: int) -> str:
    """Hook point name for the residual-stream value leaving `layer`."""
    return f"layer{layer}.layer_residual_stream"


# Mean and Resample ablation modes require a baseline activation distribution to
# calculate either the mean value or a resampled value.
# Cache the distribution in this dataclass
@dataclass
class ActivationCache:
    """
        Pooled activations collected by :func:`collect_activations`, used as
        the reference distribution for "mean" and "resample" ablation.
    """

    activations: dict[str, torch.Tensor] = field(default_factory=dict)

    # get the mean value from an activation layer values
    def mean(self, hook_point: str) -> torch.Tensor:
        """
            Mean activation (hidden_dim,) at `hook_point`, pooled over every
            token of every prompt that was passed to :func:`collect_activations`.
        """
        return self.activations[hook_point].mean(dim=0)

    # resample values from cached distribution
    def sample(self, hook_point: str) -> torch.Tensor:
        """
            A single real activation (hidden_dim,) drawn at random from the
            pool at `hook_point` — used for resample ablation.
        """
        pool = self.activations[hook_point]
        idx = int(torch.randint(0, pool.shape[0], (1,)).item())
        return pool[idx]

# translate the hook point name (a generic string) to a model layer index
def _resolve_hook_point(
    bundle: ModelBundle, hook_point: str
) -> tuple[torch.nn.Module, Literal["pre", "post"]]:
    # format is "layer_number.activation_hook_point"
    layer_str, _, suffix = hook_point.partition(".")
    # get layer number
    layer_idx = int(layer_str.removeprefix("layer"))
    # get the layer from the model chekpoint
    block = bundle.layer(layer_idx)

    # parse the mnemonic name, return the relevant layer slice
    if suffix == "self_attn_o_proj":
        return block.self_attn.o_proj, "pre"
    if suffix == "mlp_activation_down_proj":
        return block.mlp.down_proj, "pre"
    if suffix == "layer_residual_stream":
        return block, "post"
    raise ValueError(f"Unknown hook point: {hook_point!r}")

# run inference and collect activation values for caching distribution
# perform forward pass and collect activation values at every token position
def collect_activations(
    bundle: ModelBundle, prompts: Sequence[str], hook_points: Iterable[str]
) -> ActivationCache:
    """
        Run `prompts` through the model and pool the activations at each of
        `hook_points` across every token position, for use as an ablation
        reference distribution (mean/resample baselines).
    """
    hook_points = list(hook_points)
    pools: dict[str, list[torch.Tensor]] = {hp: [] for hp in hook_points}
    handles = []

    # register hooks for every hook point
    for hp in hook_points:
        # resolve pre/post hook type
        module, kind = _resolve_hook_point(bundle, hp)

        if kind == "pre":
            # prehook: run before forward() method
            def pre_hook(module, inputs, _hp=hp):
                tensor = inputs[0].detach()
                pools[_hp].append(tensor.reshape(-1, tensor.shape[-1]))

            # append handler
            handles.append(module.register_forward_pre_hook(pre_hook))
        else:
            # posthook: run after forward() method
            def post_hook(module, inputs, output, _hp=hp):
                tensor = output[0] if isinstance(output, tuple) else output
                tensor = tensor.detach()
                pools[_hp].append(tensor.reshape(-1, tensor.shape[-1]))

            # append handler
            handles.append(module.register_forward_hook(post_hook))

    # perform forward passes. hooks capture and save activation values
    # pool activations in [ntok, tok_dim] tensors per prompt
    # concatenate all in a final tensor before returning the activation distribution
    try:
        for prompt in prompts:
            inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
            with torch.no_grad():
                bundle.model(**inputs)
    finally:
        for handle in handles:
            handle.remove()

    # flatten pool tensor
    return ActivationCache(
        activations={hp: torch.cat(tensors, dim=0) for hp, tensors in pools.items()}
    )

# replace one activation/mlp/layer with new values
def _replacement(
    mode: AblationMode, hook_point: str, reference_cache: ActivationCache | None
) -> torch.Tensor | None:
    # Zero mode -> fills activations with zeros
    if mode == "zero":
        return None  # caller fills with zeros directly
    # Mean/Resample require reference Activation Distribution
    if reference_cache is None:
        raise ValueError(
            f"mode={mode!r} requires a reference_cache built with collect_activations()"
        )
    # return mean value or resampled value from cached distribution
    if mode == "mean":
        return reference_cache.mean(hook_point)
    if mode == "resample":
        return reference_cache.sample(hook_point)
    raise ValueError(f"unknown ablation mode: {mode!r}")


@contextmanager
def ablate_heads(
    bundle: ModelBundle,
    layer: int,
    heads: Sequence[int],
    mode: AblationMode = "zero",
    reference_cache: ActivationCache | None = None,
):
    """
        Ablate `heads` (0-indexed) in `layer`'s attention for the duration of
        the `with` block, by patching the input to that layer's `o_proj`.
    """
    hook_point = head_hook_point(layer)
    o_proj = bundle.layer(layer).self_attn.o_proj
    head_dim = bundle.head_dim
    slices = [slice(h * head_dim, (h + 1) * head_dim) for h in heads]
    replacement = _replacement(mode, hook_point, reference_cache)

    def pre_hook(module, inputs):
        hidden = inputs[0].clone()
        for sl in slices:
            hidden[..., sl] = 0.0 if replacement is None else replacement[sl].to(hidden.dtype)
        return (hidden, *inputs[1:])

    handle = o_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_neurons(
    bundle: ModelBundle,
    layer: int,
    neuron_idx: Sequence[int],
    mode: AblationMode = "zero",
    reference_cache: ActivationCache | None = None,
):
    """
        Ablate MLP intermediate neurons `neuron_idx` (0-indexed into the
        intermediate/hidden dimension) in `layer`, by patching the input to that
        layer's `mlp.down_proj`.
    """
    hook_point = neuron_hook_point(layer)
    down_proj = bundle.layer(layer).mlp.down_proj
    idx = torch.as_tensor(list(neuron_idx), dtype=torch.long)
    replacement = _replacement(mode, hook_point, reference_cache)

    def pre_hook(module, inputs):
        hidden = inputs[0].clone()
        hidden[..., idx] = 0.0 if replacement is None else replacement[idx].to(hidden.dtype)
        return (hidden, *inputs[1:])

    handle = down_proj.register_forward_pre_hook(pre_hook)
    try:
        yield
    finally:
        handle.remove()


@contextmanager
def ablate_layer(
    bundle: ModelBundle,
    layer: int,
    mode: AblationMode = "zero",
    reference_cache: ActivationCache | None = None,
):
    """
        Ablate the whole of `layer` for the duration of the `with` block.

        "zero" ablation means the layer contributes nothing: its output is
        replaced by its input, i.e. the block is skipped and the residual
        stream passes straight through. "mean"/"resample" instead replace the
        residual stream value leaving the layer with the mean/a resampled value
        from `reference_cache` (built over `layer_hook_point(layer)`).
    """
    hook_point = layer_hook_point(layer)
    block = bundle.layer(layer)
    replacement = _replacement(mode, hook_point, reference_cache)

    def post_hook(module, inputs, output):
        hidden_states = output[0] if isinstance(output, tuple) else output
        if replacement is None:
            new_hidden = inputs[0]  # pass the incoming residual stream straight through
        else:
            new_hidden = replacement.to(hidden_states.dtype).expand_as(hidden_states)
        if isinstance(output, tuple):
            return (new_hidden, *output[1:])
        return new_hidden

    handle = block.register_forward_hook(post_hook)
    try:
        yield
    finally:
        handle.remove()
