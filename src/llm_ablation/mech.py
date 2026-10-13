"""
    Mechanistic ablation experiments: 
    sweeping ablation over a task and summarizing the result.

    "head" mode: ablate activation at the attention layer level
    "neuron" mode: ablate activation at the MLP neuron level
    "layer" mode: ablate post activation at the whole layer level
"""

from collections.abc import Sequence
import logging
logger = logging.getLogger(__name__)

try:
    import numpy as np
    import torch

    # import from local libraries
    from llm_ablation.hooks import AblationMode, ActivationCache, ablate_heads
    from llm_ablation.models import ModelBundle
except Exception as e:
    raise(e)

# ablation example sequences (prompt, correct, incorrect)
TaskPairs = Sequence[tuple[str, str, str]]

# generate token id for a specific word
def _first_token_id(bundle: ModelBundle, word: str) -> int:
    ids = bundle.tokenizer.encode(" " + word.strip(), add_special_tokens=False)
    if not ids:
        raise ValueError(f"could not tokenize {word!r}")
    return ids[0]

# calculate logits diff between predicted correct tok and incorrect tok
def logit_diff(bundle: ModelBundle, prompt: str, correct_token: str, incorrect_token: str) -> float:
    """
        performs logit(correct_token) - logit(incorrect_token) at the next-token position.
    
        measure difference between correct prediction and incorrect prediction
    """
    # encode prompt via tokenizer
    inputs = bundle.tokenizer(prompt, return_tensors="pt").to(bundle.device)
    logger.info(f"Tokenizer: inputs: {inputs}")

    # without gradient accumulation: perform inference
    with torch.no_grad():
        logits = bundle.model(**inputs).logits[0, -1]
    logger.info(f"Model inference: {prompt} -> {logits}")

    # calculate difference
    correct_id = _first_token_id(bundle, correct_token)
    incorrect_id = _first_token_id(bundle, incorrect_token)
    logger.info(f"Correct ID {correct_id}, Incorrect ID {incorrect_id}")

    return float((logits[correct_id] - logits[incorrect_id]).item())

# run an ablation experiment @layer/head level
def head_ablation_sweep(
    bundle: ModelBundle,
    prompts: TaskPairs,
    mode: AblationMode = "zero",
    reference_cache: ActivationCache | None = None,
) -> np.ndarray:
    """
        [n_layers x n_heads] matrix of the mean logit-diff *drop*
        (baseline - ablated) caused by ablating each individual head,
        averaged over `prompts`.
    """
    # the model dimensions, as per checkpoint
    n_layers, n_heads = bundle.n_layers, bundle.n_heads

    # calculate baseline logit diff with all heads unchanged
    baseline = np.array([logit_diff(bundle, p, c, i) for p, c, i in prompts])

    # [layer x heads] matrix filled with zeros (will hold ablation results per layer)
    drops = np.zeros((n_layers, n_heads))

    # iterate over every model layer
    for layer in range(n_layers):
        # now for every head in the layer
        for head in range(n_heads):
            # perform head ablation
            with ablate_heads(bundle, layer, [head], mode=mode, reference_cache=reference_cache):
                # recompute logit diff with ablated head
                ablated = np.array([logit_diff(bundle, p, c, i) for p, c, i in prompts])
            
            # measure and save ablation run result for this layer
            drops[layer, head] = (baseline - ablated).mean()
    
    # return drop matrix
    return drops

# factual recall experiment: perform Head Ablation in respect to Domain Knowledge Tasks
def run_factual_recall_task(bundle: ModelBundle, pairs: TaskPairs, mode: str = "zero") -> dict:
    """
        Baseline logit-diffs plus the zero-ablation head sweep for a
        factual-recall style task: identify heads that contribute to a
        specific response for a given prompt domain.
    """
    # compute baseline with no head ablation
    baseline_diffs = [logit_diff(bundle, p, c, i) for p, c, i in pairs]

    # run ablation sweep
    sweep = head_ablation_sweep(bundle, pairs, mode=mode)

    # return task metadata
    return {
        "ablation_mode": mode,
        "baseline_logit_diffs": baseline_diffs,
        "mean_baseline_logit_diff": float(np.mean(baseline_diffs)),
        "head_ablation_drop": sweep,
    }

# determine the top `k` heads that contribute to a specific response
def top_heads(drop_matrix: np.ndarray, k: int = 5) -> list[tuple[int, int, float]]:
    """
        The `k` (layer, head) pairs with the largest logit-diff drop in `drop_matrix`.
    """
    # sort the flat matrix in descending order (index of biggest element first).
    # then take the first `k` indices
    flat_idx = np.argsort(drop_matrix, axis=None, descending=True)[:k]

    # get the 2D coordinates of the `k` (layer, head) pairs from the drop matrix
    layers, heads = np.unravel_index(flat_idx, drop_matrix.shape)

    # return data
    return [
        (int(layer), int(head), float(drop_matrix[layer, head]))
        for layer, head in zip(layers, heads, strict=True)
    ]

def run_ioi_style_task(bundle: ModelBundle, templates: TaskPairs) -> dict:
    """
        Same shape as `run_factual_recall_task`, for IOI-style templates.

        Exploratory: the classic IOI circuit was reverse-engineered on
        GPT-2-small. A small instruction-tuned model isn't guaranteed to show an
        equally clean pattern — treat this as "let's go look," not a reproduction.
    """
    return run_factual_recall_task(bundle, templates)
