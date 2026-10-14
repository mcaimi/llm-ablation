import pytest

from llm_ablation.models import ModelBundle, load_model

TINY_MODEL = "yujiepan/qwen2-tiny-random"


@pytest.fixture(scope="session")
def tiny_bundle() -> ModelBundle:
    """A real-but-tiny Qwen2 model with random weights (2 layers, 4 attention
    heads, 2 KV heads i.e. GQA, hidden_size=8): fast to download, small
    enough for CI, and structurally identical to the full-size model
    (`self_attn.o_proj`, `mlp.down_proj`, ...) that `hooks.py` relies on."""
    return load_model(TINY_MODEL, device="cpu")
