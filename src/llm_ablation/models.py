"""
Shared model load/accelerator detector code.

Import this to load a checkpoint from disk or from the network (HF)

Also, model structure display/plot code belong here
"""

from typing import Any
from dataclasses import dataclass
import platform

import logging
logger = logging.getLogger(__name__)

try:
    # pytorch and datatypes
    import torch
    from torch import float16, float32

    # macos shenanigans
    import torch.cuda as tc
    if platform.system() == "Darwin":
        import torch.backends.mps as tmps

    # Huggingface transformers
    from transformers import (
        AutoModelForCausalLM,
        AutoTokenizer,
        PreTrainedModel,
        PreTrainedTokenizerBase,
    )
except Exception as e:
    raise(e)

# default demo model: ~1GB footprint, enough to be run locally on basically anything
DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"

# accelerator class
@dataclass
class AcceleratorDevice:
    """ represents a supported/discovered device """
    
    accelerator: str
    datatype: torch.dtype

    @property
    def resolved_device(self):
        return self.accelerator

    @property
    def resolved_dtype(self):
        if self.datatype in (float16, float32):
            return self.datatype
        else:
            return float32 # default

# determine accelerator and data type
def get_accelerator_device():
    accelerator = "cpu"
    dtype = float32

    logger.info("Checking for the availability of a GPU...")
    if tc.is_available():
        device_name = tc.get_device_name()
        device_capabilities = tc.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in tc.mem_get_info()]
        logger.info("CUDA GPU available: %s - %s - %.1f/%.1f GB VRAM",
                     device_name, device_capabilities, device_available_mem, device_total_mem)
        accelerator = "cuda"
        dtype = float16
    elif platform.system() == "Darwin" and tmps.is_available():
        device_name = tmps.get_name()
        device_cores = tmps.get_core_count()
        logger.info("Apple MPS available: %s - %d Cores", device_name, device_cores)
        accelerator = "mps"
        dtype = float16

    return accelerator, dtype

@dataclass
class ModelBundle:
    """A loaded model + tokenizer plus the architecture metadata the ablation
    utilities need (number of layers/heads, head dimension)."""

    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    name: str
    device: str

    @property
    def n_layers(self) -> int:
        return self.model.config.num_hidden_layers

    @property
    def n_heads(self) -> int:
        return self.model.config.num_attention_heads

    @property
    def head_dim(self) -> int:
        hidden_size = self.model.config.hidden_size
        return hidden_size // self.n_heads

    def layer(self, index: int):
        """Return the transformer block module for `index`."""
        return self.model.model.layers[index]

    def generate(self, prompt: str, max_new_tokens: int = 64, **kwargs: Any) -> str:
        """
        Chat-format `prompt`, generate, and return only the newly generated text.
        """

        messages = [{"role": "user", "content": prompt}]
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)

        kwargs.setdefault("do_sample", False)

        # perform inference, no gradient accumulation
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs,
            )

        # decode sequence
        new_tokens = output_ids[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

# load a model from specific checkpoint
def load_model(
    name: str = DEFAULT_MODEL, device: str | None = None, dtype: str = "auto"
) -> ModelBundle:
    """Load a causal LM + tokenizer and wrap them in a :class:`ModelBundle`.

    `name` can be any causal LM on the Hugging Face Hub with a standard
    decoder-only architecture (Qwen2, Llama, Gemma2, ...); the ablation hooks
    in :mod:`llm_ablation.hooks` rely only on the common `model.model.layers[i]`
    / `self_attn.o_proj` / `mlp` module structure that these share.
    """

    # resolve accelerator device
    accelerator, dtype = get_accelerator_device()
    resolved_device: AcceleratorDevice = AcceleratorDevice(accelerator=accelerator, datatype=dtype)
    
    # load tokenizer and model weights
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=dtype)

    # move to accelerator (if any is discovered)
    model.to(resolved_device.accelerator)
    model.eval()
    return ModelBundle(model=model, tokenizer=tokenizer, name=name, device=resolved_device.accelerator)
