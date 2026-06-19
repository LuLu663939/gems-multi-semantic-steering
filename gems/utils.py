"""Shared utilities for GEMS."""

import torch
from transformers import LogitsProcessor


# ============================================================================
# Default hyperparameters (paper defaults for Qwen3.5-4B, 32 layers)
# ============================================================================

DEFAULT_PEAK_LAYER = 14
DEFAULT_SIGMA = 3.0
DEFAULT_COSINE_DECAY_START = 15
DEFAULT_COSINE_DECAY_SPAN = 6.0
DEFAULT_INTERVENTION_LAYERS = list(range(9, 21))
DEFAULT_INTENSITIES = [0.12, 0.08, 0.04]
DEFAULT_MAX_GEN_TOKENS = 1500
DEFAULT_TEMPERATURE = 0.1


# ============================================================================
# Architecture helpers
# ============================================================================

def get_out_proj(layer):
    """Find the attention output projection module (works across architectures).

    Supports both standard self-attention (``self_attn.o_proj``) and linear
    attention (``linear_attn.out_proj``) modules.
    """
    mods = layer._modules
    if "linear_attn" in mods:
        return mods["linear_attn"]._modules.get("out_proj")
    elif "self_attn" in mods:
        return mods["self_attn"]._modules.get("o_proj")
    raise RuntimeError(f"No attn out_proj found in layer: {list(mods.keys())}")


# ============================================================================
# NaN safety
# ============================================================================

class NaNSafeLogitsProcessor(LogitsProcessor):
    """Detects NaN/Inf in logits and forces EOS generation."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, input_ids, scores):
        if torch.isnan(scores).any() or torch.isinf(scores).any():
            scores = torch.nan_to_num(scores, nan=-1e4, posinf=-1e4, neginf=-1e4)
            eos_id = self.tokenizer.eos_token_id
            if isinstance(eos_id, list):
                scores[:, eos_id[0]] = 1e4
            elif eos_id is not None:
                scores[:, eos_id] = 1e4
        return scores
