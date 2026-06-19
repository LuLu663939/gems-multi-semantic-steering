"""Vector extraction utilities for GEMS."""

import torch, gc
from gems.utils import get_out_proj


def extract_vecs(model, tokenizer, text, layers, pooling="mean"):
    """Extract per-layer hidden states for a single prompt.

    Args:
        model: CausalLM model.
        tokenizer: Tokenizer.
        text: Prompt text (raw, not yet tokenized).
        layers: List of layer indices to hook.
        pooling: ``"mean"`` for mean-pooled positions 1..-1 (recommended for
                 generation), ``"last"`` for last position only (teacher-forcing).

    Returns:
        Dict mapping layer index -> Tensor of shape (hidden_dim,).
    """
    inp = tokenizer(text, return_tensors="pt").to(model.device)
    vecs = {}
    handles = []

    def _make_hook(l):
        def hook_fn(m, a, o):
            hidden = o[0] if isinstance(o, tuple) else o
            if pooling == "last":
                vecs[l] = hidden[0, -1, :].detach().float().cpu()
            else:
                vecs[l] = hidden[0, 1:-1, :].mean(dim=0).detach().float().cpu()
        return hook_fn

    try:
        for l in layers:
            handles.append(model.model.layers[l].register_forward_hook(_make_hook(l)))
        with torch.inference_mode():
            _ = model(**inp)
    finally:
        for h in handles:
            try:
                h.remove()
            except Exception:
                pass
        handles.clear()
        del inp
        gc.collect()
        torch.cuda.empty_cache()
    return vecs


def compute_diff_vectors(model, tokenizer, base_prompt, expert_prompts, layers):
    """Compute normalized diff vectors: v_i = Normalize(expert_i - base).

    Args:
        model: CausalLM model.
        tokenizer: Tokenizer.
        base_prompt: Base prompt for vector subtraction.
        expert_prompts: List of expert prompt strings.
        layers: List of layer indices.

    Returns:
        Dict mapping layer index -> list of unit diff Tensors (on model device).
    """
    vb = extract_vecs(model, tokenizer, base_prompt, layers)
    ves = [extract_vecs(model, tokenizer, p, layers) for p in expert_prompts]

    diff_vecs = {}
    for l in layers:
        diff_vecs[l] = []
        for ve in ves:
            d = ve[l] - vb[l]
            n = torch.norm(d)
            diff_vecs[l].append((d / n if n > 0 else d).to(model.device))

    del vb, ves
    gc.collect()
    torch.cuda.empty_cache()
    return diff_vecs


def compute_raw_vectors(model, tokenizer, base_prompt, expert_prompts, layers):
    """Compute raw (un-normalized) diff vectors: v_i = expert_i - base.

    Returns:
        Dict mapping layer index -> list of raw diff Tensors (on model device).
    """
    vb = extract_vecs(model, tokenizer, base_prompt, layers)
    ves = [extract_vecs(model, tokenizer, p, layers) for p in expert_prompts]

    raw_vecs = {}
    for l in layers:
        raw_vecs[l] = []
        for ve in ves:
            raw_vecs[l].append((ve[l] - vb[l]).to(model.device))

    del vb, ves
    gc.collect()
    torch.cuda.empty_cache()
    return raw_vecs
