# GEMS Agent Integration Guide

For AI agents that want to use GEMS as a library. Setup and usage in [README.md](README.md). Data guide in [REPRODUCE.md](REPRODUCE.md).

## Pipeline (3 steps)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from gems import GEMSHook, ActAddHook, compute_diff_vectors, get_out_proj

# 1. Load model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3.5-4B", torch_dtype=torch.float16, trust_remote_code=True).cuda()
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)

# 2. Extract steering vectors from prompt pairs
#    layers, peak, sigma, decay — all model-specific, see "Model-Specific Defaults" below
layers = list(range(9, 21))  # Qwen3.5-4B (32 layers): L9-L20
diff_vecs = compute_diff_vectors(model, tokenizer, base_prompt="neutral text", expert_prompts=["direction A", "direction B"], layers=layers)
# Returns: {layer_idx: [unit_tensor_A, unit_tensor_B, ...], ...}

# 3. Register hooks and generate
intensities = [0.12, 0.08]  # paper default, model-agnostic starting point
handles = []
for l in layers:
    def make_hook(l):
        vecs = diff_vecs[l]
        return lambda module, inputs, output: GEMSHook(vecs, intensities, l, do_orthogonalize=True)(module, inputs, output)
    handles.append(get_out_proj(model.model.layers[l]).register_forward_hook(make_hook(l)))

# Generate normally — hooks intercept automatically
output = model.generate(**tokenizer("your prompt", return_tensors="pt").to("cuda"), max_new_tokens=1500)

# Cleanup
for h in handles: h.remove()
```

## Model-Specific Defaults

The envelope and decay parameters inside `GEMSHook` are **hardcoded to Qwen3.5-4B** (32 layers). If you use a different model, you **must** override them in `gems/utils.py` or pass your own envelope logic.

| Parameter | Qwen3.5-4B value | What it controls | How to adapt for other models |
|-----------|-----------------|-----------------|------------------------------|
| `DEFAULT_PEAK_LAYER` | 14 | Gaussian envelope center | Set to `total_layers * 0.44` (roughly middle) |
| `DEFAULT_SIGMA` | 3.0 | Gaussian envelope width | Scale proportionally to layer count |
| `DEFAULT_COSINE_DECAY_START` | 15 | Where cosine decay begins | Set to `peak_layer + 1` |
| `DEFAULT_COSINE_DECAY_SPAN` | 6.0 | Cosine decay duration | Scale to cover ~4-6 layers |
| `DEFAULT_INTERVENTION_LAYERS` | 9-20 (12 layers) | Which layers to hook | Cover ~35-40% of total layers, centered around peak |

Paper cross-model mappings (no per-layer optimization, proportional only):

| Model | Total Layers | Intervention Layers | Peak | Decay Start | Decay Span |
|-------|-------------|-------------------|------|-------------|-----------|
| Qwen3.5-4B | 32 | L9-L20 | L14 | L15 | 6.0 |
| Llama-3.2-3B | 28 | L12-L23 | L19 | L19 | 4.0 |
| Qwen3.6-27B | 64 | L34-L50 | L40 | L40 | 6.0 |
| Gemma-4-31B | 60 | L40-L55 | L48 | L48 | 6.0 |

**Intensities `[0.12, 0.08, 0.04]` are model-agnostic.** They control the norm-preservation budget (sum of squares must be < 1.0) and don't depend on model architecture. Start with these and adjust.

## API

### `compute_diff_vectors(model, tokenizer, base_prompt, expert_prompts, layers)`

Extracts normalized steering vectors from contrastive prompt pairs.

- `base_prompt` (str): neutral prompt
- `expert_prompts` (list[str]): declarative statements defining each steering direction
- `layers` (list[int]): which layers to extract from (**model-specific**)
- Returns: `dict[int, list[Tensor]]` — layer index to list of unit vectors on GPU

### `compute_raw_vectors(model, tokenizer, base_prompt, expert_prompts, layers)`

Same as above but returns un-normalized diff vectors (for ActAdd baseline).

### `GEMSHook(expert_vectors, intensities, layer_idx, ...)`

Forward-pass hook. Returns a callable for `register_forward_hook`.

| Parameter | Type | Default | Model-specific? | Description |
|-----------|------|---------|----------------|-------------|
| `expert_vectors` | list[Tensor] | — | No | Unit direction vectors (on GPU). One per expert. |
| `intensities` | list[float] | — | No | Per-expert strength. Sum of squares must be < 1.0. |
| `layer_idx` | int | — | No | Current layer index (passed to envelope/decay internally). |
| `do_orthogonalize` | bool | `True` | No | Gram-Schmidt orthogonalization. Without it, directions interfere. |
| `envelope_type` | str | `"gaussian"` | No | `"gaussian"` or `"uniform"` (E=1.0 everywhere). |
| `teacher_forcing` | bool | `False` | No | If True, operates on all token positions (for PPL eval). |

Note: `layer_idx` is used to compute the Gaussian envelope and cosine decay internally via `DEFAULT_PEAK_LAYER`, `DEFAULT_SIGMA`, etc. These **defaults are Qwen3.5-4B values**. For other models, modify `gems/utils.py` before importing.

### `ActAddHook(raw_vectors, intensities, layer_idx, teacher_forcing=False)`

Pure addition baseline. No norm preservation, no orthogonalization. Collapses under multi-direction injection.

### `get_out_proj(layer)`

Finds the attention output projection in a transformer layer. Works with both standard attention (`self_attn.o_proj`) and linear attention (`linear_attn.out_proj`). Returns the module to hook.

## Intensity Guide

Intensities control steering strength per expert. Total across all experts:

| Total | Effect |
|-------|--------|
| 0.05–0.25 | Subtle, safe |
| 0.25–0.45 | Strong, clear style shifts |
| > 0.45 | Risk of collapse or artifacts |

Paper defaults: `[0.12, 0.08, 0.04]` (total 0.24). Model-agnostic starting point.

## Constraints

- **Must hook `o_proj`**, not the full layer output. Full-layer hook corrupts the MLP pathway and causes immediate collapse.
- **Vectors must be unit-normalized** for GEMSHook. Use `compute_diff_vectors` (which normalizes), not `compute_raw_vectors`.
- **Layer range and envelope parameters are model-specific.** Qwen3.5-4B defaults are hardcoded in `gems/utils.py`. For other models, modify these values before use.
- Generation mode (default) skips prefill automatically. Teacher-forcing mode operates on all positions.
