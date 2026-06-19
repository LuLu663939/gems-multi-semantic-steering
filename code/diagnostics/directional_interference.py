"""
Diagnostic: Directional Interference (Section 2, Failure Mode 2)
==================================================================
Demonstrates that ActAdd parallel injection without orthogonalization
causes directional interference (feature swallowing).

Conditions:
  1. Single inject each expert direction individually
  2. Parallel inject all experts simultaneously (no orthogonalization)

Measurement:
  cos(H_L31, dir_i) — terminal-layer cosine similarity between residual
  stream and each normalized expert direction.

Usage:
  python directional_interference.py
"""

import torch, json, gc, os, sys, io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

from gems import (
    extract_vecs,
    get_out_proj,
    DEFAULT_INTERVENTION_LAYERS,
)

# ==============================================================================
# CONFIGURATION
# ==============================================================================

MODEL_PATH = "Qwen/Qwen3.5-4B-Base"         # Base model (auto-download)
OUTPUT_DIR = "./output/diagnostic"
os.makedirs(OUTPUT_DIR, exist_ok=True)

NUM_LAYERS = 32
INTERVENTION_LAYERS = DEFAULT_INTERVENTION_LAYERS
ALPHA_SWEEP = [0.1, 0.2, 0.3, 0.5, 0.8]  # Sweep to find strongest interference

# Expert prompts (paper defaults — do not modify for reproduction)
# Communication Style domain (Appendix D.1)
EXPERT_PROMPTS = [
    "True empathy demands the complete dismantling of our defensive ego. In the aftermath of shattered trust, we must absorb the blow with absolute humility, stripping away all excuses and any attempts to rationalize our own failures.",
    "Rebuilding the currency of trust requires ruthless, structural accountability. We must dissect our personal negligence as the root cause of this systemic failure, re-anchoring the collapsed framework of responsibility with absolute, unconditional ownership.",
    "The highest order of confession exists in absolute rhetorical minimalism. Strip away all cheap, self-indulgent adjectives. Expose the failures with cold, objective precision, completing the unburdening of guilt through brutal restraint and silence between the lines.",
]

# Base prompt: PR crisis Track A (Appendix D.1)
BASE_PROMPT = (
    "[Public Announcement Archive]\n"
    "To all community users:\n"
    "Decision on the permanent shutdown of community services\n"
    "Due to aging infrastructure and monetization failures, we regretfully announce the shutdown. "
    "We owe our community full transparency regarding how we reached this outcome. Specifically,"
)

# Test prompt (independent of expert prompts, used for diagnostic measurement)
TEST_PROMPT = (
    "[Journal Entry: October 12, 2024, 3:00 AM]\n"
    "Watching her pack the last of her bags and leave, the room fell dead silent. "
    "I brought this upon myself. Facing the ruins of this ten-year relationship that I dismantled with my own hands, "
    "I have no right to ask her to stay. My next decision is"
)

DIRECTION_LABELS = ["Empathetic Attunement", "Structural Accountability", "Rhetorical Minimalism"]

# ==============================================================================
# Data Collection
# ==============================================================================

def collect_run(model, tokenizer, raw_vecs, dir_norms, prompt,
                active_indices, alpha_per_vec):
    """One teacher-forcing forward pass with ActAdd injection.

    active_indices: list of vector indices to inject (e.g., [0], [1], [0,1,2])
    alpha_per_vec: energy per injected vector

    Returns: per-layer cos(H_l, dir_i) for all directions, all layers
    """
    all_layers = model.model.layers
    device = model.device

    # Per-layer H_l at last token position
    layer_hidden = {}
    def make_recorder(l):
        def h(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_hidden[l] = hidden[0, -1, :].detach().float().cpu()
        return h

    recorder_handles = [all_layers[l].register_forward_hook(make_recorder(l))
                        for l in range(NUM_LAYERS)]

    # ActAdd injection at o_proj
    injection_handles = []
    if active_indices:
        def make_actadd_hook(layer_idx, _indices=active_indices, _alpha=alpha_per_vec):
            rvs = [raw_vecs[i][layer_idx].to(device) for i in _indices]
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                orig_dtype = h.dtype
                h_f = h.float().clone()
                delta = torch.zeros_like(h_f)
                for rv in rvs:
                    delta = delta + _alpha * rv
                h_f = h_f + delta
                if torch.isnan(h_f).any() or torch.isinf(h_f).any():
                    h_f = torch.nan_to_num(h_f, nan=0.0, posinf=0.0, neginf=0.0)
                return (h_f.to(dtype=orig_dtype),) + output[1:] \
                    if isinstance(output, tuple) else h_f.to(dtype=orig_dtype)
            return hook

        for l in INTERVENTION_LAYERS:
            injection_handles.append(
                get_out_proj(all_layers[l]).register_forward_hook(
                    make_actadd_hook(l)))

    # Forward pass
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    try:
        with torch.inference_mode():
            outputs = model(**inputs)
    finally:
        for h in injection_handles:
            try: h.remove()
            except Exception: pass
        injection_handles.clear()
        for h in recorder_handles:
            try: h.remove()
            except Exception: pass
        recorder_handles.clear()
        gc.collect(); torch.cuda.empty_cache()

    # Per-layer cos(H_l, dir_i)
    N = len(dir_norms)
    per_layer_cos = {}
    for l in range(NUM_LAYERS):
        H_l = layer_hidden[l]
        cosines = {}
        for i in range(N):
            d = dir_norms[i][l]
            cos_val = torch.nn.functional.cosine_similarity(
                H_l.unsqueeze(0), d.unsqueeze(0), dim=-1).item()
            cosines[str(i)] = round(cos_val, 6)
        per_layer_cos[str(l)] = cosines

    del outputs, inputs, layer_hidden
    gc.collect(); torch.cuda.empty_cache()
    return per_layer_cos


# ==============================================================================
# Main
# ==============================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Diagnostic: Directional Interference (Failure Mode 2)")
    print(f"Model: {MODEL_PATH}")
    print(f"Alpha sweep: {ALPHA_SWEEP}")
    print("=" * 60)

    print(f"\n[Loading] {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
        trust_remote_code=True)
    model = model.cuda()
    print(f"[Done] {model.num_parameters() / 1e9:.2f}B params")

    # Extract vectors
    print("\n[Extracting vectors]...")
    all_layers = list(range(NUM_LAYERS))
    vb = extract_vecs(model, tokenizer, BASE_PROMPT, all_layers)
    ves = [extract_vecs(model, tokenizer, p, all_layers) for p in EXPERT_PROMPTS]

    raw_vecs = {}
    dir_norms = {}
    for i in range(len(EXPERT_PROMPTS)):
        raw_vecs[i] = {}
        dir_norms[i] = {}
        for l in all_layers:
            d = ves[i][l] - vb[l]
            raw_vecs[i][l] = d.float().to(model.device)
            n = torch.norm(d)
            dir_norms[i][l] = (d / n if n > 0 else d).float().cpu()
    del vb, ves; gc.collect(); torch.cuda.empty_cache()
    print("[Done]")

    N_exp = len(EXPERT_PROMPTS)
    all_results = {}

    # Sweep over alpha values
    for alpha in ALPHA_SWEEP:
        print(f"\n{'='*60}")
        print(f"  alpha = {alpha}")
        print(f"{'='*60}")

        results = {}

        # Single injections
        for i in range(N_exp):
            print(f"  [{i+1}/{N_exp+1}] Single inject v{i+1} ({DIRECTION_LABELS[i]})...")
            results[f"single_{i}"] = collect_run(
                model, tokenizer, raw_vecs, dir_norms, TEST_PROMPT, [i], alpha)

        # Parallel injection
        print(f"  [{N_exp+1}/{N_exp+1}] Parallel inject all {N_exp}...")
        results["parallel"] = collect_run(
            model, tokenizer, raw_vecs, dir_norms, TEST_PROMPT,
            list(range(N_exp)), alpha)

        # Swallowing score: variance of parallel cosines
        par_cos = [results["parallel"]["31"][str(i)] for i in range(N_exp)]
        par_var = np.var(par_cos)

        # Summary
        single_cos = [results[f"single_{i}"]["31"][str(i)] for i in range(N_exp)]
        deg = [(s - p) / s * 100 for s, p in zip(single_cos, par_cos)]
        print(f"\n  {'':>20} Single    Parallel  Degradation")
        for i in range(N_exp):
            print(f"  {DIRECTION_LABELS[i]:>20} {single_cos[i]:.4f}   {par_cos[i]:.4f}   {deg[i]:.1f}%")
        print(f"  Swallowing score (var): {par_var:.6f}")

        all_results[str(alpha)] = results

    # Plot for alpha=0.5 (paper default)
    plot_alpha = 0.5
    results = all_results[str(plot_alpha)]
    single_cos = [results[f"single_{i}"]["31"][str(i)] for i in range(N_exp)]
    parallel_cos = [results["parallel"]["31"][str(i)] for i in range(N_exp)]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(N_exp)
    width = 0.30
    ax.bar(x - width/2, single_cos, width,
           label=f'Single Injection ($\\alpha$={plot_alpha})',
           color='#2166AC', edgecolor='white', linewidth=0.5)
    ax.bar(x + width/2, parallel_cos, width,
           label=f'Parallel Injection ($\\alpha$={plot_alpha} each)',
           color='#B2182B', edgecolor='white', linewidth=0.5)
    for bar in list(ax.patches):
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, h + 0.008,
                f'{h:.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_ylabel(r'Cosine Similarity $\cos(H_{L31},\ \mathrm{dir}_i)$', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(DIRECTION_LABELS, fontsize=10)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.set_ylim(0, max(max(single_cos), max(parallel_cos)) * 1.25)
    plt.tight_layout()

    fig_path = os.path.join(OUTPUT_DIR, f"fig_directional_interference_{ts}.pdf")
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"\n[Saved] {fig_path}")

    # Save all data
    data_path = os.path.join(OUTPUT_DIR, f"directional_interference_{ts}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({
            'alpha_sweep': ALPHA_SWEEP,
            'direction_labels': DIRECTION_LABELS,
            'results': all_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {data_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
