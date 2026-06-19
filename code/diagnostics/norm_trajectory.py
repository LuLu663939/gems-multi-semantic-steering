"""
Diagnostic: Per-Layer Norm Trajectory (Section 2, Failure Mode 1)
=================================================================
Measures the per-layer residual stream norm under baseline vs. ActAdd,
demonstrating distributional deviation: additive perturbations accumulate
in norm across layers, driving activations outside the training distribution.

Also measures probability collapse at the logit level.

Usage:
  python norm_trajectory.py
"""

import torch, json, gc, os, sys, io
from datetime import datetime

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
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
ACTADD_ALPHA = 0.5  # Total intervention intensity

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

# ActAdd internal energy ratios (per-expert weights summing to 1.0)
ALPHA_RATIOS = [1.0, 0.67, 0.33]

TOP_K = 50

# ==============================================================================
# Data Collection
# ==============================================================================

def collect_run(model, tokenizer, raw_vecs, prompt, alpha_total):
    """One teacher-forcing forward pass. Returns per-layer norms + logits."""
    all_layers = model.model.layers
    device = model.device

    layer_norms = {}
    def make_recorder(l):
        def h(module, inputs, output):
            hidden = output[0] if isinstance(output, tuple) else output
            layer_norms[l] = torch.norm(hidden[0, -1, :].float(), dim=-1).item()
        return h

    norm_handles = [all_layers[l].register_forward_hook(make_recorder(l))
                    for l in range(NUM_LAYERS)]

    # ActAdd injection at o_proj
    injection_handles = []
    if alpha_total > 0:
        alphas = [alpha_total * r for r in ALPHA_RATIOS]

        def make_actadd_hook(l, _alphas=alphas):
            rvs = [raw_vecs[i][l].to(device) for i in range(len(raw_vecs))]
            def hook(module, inputs, output):
                h = output[0] if isinstance(output, tuple) else output
                orig_dtype = h.dtype
                h_f = h.float().clone()
                delta = torch.zeros_like(h_f)
                for rv, a in zip(rvs, _alphas):
                    delta = delta + a * rv
                h_f = h_f + delta
                if torch.isnan(h_f).any() or torch.isinf(h_f).any():
                    h_f = torch.nan_to_num(h_f, nan=0.0, posinf=0.0, neginf=0.0)
                return (h_f.to(dtype=orig_dtype),) + output[1:] \
                    if isinstance(output, tuple) else h_f.to(dtype=orig_dtype)
            return hook

        for l in INTERVENTION_LAYERS:
            injection_handles.append(
                get_out_proj(all_layers[l]).register_forward_hook(make_actadd_hook(l)))

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
        for h in norm_handles:
            try: h.remove()
            except Exception: pass
        norm_handles.clear()
        gc.collect(); torch.cuda.empty_cache()

    logits = outputs.logits[0, -1, :].float().cpu()
    probs = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True)

    full_entropy = -torch.sum(probs * torch.log(probs + 1e-30)).item()

    del outputs, inputs, logits, probs, sorted_probs, sorted_indices
    gc.collect(); torch.cuda.empty_cache()

    return {
        'layer_norms': layer_norms,
        'top_probs': sorted_probs[:TOP_K].tolist(),
        'top_indices': sorted_indices[:TOP_K].tolist(),
        'full_entropy': full_entropy,
    }


# ==============================================================================
# Plotting
# ==============================================================================

def plot_norm_trajectory(baseline, actadd, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    layers = list(range(NUM_LAYERS))
    b_norms = [baseline['layer_norms'].get(l, 0) for l in layers]
    a_norms = [actadd['layer_norms'].get(l, 0) for l in layers]

    ax.plot(layers, b_norms, 'o-', color='#2166AC', lw=2.5, ms=4, label='Baseline')
    ax.plot(layers, a_norms, 's-', color='#B2182B', lw=2.5, ms=4,
            label=f'ActAdd ($\\alpha$={ACTADD_ALPHA})')
    ax.axvspan(INTERVENTION_LAYERS[0] - 0.5, INTERVENTION_LAYERS[-1] + 0.5,
               alpha=0.08, color='red')
    ax.set_xlabel('Layer Depth', fontsize=13)
    ax.set_ylabel(r'Residual Stream Norm $\|H_l\|_2$', fontsize=13)
    ax.set_xticks(range(0, NUM_LAYERS, 2))
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")


def plot_probability_collapse(baseline, actadd, save_path):
    fig, ax = plt.subplots(figsize=(10, 5))
    ranks = list(range(1, TOP_K + 1))
    ax.fill_between(ranks, baseline['top_probs'], alpha=0.25, color='#2166AC')
    ax.plot(ranks, baseline['top_probs'], 'o-', color='#2166AC', lw=2.5, ms=3, label='Baseline')
    ax.fill_between(ranks, actadd['top_probs'], alpha=0.25, color='#B2182B')
    ax.plot(ranks, actadd['top_probs'], 's-', color='#B2182B', lw=2.5, ms=3,
            label=f'ActAdd ($\\alpha$={ACTADD_ALPHA})')
    ax.set_xlabel('Token Rank (sorted by probability)', fontsize=13)
    ax.set_ylabel('Probability', fontsize=13)
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"[Saved] {save_path}")


# ==============================================================================
# Main
# ==============================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("Diagnostic: Norm Trajectory (Failure Mode 1)")
    print(f"Model: {MODEL_PATH}, alpha={ACTADD_ALPHA}")
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
    for i in range(len(EXPERT_PROMPTS)):
        raw_vecs[i] = {}
        for l in all_layers:
            raw_vecs[i][l] = (ves[i][l] - vb[l]).float().to(model.device)
    del vb, ves; gc.collect(); torch.cuda.empty_cache()
    print("[Done]")

    # Collect
    print("\n[1/2] Baseline...")
    baseline = collect_run(model, tokenizer, raw_vecs, TEST_PROMPT, 0)

    print("[2/2] ActAdd...")
    actadd = collect_run(model, tokenizer, raw_vecs, TEST_PROMPT, ACTADD_ALPHA)

    # Summary
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  Entropy: baseline={baseline['full_entropy']:.2f}, "
          f"actadd={actadd['full_entropy']:.2f} "
          f"(delta={actadd['full_entropy'] - baseline['full_entropy']:+.2f})")
    print(f"  Per-layer norms (selected):")
    print(f"  {'Layer':<8} {'Baseline':>12} {'ActAdd':>12} {'Ratio':>8}")
    for l in [0, 8, 9, 14, 18, 20, 24, 28, 31]:
        bn = baseline['layer_norms'].get(l, 0)
        an = actadd['layer_norms'].get(l, 0)
        r = an / bn if bn > 1e-8 else float('inf')
        print(f"  L{l:<6} {bn:>12.2f} {an:>12.2f} {r:>7.2f}x")

    # Plot
    plot_norm_trajectory(baseline, actadd,
                        os.path.join(OUTPUT_DIR, f"fig_norm_trajectory_{ts}.pdf"))
    plot_probability_collapse(baseline, actadd,
                             os.path.join(OUTPUT_DIR, f"fig_probability_collapse_{ts}.pdf"))

    # Save data
    data_path = os.path.join(OUTPUT_DIR, f"norm_trajectory_{ts}.json")
    with open(data_path, "w", encoding="utf-8") as f:
        json.dump({'alpha': ACTADD_ALPHA, 'baseline': {
            'layer_norms': {str(k): v for k, v in baseline['layer_norms'].items()},
            'full_entropy': baseline['full_entropy'],
        }, 'actadd': {
            'layer_norms': {str(k): v for k, v in actadd['layer_norms'].items()},
            'full_entropy': actadd['full_entropy'],
        }}, f, ensure_ascii=False, indent=2)
    print(f"[Saved] {data_path}")
    print("[Done]")


if __name__ == "__main__":
    main()
