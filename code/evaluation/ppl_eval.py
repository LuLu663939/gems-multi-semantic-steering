"""
GEMS Wikitext-2 PPL Evaluation: 6-Track Comparison
=====================================================
Reproduces Table 4 in the paper. Measures language modeling quality
under multi-directional steering via teacher-forcing perplexity.

6 tracks:
  A0: No hooks (true baseline)
  A1: GEMS hook with zero energies (infrastructure overhead control)
  B:  GEMS Full (ortho + envelope + decay + norm)
  C:  GEMS Naive (no orthogonalization)
  E:  GEMS without envelope (E=1.0, tau=1.0)
  D:  ActAdd pure (raw vectors, no constraints)

Usage:
  python ppl_eval.py
"""

import torch, math, json, gc, os, sys, io, time
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from gems import (
    GEMSHook, ActAddHook,
    extract_vecs,
    get_out_proj,
    DEFAULT_INTERVENTION_LAYERS, DEFAULT_INTENSITIES,
)

# ==============================================================================
# CONFIGURATION — Hardcoded paper defaults (do not modify for reproduction)
# ==============================================================================

MODEL_PATH = "Qwen/Qwen3.5-4B-Base"         # Base model (auto-download)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVENTION_LAYERS = DEFAULT_INTERVENTION_LAYERS

NUM_SAMPLES = 100
CHUNK_LENGTH = 128

# Expert prompts (paper defaults — do not modify for reproduction)
# Communication Style domain (Appendix D.1, used for PPL evaluation)
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

INTERVENTION_INTENSITIES = DEFAULT_INTENSITIES

# ==============================================================================
# PPL Computation
# ==============================================================================

def compute_ppl_no_hook(input_ids_list):
    total_loss = 0.0
    total_tokens = 0
    for ids in input_ids_list:
        with torch.inference_mode():
            outputs = model(ids)
        shift_logits = outputs.logits[:, :-1, :].contiguous()
        shift_labels = ids[:, 1:].contiguous()
        loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                shift_labels.view(-1), reduction='none')
        total_loss += loss.sum().item()
        total_tokens += loss.numel()
    return math.exp(total_loss / max(1, total_tokens))


def compute_ppl_with_hook(input_ids_list, hook_class, hook_kwargs_list):
    total_loss = 0.0
    total_tokens = 0
    for idx, ids in enumerate(input_ids_list):
        handles = []
        try:
            for l_idx, l in enumerate(INTERVENTION_LAYERS):
                e, ortho, uniform = hook_kwargs_list[l_idx]
                if hook_class == "gems":
                    env_type = "uniform" if uniform else "gaussian"
                    hook = GEMSHook(
                        expert_vectors=layer_unit_vecs[l],
                        intensities=e, layer_idx=l,
                        do_orthogonalize=ortho,
                        envelope_type=env_type,
                        teacher_forcing=True)
                else:
                    hook = ActAddHook(
                        raw_vectors=layer_raw_vecs[l], intensities=e,
                        layer_idx=l, teacher_forcing=True)
                handles.append(get_out_proj(model.model.layers[l]).register_forward_hook(hook))
            with torch.inference_mode():
                outputs = model(ids)
            shift_logits = outputs.logits[:, :-1, :].contiguous()
            shift_labels = ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                    shift_labels.view(-1), reduction='none')
            total_loss += loss.sum().item()
            total_tokens += loss.numel()
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            handles.clear()
        if (idx + 1) % 10 == 0:
            print(f"    [{idx + 1}/{len(input_ids_list)}]", flush=True)
    return math.exp(total_loss / max(1, total_tokens))


# ==============================================================================
# Main
# ==============================================================================

def main():
    global model

    print("=" * 60)
    print("GEMS Wikitext-2 PPL 6-Track Evaluation")
    print(f"Model: {MODEL_PATH}")
    print("=" * 60)

    print("\n[1/4] Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.cuda()
    model.eval()
    all_layers = model.model.layers
    print(f"  {len(all_layers)} layers, hidden_dim={model.config.hidden_size}")

    # Extract vectors (use last-position pooling for teacher-forcing)
    print("[2/4] Extracting vectors...")
    vb = extract_vecs(model, tokenizer, BASE_PROMPT, INTERVENTION_LAYERS, pooling="last")
    ves = [extract_vecs(model, tokenizer, p, INTERVENTION_LAYERS, pooling="last")
           for p in EXPERT_PROMPTS]

    layer_unit_vecs = {}
    layer_raw_vecs = {}
    for l in INTERVENTION_LAYERS:
        layer_unit_vecs[l] = []
        layer_raw_vecs[l] = []
        for i, ve in enumerate(ves):
            d = ve[l] - vb[l]
            n = torch.norm(d)
            layer_unit_vecs[l].append(d / n if n > 0 else d)
            layer_raw_vecs[l].append(d)
    del vb, ves; gc.collect()
    print("  [Done]")

    noop_e = [0.0, 0.0, 0.0]
    exp_e = INTERVENTION_INTENSITIES

    # Load Wikitext-2
    print("[3/4] Loading Wikitext-2...")
    from datasets import load_dataset
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts_raw = [item['text'] for item in dataset if item['text'].strip()]
    valid_texts = []
    for text in texts_raw:
        if not text.strip():
            continue
        enc = tokenizer(text, return_tensors="pt", truncation=True,
                        max_length=CHUNK_LENGTH + 1)
        if enc.input_ids.shape[1] >= 32:
            valid_texts.append(text)
    n_samples = min(NUM_SAMPLES, len(valid_texts))
    all_input_ids = [tokenizer(t, return_tensors="pt", truncation=True,
                             max_length=CHUNK_LENGTH + 1).input_ids.to(model.device)
                     for t in valid_texts[:n_samples]]
    total_tok = sum(ids.shape[1] - 1 for ids in all_input_ids)
    print(f"  {n_samples} samples, {total_tok} tokens")

    # Precompute hook kwargs
    gems_noop_kwargs = [(noop_e, True, False) for _ in INTERVENTION_LAYERS]
    gems_full_kwargs = [(exp_e, True, False) for _ in INTERVENTION_LAYERS]
    gems_naive_kwargs = [(exp_e, False, False) for _ in INTERVENTION_LAYERS]
    gems_noenv_kwargs = [(exp_e, True, True) for _ in INTERVENTION_LAYERS]
    actadd_kwargs = [(exp_e, None, None) for _ in INTERVENTION_LAYERS]

    # Run
    print("\n[4/4] Computing PPL...")

    print("  A0 (No hooks)...", end=" ", flush=True)
    t0 = time.time()
    a0 = compute_ppl_no_hook(all_input_ids)
    print(f"PPL={a0:.4f} ({time.time()-t0:.1f}s)")

    print("  A1 (Hook E=[0,0,0])...", end=" ", flush=True)
    t1 = time.time()
    a1 = compute_ppl_with_hook(all_input_ids, "gems", gems_noop_kwargs)
    print(f"PPL={a1:.4f} ({time.time()-t1:.1f}s)")

    print("  B (GEMS Full)...", end=" ", flush=True)
    t2 = time.time()
    b = compute_ppl_with_hook(all_input_ids, "gems", gems_full_kwargs)
    print(f"PPL={b:.4f} ({time.time()-t2:.1f}s)")

    print("  C (No ortho)...", end=" ", flush=True)
    t3 = time.time()
    c = compute_ppl_with_hook(all_input_ids, "gems", gems_naive_kwargs)
    print(f"PPL={c:.4f} ({time.time()-t3:.1f}s)")

    print("  E (No envelope)...", end=" ", flush=True)
    t4 = time.time()
    e = compute_ppl_with_hook(all_input_ids, "gems", gems_noenv_kwargs)
    print(f"PPL={e:.4f} ({time.time()-t4:.1f}s)")

    print("  D (ActAdd)...", end=" ", flush=True)
    t5 = time.time()
    d = compute_ppl_with_hook(all_input_ids, "actadd", actadd_kwargs)
    print(f"PPL={d:.4f} ({time.time()-t5:.1f}s)")

    # Summary
    print(f"\n{'='*60}")
    print("Results")
    print(f"{'='*60}")
    print(f"  A0 (No hooks):   PPL = {a0:.4f}")
    print(f"  A1 (Infra ctrl):  PPL = {a1:.4f}  d={a1-a0:+.4f}")
    print(f"  B  (GEMS Full): PPL = {b:.4f}  d={b-a0:+.4f}  ({(b/a0-1)*100:+.2f}%)")
    print(f"  C  (No ortho):  PPL = {c:.4f}  d={c-a0:+.4f}")
    print(f"  E  (No env):    PPL = {e:.4f}  d={e-a0:+.4f}")
    print(f"  D  (ActAdd):    PPL = {d:.4f}  d={d-a0:+.4f}")
    print(f"\n  Orthogonalization effect (B-C): {b-c:+.4f}")
    print(f"  Envelope effect (E-B):        {e-b:+.4f}")
    print(f"  ActAdd vs GEMS (D-B):       {d-b:+.4f}")
    print(f"{'='*60}")

    # Save
    out_path = os.path.join(OUTPUT_DIR, "wikitext2_ppl_6track.json")
    with open(out_path, 'w') as f:
        json.dump({
            "A0": a0, "A1": a1, "B": b, "C": c, "E": e, "D": d,
            "num_samples": n_samples,
            "intervention_layers": INTERVENTION_LAYERS,
            "intensities": exp_e,
        }, f, indent=2)
    print(f"\n[Saved] {out_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
