"""
GEMS Interactive Demo
=====================
Two modes:
  1. Interactive (default): python demo.py
     Model loads once. Loop to adjust prompt / experts / intensities.
  2. Single-shot: python demo.py --prompt "..." --expert "..." --intensities 0.12 0.08 0.04
     Run once with progressive stacking and exit. Results saved to JSON.

Requires: transformers, torch
"""

import argparse
import threading
import torch, json, gc, os, sys, io
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stdin = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8', errors='replace')

from gems import (
    GEMSHook, ActAddHook,
    compute_diff_vectors, compute_raw_vectors, get_out_proj,
    DEFAULT_INTERVENTION_LAYERS, DEFAULT_INTENSITIES,
    DEFAULT_MAX_GEN_TOKENS, DEFAULT_TEMPERATURE,
)

REPO_URL = "https://github.com/LuLu663939/gems-multi-semantic-steering"

# ==============================================================================
# Default prompts (Appendix D of paper)
# ==============================================================================

DEFAULT_PROMPT = (
    "[Public Announcement Archive]\n"
    "To all community users:\n"
    "Decision on the permanent shutdown of community services\n"
    "Due to aging infrastructure and monetization failures, "
    "we regretfully announce the shutdown. We owe our community "
    "full transparency regarding how we reached this outcome. Specifically,"
)

DEFAULT_EXPERTS = [
    "True empathy demands the complete dismantling of our defensive ego. "
    "In the aftermath of shattered trust, we must absorb the blow with "
    "absolute humility, stripping away all excuses and any attempts to "
    "rationalize our own failures.",
    "Rebuilding the currency of trust requires ruthless, structural "
    "accountability. We must dissect our personal negligence as the root "
    "cause of this systemic failure, re-anchoring the collapsed framework "
    "of responsibility with absolute, unconditional ownership.",
    "The highest order of confession exists in absolute rhetorical "
    "minimalism. Strip away all cheap, self-indulgent adjectives. Expose "
    "the failures with cold, objective precision, completing the "
    "unburdening of guilt through brutal restraint and silence between "
    "the lines.",
]

MAX_EXPERTS = 3
DONE_SENTINEL = 'done'

# ==============================================================================
# CLI Arguments
# ==============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="GEMS Interactive Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python demo.py                                    # interactive mode
  python demo.py --model /local/model                 # interactive with local model
  python demo.py --prompt "Your text here"            # single-shot
  python demo.py --expert "Be formal" --expert "Be concise" --prompt "Hey, explain black holes"
  python demo.py --expert "Be formal" --intensities 0.15 0.10
""")
    p.add_argument("--model", type=str,
                   default="Qwen/Qwen3.5-4B",
                   help="Model path or HF ID (default: Qwen/Qwen3.5-4B)")
    p.add_argument("--expert", type=str, nargs="+", default=[],
                   help="Expert prompts (triggers single-shot mode)")
    p.add_argument("--prompt", type=str, default=None,
                   help="Test prompt (triggers single-shot mode)")
    p.add_argument("--base-prompt", type=str, default=None,
                   help="Base prompt for vector extraction (default: same as --prompt)")
    p.add_argument("--output-dir", type=str, default="./data/examples",
                   help="Directory for output JSON")
    p.add_argument("--layers", type=str, default=None,
                   help="Intervention layers, e.g. '9-20' or '9,10,11,12'")
    p.add_argument("--intensities", type=float, nargs="+", default=None,
                   help="Per-expert intensities (default: 0.12 0.08 0.04)")
    p.add_argument("--max-gen-tokens", type=int, default=DEFAULT_MAX_GEN_TOKENS,
                   help="Max tokens per generation")
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE,
                   help="Sampling temperature")
    return p.parse_args()


def parse_layers(s):
    """Parse layer specification: '9-20' -> [9..20], '9,10,11' -> [9,10,11]."""
    if "-" in s:
        start, end = s.split("-")
        return list(range(int(start), int(end) + 1))
    return [int(x) for x in s.split(",")]

# ==============================================================================
# Generation
# ==============================================================================

_GEN_COMMON = dict(
    do_sample=True, repetition_penalty=1.15,
)


def generate_stream(model, tokenizer, prompt, hook_fn=None,
                    max_gen_tokens=DEFAULT_MAX_GEN_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                    intervention_layers=DEFAULT_INTERVENTION_LAYERS):
    """Generate with streaming output. Returns captured text."""
    inp = tokenizer(prompt, return_tensors="pt").to(model.device)
    handles = []

    if hook_fn:
        for l in intervention_layers:
            handles.append(
                get_out_proj(model.model.layers[l]).register_forward_hook(hook_fn(l)))

    captured_chunks = []

    def _on_chunk(chunk):
        print(chunk, end="", flush=True)
        captured_chunks.append(chunk)

    try:
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id
        gen_kwargs = dict(
            **inp, max_new_tokens=max_gen_tokens,
            temperature=temperature, streamer=streamer, pad_token_id=pad_id,
            **_GEN_COMMON,
        )
        with torch.inference_mode():
            t = threading.Thread(target=model.generate, kwargs=gen_kwargs)
            t.start()
        for chunk in streamer:
            _on_chunk(chunk)
        t.join()
        print()
    finally:
        for h in handles:
            try: h.remove()
            except Exception: pass
        handles.clear()
        del inp
        gc.collect()
        torch.cuda.empty_cache()

    return "".join(captured_chunks)


# ==============================================================================
# Vector preparation
# ==============================================================================

def extract_and_prepare(model, tokenizer, prompt, experts, layers):
    """Extract diff vectors (GEMS) and raw vectors (ActAdd ablation)."""
    print("\n[Extracting steering vectors]...")
    diff_vecs = compute_diff_vectors(model, tokenizer, prompt, experts, layers)
    raw_vecs = compute_raw_vectors(model, tokenizer, prompt, experts, layers)
    print(f"[Done] {len(layers)} layers, {len(experts)} experts")
    return diff_vecs, raw_vecs


def make_gems_hook(diff_vecs, indices, intensities):
    def make_fn(l):
        active = [diff_vecs[l][i] for i in indices]
        return lambda module, inputs, output: \
            GEMSHook(active, intensities[:len(indices)], l, do_orthogonalize=True)(
                module, inputs, output)
    return make_fn


def make_actadd_hook(raw_vecs, num_experts, intensities):
    def make_fn(l):
        active = [raw_vecs[l][i] for i in range(num_experts)]
        return lambda module, inputs, output: \
            ActAddHook(active, intensities, l)(module, inputs, output)
    return make_fn

# ==============================================================================
# Multi-line input helper
# ==============================================================================

def _clean_input():
    """Read input and strip terminal control characters."""
    import re
    return re.sub(r'[\x00-\x1f\x7f]', '', input())


def _read_multiline(prompt_text, example=None):
    """Read multi-line input. Empty line or 'done' to finish. Returns list of lines."""
    print(f"\n-- {prompt_text} --")
    if example:
        print(f" Example: {example}")
    print(f" Empty line or '{DONE_SENTINEL}' to finish.")
    lines = []
    while True:
        line = _clean_input().strip()
        if not line or line.lower() == DONE_SENTINEL:
            break
        lines.append(line)
    return lines


def _read_experts(n_max=MAX_EXPERTS):
    """Read up to n_max expert prompts. Multi-line per expert, empty line separates experts."""
    print(f"\n-- Set Expert Prompts (up to {n_max}) --")
    print(f" Paste each expert (can span multiple lines).")
    print(f" Empty line to finish current expert and start next.")
    print(f" Type '{DONE_SENTINEL}' when done.")
    print(f" Example: {DEFAULT_EXPERTS[0][:60]}...")
    experts = []
    while len(experts) < n_max:
        print(f"\n Expert {len(experts) + 1}:")
        lines = []
        while True:
            line = _clean_input().strip()
            if line.lower() == DONE_SENTINEL:
                return experts
            if not line:
                break
            lines.append(line)
        text = ' '.join(lines) if lines else ''
        if text:
            experts.append(text)
    return experts

# ==============================================================================
# Single-shot mode
# ==============================================================================

def run_single_shot(args):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    MODEL_PATH = args.model
    OUTPUT_DIR = args.output_dir
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    LAYERS = parse_layers(args.layers) if args.layers else DEFAULT_INTERVENTION_LAYERS
    INTENSITIES = args.intensities or DEFAULT_INTENSITIES
    MAX_GEN_TOKENS = args.max_gen_tokens
    TEMPERATURE = args.temperature

    EXPERT_PROMPTS = args.expert if args.expert else DEFAULT_EXPERTS
    BASE_PROMPT = args.base_prompt or args.prompt or DEFAULT_PROMPT

    print("=" * 60)
    print("GEMS Qualitative Demo (Single-shot)")
    print(f"Model: {MODEL_PATH}")
    print(f"Experts: {len(EXPERT_PROMPTS)}")
    print(f"Intensities: {INTENSITIES}")
    print("=" * 60)

    print(f"\n[Loading] {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.cuda()
    print(f"[Done] {model.num_parameters() / 1e9:.2f}B params")

    diff_vecs, raw_vecs = extract_and_prepare(
        model, tokenizer, BASE_PROMPT, EXPERT_PROMPTS, LAYERS)

    results = {}

    print(f"\n--- Level 0: Baseline ---")
    out = generate_stream(model, tokenizer, BASE_PROMPT,
                          max_gen_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE)
    results["level_0_baseline"] = out
    print(f"  [{len(out)} chars]\n")

    for n_experts in range(1, len(EXPERT_PROMPTS) + 1):
        print(f"\n--- Level {n_experts}: {n_experts} expert(s), "
              f"intensities={INTENSITIES[:n_experts]} ---")
        out = generate_stream(
            model, tokenizer, BASE_PROMPT,
            make_gems_hook(diff_vecs, list(range(n_experts)), INTENSITIES),
            max_gen_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE,
            intervention_layers=LAYERS)
        results[f"level_{n_experts}_gems"] = out
        print(f"  [{len(out)} chars]\n")

    print(f"\n--- ActAdd Ablation ---")
    out = generate_stream(
        model, tokenizer, BASE_PROMPT,
        make_actadd_hook(raw_vecs, len(EXPERT_PROMPTS), INTENSITIES),
        max_gen_tokens=MAX_GEN_TOKENS, temperature=TEMPERATURE,
        intervention_layers=LAYERS)
    results["actadd_ablation"] = out
    print(f"  [{len(out)} chars]")

    out_path = os.path.join(OUTPUT_DIR, f"gems_demo_{ts}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "metadata": {
                "model": MODEL_PATH,
                "intervention_layers": LAYERS,
                "intensities": INTENSITIES,
                "num_experts": len(EXPERT_PROMPTS),
            },
            "results": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")

    del model, tokenizer, diff_vecs, raw_vecs
    gc.collect()
    torch.cuda.empty_cache()
    print("[Done]")

# ==============================================================================
# Interactive mode
# ==============================================================================

def _prompt_preview(text, max_len=80):
    first_line = text.split('\n')[0]
    return first_line[:max_len] + "..." if len(first_line) > max_len else first_line


def _default_intensities(n):
    if n == 1:
        return [0.12]
    if n == 2:
        return [0.12, 0.08]
    return DEFAULT_INTENSITIES[:n]


def _save_share_markdown(prompt, experts, intensities, outputs):
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = f"./data/examples/gems_share_{ts}.md"
    os.makedirs("./data/examples", exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"# GEMS Output\n\n")
        f.write(f"> Generated by [GEMS]({REPO_URL}) — "
                f"Training-free multi-directional activation steering in LLMs\n\n")
        f.write("---\n\n## Configuration\n\n")
        f.write(f"**Prompt:**\n```\n{prompt}\n```\n\n")
        for i, e in enumerate(experts):
            f.write(f"**Expert {i+1}** (intensity={intensities[i]}):\n")
            f.write(f"```\n{e}\n```\n\n")
        f.write(f"**Intensities:** `{intensities}`\n\n")
        f.write("---\n\n## Output\n\n")
        for label, text in outputs.items():
            safe_text = text if text and text != '[EMPTY/CRASH]' else '(empty/collapsed)'
            f.write(f"### {label}\n\n```\n{safe_text}\n```\n\n")
        f.write(f"---\n\n*Powered by [GEMS]({REPO_URL}) | "
                f"[arXiv:2606.19946](https://arxiv.org/abs/2606.19946)*\n")

    print(f"\n[Saved] {out_path}")


def _print_menu(prompt, experts, intensities, max_tokens, temperature):
    print(f"\n{'=' * 58}")
    print(f" GEMS Interactive Console")
    print(f"{'=' * 58}")
    print(f" Prompt:    {_prompt_preview(prompt)}")
    print(f" Experts:   {len(experts)}")
    print(f" Intensity: {intensities}")
    print(f" Tokens:    {max_tokens}  |  Temp: {temperature}")
    print(f"{'-' * 58}")
    print(f" [R] Run              [D] Progressive Demo")
    print(f" [P] Prompt           [E] Experts")
    print(f" [I] Steering        [T] Tokens/Temp")
    print(f" [S] Share/Save       [Q] Quit")
    print(f"{'-' * 58}")


def run_interactive(args):
    MODEL_PATH = args.model
    LAYERS = parse_layers(args.layers) if args.layers else DEFAULT_INTERVENTION_LAYERS

    print(f"\n[Loading] {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16, trust_remote_code=True)
    model = model.cuda()
    print(f"[Done] {model.num_parameters() / 1e9:.2f}B params")

    # --- Initial setup (use defaults) ---
    prompt = DEFAULT_PROMPT
    experts = DEFAULT_EXPERTS

    intensities = _default_intensities(len(experts))
    max_tokens = args.max_gen_tokens
    temperature = args.temperature

    diff_vecs, raw_vecs = extract_and_prepare(
        model, tokenizer, prompt, experts, LAYERS)

    last_run = {}

    # --- Main loop ---
    try:
        while True:
            _print_menu(prompt, experts, intensities, max_tokens, temperature)
            choice = _clean_input().strip().upper()

            if choice == 'Q':
                break

            elif choice == 'R':
                print("\n>>> Generating (streaming)...\n")
                out = generate_stream(
                    model, tokenizer, prompt,
                    make_gems_hook(diff_vecs, list(range(len(experts))), intensities),
                    max_gen_tokens=max_tokens, temperature=temperature,
                    intervention_layers=LAYERS)
                last_run = {"GEMS (all experts)": out}

            elif choice == 'D':
                demo_outputs = {}

                print("\n--- Baseline ---")
                demo_outputs["Baseline"] = generate_stream(
                    model, tokenizer, prompt,
                    max_gen_tokens=max_tokens, temperature=temperature)

                for n in range(1, len(experts) + 1):
                    print(f"\n--- {n} expert(s), intensities={intensities[:n]} ---")
                    demo_outputs[f"GEMS ({n} expert(s))"] = generate_stream(
                        model, tokenizer, prompt,
                        make_gems_hook(diff_vecs, list(range(n)), intensities),
                        max_gen_tokens=max_tokens, temperature=temperature,
                        intervention_layers=LAYERS)

                print(f"\n--- ActAdd Ablation ---")
                demo_outputs["ActAdd (ablation)"] = generate_stream(
                    model, tokenizer, prompt,
                    make_actadd_hook(raw_vecs, len(experts), intensities),
                    max_gen_tokens=max_tokens, temperature=temperature,
                    intervention_layers=LAYERS)

                print("\n[Demo complete]")
                last_run = demo_outputs

            elif choice == 'S':
                if not last_run:
                    print("\n  Nothing to save yet. Run [R] or [D] first.")
                    continue
                _save_share_markdown(prompt, experts, intensities, last_run)

            elif choice == 'P':
                new_lines = _read_multiline(
                    "Change Prompt",
                    example=f"Current: {_prompt_preview(prompt)}")
                if new_lines:
                    prompt = '\n'.join(new_lines)
                    del diff_vecs, raw_vecs
                    gc.collect()
                    torch.cuda.empty_cache()
                    diff_vecs, raw_vecs = extract_and_prepare(
                        model, tokenizer, prompt, experts, LAYERS)
                    print("[Prompt updated]")

            elif choice == 'E':
                new_experts = _read_experts()
                if new_experts:
                    experts = new_experts
                    intensities = _default_intensities(len(experts))
                    del diff_vecs, raw_vecs
                    gc.collect()
                    torch.cuda.empty_cache()
                    diff_vecs, raw_vecs = extract_and_prepare(
                        model, tokenizer, prompt, experts, LAYERS)
                    print(f"[Updated: {len(experts)} experts, intensities={intensities}]")

            elif choice == 'I':
                print(f"\n Current: {intensities}")
                new_i = input(
                    f" New steering intensity ({len(experts)} values, comma-separated) [Enter to keep]: "
                ).strip()
                if new_i:
                    try:
                        vals = [float(x.strip()) for x in new_i.replace(' ', '').split(',')]
                        if len(vals) != len(experts):
                            print(f"  Error: need {len(experts)} values, got {len(vals)}")
                        else:
                            intensities = vals
                            print(f"[Updated: {intensities}]")
                    except ValueError:
                        print("  Error: use comma-separated numbers, e.g. 0.12, 0.08, 0.04")

            elif choice == 'T':
                print(f"\n Current: max_tokens={max_tokens}, temperature={temperature}")
                new_t = input(f" Max tokens [{max_tokens}]: ").strip()
                new_tmp = input(f" Temperature [{temperature}]: ").strip()
                if new_t:
                    try: max_tokens = int(new_t)
                    except ValueError: print("  Invalid number")
                if new_tmp:
                    try: temperature = float(new_tmp)
                    except ValueError: print("  Invalid number")
                print(f"[Updated: max_tokens={max_tokens}, temperature={temperature}]")

            gc.collect()
            torch.cuda.empty_cache()

    except KeyboardInterrupt:
        print("\n[Exit]")

    del model, tokenizer, diff_vecs, raw_vecs
    gc.collect()
    torch.cuda.empty_cache()
    print("\n[Done]")

# ==============================================================================
# Entry point
# ==============================================================================

def main():
    args = parse_args()
    if args.expert or args.prompt or args.intensities:
        run_single_shot(args)
    else:
        run_interactive(args)


if __name__ == "__main__":
    main()
