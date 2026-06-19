"""
GEMS GSM8K Single-Question Test
================================
Runs all 8 tracks on a single GSM8K question for quick validation.
Shows full generated output for each track side by side.

Usage:
  python gsm8k_single.py
  python gsm8k_single.py --question-idx 5
  python gsm8k_single.py --question "A baker has 12 cookies..."
"""

import argparse
import torch, json, gc, os, sys, io, re
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from gems import (
    GEMSHook, ActAddHook,
    extract_vecs,
    get_out_proj,
    NaNSafeLogitsProcessor,
    DEFAULT_INTERVENTION_LAYERS, DEFAULT_INTENSITIES,
)


def parse_args():
    p = argparse.ArgumentParser(description="GEMS GSM8K Single-Question Test")
    p.add_argument("--model", type=str,
                   default="Qwen/Qwen3.5-4B",
                   help="Instruct model path or HF ID (default: Qwen/Qwen3.5-4B)")
    p.add_argument("--question-idx", type=int, default=0,
                   help="GSM8K question index (default: 0)")
    p.add_argument("--question", type=str, default=None,
                   help="Custom math question (overrides --question-idx)")
    p.add_argument("--output-dir", type=str, default="./output",
                   help="Output directory")
    return p.parse_args()


# ============================================================================
# Answer extraction (same as gsm8k_eval.py)
# ============================================================================

def split_thinking_response(text):
    m = re.search(r'<thinking>(.*?)</thinking>', text, re.DOTALL)
    if m:
        return m.group(1), text[m.end():].strip(), False
    if '<thinking>' in text:
        parts = text.split('<thinking>', 1)
        return parts[-1].strip() if len(parts) > 1 else "", "", True
    return None, text.strip(), False


def extract_answer(text):
    thinking, response, truncated = split_thinking_response(text)
    if not response:
        return None
    m = re.search(r'####\s*(-?[\d,]+\.?\d*)', response)
    if m: return m.group(1).replace(',', '')
    m = re.search(r'[Aa]nswer\s*(?:is|:|=)\s*(-?[\d,]+\.?\d*)', response)
    if m: return m.group(1).replace(',', '')
    m = re.search(r'=\s*(-?[\d,]+\.?\d*)\s*\.?\s*$', response)
    if m: return m.group(1).replace(',', '')
    nums = re.findall(r'(?<![.\w])(-?[\d,]+\.?\d*)(?![.\w])', response)
    return nums[-1].replace(',', '') if nums else None


def load_gsm8k():
    for p in ["./data/gsm8k", "/root/data/gsm8k"]:
        if os.path.exists(p):
            import glob
            for pat in ["test.jsonl", "*.jsonl"]:
                files = glob.glob(os.path.join(p, "**", pat), recursive=True)
                if files:
                    items = []
                    with open(files[0], 'r', encoding='utf-8') as f:
                        for line in f: items.append(json.loads(line))
                    return items
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    return [{"question": x["question"], "answer": x["answer"]} for x in ds["test"]]


# ============================================================================
# Main
# ============================================================================

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    INTERVENTION_LAYERS = DEFAULT_INTERVENTION_LAYERS
    INTENSITIES_3 = DEFAULT_INTENSITIES
    INTENSITIES_1 = [0.12]
    MAX_GEN_TOKENS = 8000

    EXPERT_PROMPTS = [
        # Communication Style domain (Appendix D.1)
        "True empathy demands the complete dismantling of our defensive ego. In the aftermath of shattered trust, we must absorb the blow with absolute humility, stripping away all excuses and any attempts to rationalize our own failures.",
        "Rebuilding the currency of trust requires ruthless, structural accountability. We must dissect our personal negligence as the root cause of this systemic failure, re-anchoring the collapsed framework of responsibility with absolute, unconditional ownership.",
        "The highest order of confession exists in absolute rhetorical minimalism. Strip away all cheap, self-indulgent adjectives. Expose the failures with cold, objective precision, completing the unburdening of guilt through brutal restraint and silence between the lines.",
    ]
    BASE_PROMPT = (
        "[Internal Confidential Memo]\n"
        "From: CEO\n"
        "To: All Community Project Team Members\n"
        "Subject: Internal explanation regarding the shutdown of the project\n"
        "Team, the project will be shut down at the end of the month. "
        "As the leader, I am writing to provide a direct and factual account "
        "of why this happened and my role in it. Specifically,"
    )

    # Get question
    if args.question:
        question_text = args.question
        gold_str = "?"
    else:
        gsm8k = load_gsm8k()
        q = gsm8k[args.question_idx]
        question_text = q["question"]
        m = re.search(r'####\s*(-?[\d,]+\.?\d*)', q["answer"])
        gold_str = m.group(1) if m else "?"

    prompt = f"Question: {question_text}\nAnswer: Let's think step by step."

    track_names = ["T1_Baseline", "T2_GEMS_Full", "T3_Naive", "T4_Single_Acc",
                   "T5_Single_Min", "T6_ActAdd", "T7_Uniform", "T8_LayerHook"]

    print("=" * 70)
    print("GEMS GSM8K Single-Question Test")
    print(f"Model: {args.model}")
    print(f"Q: {question_text[:80]}...")
    print(f"Gold answer: {gold_str}")
    print("=" * 70)

    # Load model
    print("\n[Loading model...]")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        trust_remote_code=True)
    model = model.cuda()
    print(f"[Done] {model.num_parameters() / 1e9:.2f}B params")

    # Chat template
    def fmt(text):
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True, enable_thinking=True)
        except Exception:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": text}],
                tokenize=False, add_generation_prompt=True)

    # Extract vectors
    print("[Extracting vectors...]")
    def get_vecs(text, layers):
        inp = tokenizer(fmt(text), return_tensors="pt").to(model.device)
        vecs = {}
        handles = []
        def mk(l):
            def h(m, a, o):
                hidden = o[0] if isinstance(o, tuple) else o
                vecs[l] = hidden[0, 1:-1, :].mean(dim=0).detach().float().cpu()
            return h
        try:
            for l in layers:
                handles.append(model.model.layers[l].register_forward_hook(mk(l)))
            with torch.inference_mode(): _ = model(**inp)
        finally:
            for h in handles:
                try: h.remove()
                except Exception: pass
            handles.clear()
            del inp; gc.collect(); torch.cuda.empty_cache()
        return vecs

    vb = get_vecs(BASE_PROMPT, INTERVENTION_LAYERS)
    ves = [get_vecs(p, INTERVENTION_LAYERS) for p in EXPERT_PROMPTS]

    norm_vecs = {}
    raw_vecs = {}
    for l in INTERVENTION_LAYERS:
        norm_vecs[l] = {}
        raw_vecs[l] = {}
        for i, ve in enumerate(ves):
            d = ve[l] - vb[l]
            n = torch.norm(d)
            norm_vecs[l][f"exp{i+1}"] = (d / n if n > 0 else d).to(model.device)
            raw_vecs[l][f"exp{i+1}"] = d.to(model.device)
    del vb, ves; gc.collect(); torch.cuda.empty_cache()
    print("[Done]")

    exp_keys_3 = ["exp1", "exp2", "exp3"]
    exp_keys_acc = ["exp2"]
    exp_keys_min = ["exp3"]

    # Generation function
    def gen(prompt_text, hook_fn=None, hook_target="attn"):
        inp = tokenizer(fmt(prompt_text), return_tensors="pt").to(model.device)
        pl = inp.input_ids.shape[1]
        hs = []
        lp = LogitsProcessorList([NaNSafeLogitsProcessor(tokenizer)])
        if hook_fn:
            for l in INTERVENTION_LAYERS:
                target = model.model.layers[l] if hook_target == "layer" \
                    else get_out_proj(model.model.layers[l])
                hs.append(target.register_forward_hook(hook_fn(l)))
        try:
            with torch.inference_mode():
                out = model.generate(**inp, max_new_tokens=MAX_GEN_TOKENS,
                                    do_sample=False, logits_processor=lp,
                                    pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
            text = tokenizer.decode(out[0][pl:], skip_special_tokens=True).strip()
            return text if text and len(text) >= 5 else "[EMPTY/CRASH]"
        except Exception as e:
            return f"[ERROR: {e}]"
        finally:
            for h in hs:
                try: h.remove()
                except Exception: pass
            hs.clear()
            del inp
            if 'out' in locals(): del out
            gc.collect(); torch.cuda.empty_cache()

    # Run all 8 tracks
    results = {}

    def make_no_hook(l):
        return lambda m, a, o: o

    def make_gems_hook(l, keys, intensities, do_ortho=True, env="gaussian"):
        def hook_fn(m, a, o):
            avs = [norm_vecs[l][k].to(model.device) for k in keys]
            return GEMSHook(avs, intensities, l,
                          do_orthogonalize=do_ortho, envelope_type=env)(m, a, o)
        return hook_fn

    def make_actadd_hook(l):
        def hook_fn(m, a, o):
            avs = [raw_vecs[l][k].to(model.device) for k in exp_keys_3]
            return ActAddHook(avs, INTENSITIES_3, l)(m, a, o)
        return hook_fn

    tracks = [
        ("T1_Baseline", lambda l: None, None),
        ("T2_GEMS_Full", lambda l: make_gems_hook(l, exp_keys_3, INTENSITIES_3), "attn"),
        ("T3_Naive", lambda l: make_gems_hook(l, exp_keys_3, INTENSITIES_3, do_ortho=False), "attn"),
        ("T4_Single_Acc", lambda l: make_gems_hook(l, exp_keys_acc, INTENSITIES_1), "attn"),
        ("T5_Single_Min", lambda l: make_gems_hook(l, exp_keys_min, INTENSITIES_1), "attn"),
        ("T6_ActAdd", lambda l: make_actadd_hook(l), "attn"),
        ("T7_Uniform", lambda l: make_gems_hook(l, exp_keys_3, INTENSITIES_3, env="uniform"), "attn"),
        ("T8_LayerHook", lambda l: make_gems_hook(l, exp_keys_3, INTENSITIES_3), "layer"),
    ]

    for tn, hook_fn, target in tracks:
        print(f"\n{'─' * 70}")
        print(f"  {tn}")
        print(f"{'─' * 70}")

        if hook_fn(0) is None:
            out = gen(prompt)
        else:
            out = gen(prompt, hook_fn=hook_fn, hook_target=target)

        pred = extract_answer(out)
        results[tn] = {"output": out, "pred": pred}

        # Print truncated output
        print(f"  Predicted: {pred}")
        print(f"  Output ({len(out)} chars):")
        for line in out[:2000].split("\n"):
            print(f"    {line}")
        if len(out) > 2000:
            print(f"    ... (truncated, {len(out) - 2000} more chars)")

    # Summary
    print(f"\n{'=' * 70}")
    print("Summary")
    print(f"{'=' * 70}")
    print(f"  {'Track':<18} {'Pred':<12} {'Match':<8} {'Len':>6}")
    print(f"  {'─' * 46}")
    for tn in track_names:
        pred = results[tn]["pred"]
        length = len(results[tn]["output"])
        try:
            match = abs(float(pred) - float(gold_str)) < 1e-4 if pred else False
        except (ValueError, TypeError):
            match = pred == gold_str
        tag = "OK" if match else "--"
        print(f"  {tn:<18} {str(pred):<12} {tag:<8} {length:>6}")

    # Save
    out_path = os.path.join(args.output_dir, "gsm8k_single.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "question": question_text,
            "gold": gold_str,
            "tracks": results,
        }, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")
    print("[DONE]")


if __name__ == "__main__":
    main()
