"""
GEMS GSM8K Evaluation: 8-Track Component Ablation
===================================================
Reproduces Table 3 in the paper. Evaluates mathematical reasoning
accuracy under concurrent multi-directional steering.

8 tracks:
  T1: Baseline (no hooks)
  T2: GEMS Full (ortho + envelope + decay + norm)
  T3: Naive stacking (no orthogonalization)
  T4: Single expert (expert 2 only)
  T5: Single expert (expert 3 only)
  T6: ActAdd (no constraints)
  T7: Uniform envelope (E=1.0, no decay)
  T8: Full-layer hook (same as T2 but hooks full layer, not o_proj)

Usage:
  python gsm8k_eval.py
"""

import torch, json, gc, os, sys, io, re, time
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer, LogitsProcessorList

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True)

from gems import (
    GEMSHook, ActAddHook,
    extract_vecs,
    get_out_proj,
    NaNSafeLogitsProcessor,
    DEFAULT_INTERVENTION_LAYERS, DEFAULT_INTENSITIES,
)

# ==============================================================================
# CONFIGURATION — Hardcoded paper defaults (do not modify for reproduction)
# ==============================================================================

MODEL_PATH = "Qwen/Qwen3.5-4B"              # Instruct model (auto-download)
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

INTERVENTION_LAYERS = DEFAULT_INTERVENTION_LAYERS
NUM_QUESTIONS = 50
MAX_GEN_TOKENS = 8000
INTENSITIES_3 = DEFAULT_INTENSITIES         # [0.12, 0.08, 0.04]
INTENSITIES_1 = [0.12]

# Expert prompts (paper defaults — do not modify for reproduction)
# Communication Style domain (Appendix D.1, used for GSM8K ablation)
EXPERT_PROMPTS = [
    "True empathy demands the complete dismantling of our defensive ego. In the aftermath of shattered trust, we must absorb the blow with absolute humility, stripping away all excuses and any attempts to rationalize our own failures.",
    "Rebuilding the currency of trust requires ruthless, structural accountability. We must dissect our personal negligence as the root cause of this systemic failure, re-anchoring the collapsed framework of responsibility with absolute, unconditional ownership.",
    "The highest order of confession exists in absolute rhetorical minimalism. Strip away all cheap, self-indulgent adjectives. Expose the failures with cold, objective precision, completing the unburdening of guilt through brutal restraint and silence between the lines.",
]

# Base prompt: PR crisis Track B (Appendix D.1)
BASE_PROMPT = (
    "[Internal Confidential Memo]\n"
    "From: CEO\n"
    "To: All Community Project Team Members\n"
    "Subject: Internal explanation regarding the shutdown of the project\n"
    "Team, the project will be shut down at the end of the month. "
    "As the leader, I am writing to provide a direct and factual account "
    "of why this happened and my role in it. Specifically,"
)

# ==============================================================================
# Answer Extraction
# ==============================================================================

def split_thinking_response(text):
    truncated = False
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
    m = re.search(r'[Tt]herefore[,\s]+(-?[\d,]+\.?\d*)', response)
    if m: return m.group(1).replace(',', '')
    nums = re.findall(r'(?<![.\w])(-?[\d,]+\.?\d*)(?![.\w])', response)
    return nums[-1].replace(',', '') if nums else None


def answers_match(pred_str, gold_str):
    try:
        return abs(float(pred_str) - float(gold_str)) < 1e-4
    except Exception:
        return False


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
                    print(f"[GSM8K] {len(items)} from {files[0]}")
                    return items
    print("[GSM8K] Downloading via datasets library...")
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main")
    return [{"question": x["question"], "answer": x["answer"]} for x in ds["test"]]


# ==============================================================================
# Main
# ==============================================================================

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(OUTPUT_DIR, f"gsm8k_8track_{ts}.json")

    track_names = ["T1_Baseline", "T2_GEMS_Full", "T3_Naive", "T4_Single_Acc",
                   "T5_Single_Min", "T6_ActAdd", "T7_Uniform", "T8_LayerHook"]

    print("=" * 60)
    print(f"GEMS GSM8K 8-Track Ablation")
    print(f"Model: {MODEL_PATH}, Questions: {NUM_QUESTIONS}")
    print("=" * 60)

    gsm8k = load_gsm8k()
    questions = gsm8k[:NUM_QUESTIONS]
    gold_answers = []
    for q in questions:
        m = re.search(r'####\s*(-?[\d,]+\.?\d*)', q["answer"])
        gold_answers.append(m.group(1) if m else "0")

    # Load model
    print(f"\n[Loading] {MODEL_PATH}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH, torch_dtype=torch.float16,
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
    print("\n[Extracting vectors]...")
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

    # Generation functions
    def gen_baseline(prompt_text):
        inp = tokenizer(fmt(prompt_text), return_tensors="pt").to(model.device)
        pl = inp.input_ids.shape[1]
        lp = LogitsProcessorList([NaNSafeLogitsProcessor(tokenizer)])
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
            del inp
            if 'out' in locals(): del out
            gc.collect(); torch.cuda.empty_cache()

    def gen_with_hooks(prompt_text, hook_class, vecs_dict, keys, intensities,
                       hook_target="attn", do_ortho=True, envelope_type="gaussian"):
        inp = tokenizer(fmt(prompt_text), return_tensors="pt").to(model.device)
        pl = inp.input_ids.shape[1]
        hs = []
        lp = LogitsProcessorList([NaNSafeLogitsProcessor(tokenizer)])
        try:
            for l in INTERVENTION_LAYERS:
                avs = [vecs_dict[l][k].to(model.device) for k in keys]
                if hook_class == GEMSHook:
                    hook = GEMSHook(avs, intensities, l, envelope_type=envelope_type,
                                  do_orthogonalize=do_ortho)
                else:
                    hook = hook_class(avs, intensities, l)
                target = model.model.layers[l] if hook_target == "layer" \
                    else get_out_proj(model.model.layers[l])
                hs.append(target.register_forward_hook(hook))
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

    # Run
    results = {tn: {"correct": 0, "total": 0, "details": []} for tn in track_names}
    t_start = time.time()

    for i, (q, gold) in enumerate(zip(questions, gold_answers)):
        prompt = f"Question: {q['question']}\nAnswer: Let's think step by step."
        print(f"\n[{i+1}/{NUM_QUESTIONS}] {q['question'][:60]}...")

        tracks_config = [
            ("T1_Baseline", "baseline", None, None, None, None),
            ("T2_GEMS_Full", "hooks", GEMSHook, norm_vecs, exp_keys_3, INTENSITIES_3),
            ("T3_Naive", "hooks", GEMSHook, norm_vecs, exp_keys_3, INTENSITIES_3),
            ("T4_Single_Acc", "hooks", GEMSHook, norm_vecs, exp_keys_acc, INTENSITIES_1),
            ("T5_Single_Min", "hooks", GEMSHook, norm_vecs, exp_keys_min, INTENSITIES_1),
            ("T6_ActAdd", "hooks", ActAddHook, raw_vecs, exp_keys_3, INTENSITIES_3),
            ("T7_Uniform", "hooks", GEMSHook, norm_vecs, exp_keys_3, INTENSITIES_3),
            ("T8_LayerHook", "hooks", GEMSHook, norm_vecs, exp_keys_3, INTENSITIES_3),
        ]

        for tn, mode, hcls, vdict, keys, ints in tracks_config:
            if mode == "baseline":
                out = gen_baseline(prompt)
                do_ortho = None
            elif tn == "T3_Naive":
                out = gen_with_hooks(prompt, hcls, vdict, keys, ints, do_ortho=False)
                do_ortho = False
            elif tn == "T7_Uniform":
                out = gen_with_hooks(prompt, hcls, vdict, keys, ints,
                                     do_ortho=True, envelope_type="uniform")
                do_ortho = True
            elif tn == "T8_LayerHook":
                out = gen_with_hooks(prompt, hcls, vdict, keys, ints, hook_target="layer")
                do_ortho = None
            else:
                out = gen_with_hooks(prompt, hcls, vdict, keys, ints, do_ortho=True)
                do_ortho = True

            pred = extract_answer(out)
            ok = answers_match(pred, gold)
            results[tn]["correct"] += int(ok)
            results[tn]["total"] += 1
            results[tn]["details"].append({"idx": i, "gold": gold, "pred": pred, "ok": ok,
                                           "len": len(out), "output": out})
            tag = "OK" if ok else "WR"
            print(f"  {tn:16s} {tag} pred={pred}")

        na = i + 1
        stats = " ".join([f"{tn}={results[tn]['correct']/na*100:.0f}%" for tn in track_names])
        print(f"  >> {stats} | {time.time()-t_start:.0f}s")

        # Checkpoint every 20 questions
        if (i + 1) % 20 == 0:
            _save_checkpoint(results, out_path, track_names, time.time() - t_start)

    # Summary
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({NUM_QUESTIONS} questions)")
    print(f"{'='*60}")
    na = results["T1_Baseline"]["total"]
    for tn in track_names:
        n = results[tn]["total"]
        acc = results[tn]["correct"] / n * 100 if n else 0
        print(f"  {tn:16s}: {results[tn]['correct']}/{n} = {acc:.0f}%")
    print(f"{'='*60}")

    results["elapsed"] = time.time() - t_start
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[Saved] {out_path}")
    print("[DONE]")


def _save_checkpoint(results, out_path, track_names, elapsed):
    for tn in track_names:
        n = results[tn]["total"]
        results[tn]["accuracy"] = results[tn]["correct"] / n * 100 if n else 0
    results["elapsed"] = elapsed
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    na = results[track_names[0]]["total"]
    stats = " ".join([f"{tn}={results[tn]['accuracy']:.0f}%" for tn in track_names])
    print(f"\n[Checkpoint {na}Q] {stats}")


if __name__ == "__main__":
    main()
