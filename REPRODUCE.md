# Reproduction Guide

Experiment environment: PyTorch 2.5.1+cu121, Transformers 5.9.0, NVIDIA V100.

Transformers `>=4.57.0` is required (Qwen3.5 hybrid architecture). Other versions are flexible.

## Quick Check

```bash
# Single GSM8K question, all 8 tracks — runs in a few minutes
python code/evaluation/gsm8k_single.py --question-idx 0
```

If this completes without errors, the entire pipeline works.

## Evaluation Scripts

### `code/evaluation/gsm8k_eval.py` — Table 3: GSM8K 8-Track Ablation

Runs 50 GSM8K math problems under 8 conditions to test whether GEMS preserves mathematical reasoning while injecting 3 non-mathematical directions.

| Track | What it tests | Expected |
|-------|--------------|----------|
| T1 | Baseline (no hooks) | 92% |
| T2 | GEMS Full | 98% |
| T3 | No orthogonalization | 92% |
| T4 | Single expert (accountability) | 96% |
| T5 | Single expert (minimalism) | 92% |
| T6 | ActAdd (no constraints) | 4% (collapse) |
| T7 | No envelope (uniform) | 96% |
| T8 | Full-layer hook (not o_proj) | 4% (collapse) |

Output: `data/tables/table3_gsm8k.json`

### `code/evaluation/gsm8k_single.py` — Single-Question Smoke Test

Same 8 tracks on one question. Shows full generated output per track side by side. Use `--question-idx N` to pick a question.

### `code/evaluation/ppl_eval.py` — Table 4: Wikitext-2 PPL 6-Track

Measures language modeling quality under steering via teacher-forcing perplexity on Wikitext-2.

| Track | What it tests | Expected PPL |
|-------|--------------|-------------|
| A0 | No steering (baseline) | 14.82 |
| A1 | Hook infrastructure only (E=0) | 14.82 |
| B | GEMS Full | 15.15 |
| C | No orthogonalization | 15.49 |
| E | No envelope | 16.08 |
| A6 | ActAdd | 25,173 (collapse) |

Output: `data/tables/table4_ppl.json`

### `code/diagnostics/norm_trajectory.py` — Section 2: Failure Mode 1

Measures per-layer residual stream norm under baseline vs ActAdd. Shows that additive perturbations accumulate in norm across layers (3.4x at peak), driving activations outside the training distribution.

Output: `data/section2_diagnostic_fm1/fm1_distributional_deviation_norm_trajectory.json`

### `code/diagnostics/directional_interference.py` — Section 2: Failure Mode 2

Measures terminal-layer cosine similarity between the residual stream and each expert direction. Shows that parallel injection without orthogonalization causes all directions to simultaneously degrade (34-43% reduction).

Output: `data/section2_diagnostic_fm2/fm2_directional_interference_cosine_swallowing.json`

## Data Files

All pre-computed results from the paper. Organized by paper section.

### `data/section2_diagnostic_fm1/` — Distributional Deviation (Section 2, Failure Mode 1)

| File | Content |
|------|---------|
| `actadd_alpha_sweep_ppl_norm_ratio.json` | PPL at intervention strengths alpha=0.05 to 2.0. Shows collapse from PPL 14.97 to 80,186 at alpha=0.3 |
| `fm1_diagnostic_probes_wikitext2.json` | Per-layer diagnostic probes on Wikitext-2 |
| `fm1_distributional_deviation_norm_trajectory.json` | Per-layer norm trajectory (L0-L31), baseline vs ActAdd |

### `data/section2_diagnostic_fm2/` — Directional Interference (Section 2, Failure Mode 2)

| File | Content |
|------|---------|
| `fm2_directional_interference_cosine_swallowing.json` | Cosine similarity of each expert direction under single vs parallel injection |

### `data/section41_qualitative_stacking/` — Qualitative Stacking (Section 4.1)

Progressive stacking outputs (baseline → 1 expert → 2 experts → 3 experts) across 3 scenarios:

| File | Content |
|------|---------|
| `trolley_track_a_track_b_progressive_stacking.json` | Trolley Problem: open-ended (Track A) and constrained (Track B) |
| `pr_track_a_track_b_progressive_stacking.json` | PR crisis: open-ended (Track A) and constrained (Track B) |
| `apology_track_a_track_b_progressive_stacking.json` | Apology letter: open-ended (Track A) and constrained (Track B) |

### `data/section42_activation_validation/` — Activation-Space Validation (Section 4.2)

| File | Content |
|------|---------|
| `expert_vocab_probability_shift.json` | Vocabulary cluster probability shift under expert vectors. 5/6 conditions significant (70-88% positive rate) |
| `pr_50prompt_projection.json` | 50-prompt batch projection for PR scenario |
| `trolley_50prompt_projection.json` | 50-prompt batch projection for Trolley scenario |
| `50prompt_batch_projection.json` | Aggregated 50-prompt cosine similarity data for both scenarios (trolley + PR) |
| `random_vector_control.json` | Random vector control (3 seeds). Shows ~48% positive rate vs expert vectors' ~77% |

### `data/section43_qualitative_ablation/` — 6-Condition Qualitative Ablation (Section 4.3)

| File | Content |
|------|---------|
| `base_model_c0_c5_ablation.json` | Base model: 6 conditions (C0 baseline, C1 GEMS Full, C2 ActAdd, C3 Uniform, C4 Non-ortho, C5 LayerHook) on Trolley + PR |
| `instruct_model_c0_c5_ablation.json` | Instruct model: same 6 conditions |

### `data/section43_quantitative_gsm8k/` — GSM8K Results (Section 4.3.1)

| File | Content |
|------|---------|
| `gsm8k_8track_ablation_results.json` | Full 8-track ablation results (50 questions), per-problem accuracy and error breakdown |

### `data/section43_quantitative_ppl/` — PPL Results (Section 4.3.2)

| File | Content |
|------|---------|
| `wikitext2_ppl_6track_results.json` | Full 6-track PPL results with loss values |

### `data/section44_cross_model/` — Cross-Model Validation (Section 4.4)

| File | Content |
|------|---------|
| `llama3.2_3b_instruct_gems.json` | Llama-3.2-3B-Instruct: qualitative output under GEMS |
| `qwen3.5_4b_instruct_gems.json` | Qwen3.5-4B-Instruct: qualitative output under GEMS |
| `qwen3.6_27b_instruct_gems.json` | Qwen3.6-27B-Instruct: qualitative output under GEMS |
| `gemma4_31b_instruct_gems.json` | Gemma-4-31B-Instruct: qualitative output under GEMS |

### `data/tables/` — Aggregated Table Data

| File | Content |
|------|---------|
| `table3_gsm8k.json` | Final numbers for paper Table 3 |
| `table4_ppl.json` | Final numbers for paper Table 4 |
