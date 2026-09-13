# tasks/05 — Analysis, seed aggregation, hypothesis grading (Phase 4)

Owner: agent for code; **Laksh grades the hypotheses**. Files: `analysis/stats.py`, `analysis/plots.py`,
`analysis/report.py`, `analysis/sanity.py` cross-run checks.

## Inputs
All runs in `runs/eval/`, `runs/rft/`, `runs/grpo/` with metrics.json, budgets.json, curves.json.

## Produce
1. **Cross-run sanity** (fail loudly): same cap and prompt template across every run compared; disjoint splits;
   final checkpoints only; seeds actually differ (adapter hashes differ); budgets present.
2. **Tables** (markdown + CSV): per arm, seeds as rows, μ ± σ; columns val/test/ood greedy (± CI), mean@8,
   pass@8, pass@64, trunc %, extraction-fail %, prompts, completions available/consumed, training tokens,
   optimizer steps, GPU-hours. Per-tier breakdown on test. Reference models as extra rows.
3. **Paired comparisons** for the pre-registered contrasts, seed-wise (seed i vs seed i), with mean Δ, seed
   std of Δ, and the SPEC §10 criterion evaluated (|Δ| > 2× pooled seed std AND same sign on ood):
   H1: RFT-Mixed − RFT-Easy; GRPO-Mixed − GRPO-Easy.
   H2: (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed), with the pre-registered threshold.
   H3: GRPO-Curated − RFT-Curated on test and ood.
   Controls: C1 and C2 vs Base.
4. **Plots**: (a) test accuracy by arm with every seed as a dot and μ ± σ bar; (b) same for ood; (c) per-tier
   accuracy by arm; (d) pass@k curves base vs each arm (sharpening vs expansion); (e) GRPO training reward vs
   val accuracy at checkpoints, per arm; (f) `frac_reward_zero_std` over training per arm and tier (the
   implicit-filtering view); (g) accuracy vs training tokens and vs optimizer steps (the budget view);
   (h) completion length over training. All figures show seeds and CIs.
5. **Hypothesis grading sheet** `outputs/hypotheses.md`: for H1–H3 and the two pre-registered predictions, the
   number, the criterion, and a blank verdict line **for Laksh to fill**. The agent does not write verdicts.
6. **"How could this be wrong" list** auto-generated per headline contrast: truncation gap between arms,
   extraction-failure gap, budget gap, tier-composition gap, seed range overlap.

## Acceptance
- `make analysis` regenerates everything from run dirs deterministically.
- No number in a table lacks n/seed/CI/trunc.
