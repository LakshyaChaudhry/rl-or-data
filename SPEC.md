# SPEC.md — Is it the RL or the data?

**Under matched prompt and rollout budgets, what portion of low-data RLVR gains comes from data selection versus the RL objective itself? A controlled study on procedurally generated counting tasks in the regime of Bauer et al. (Snorkel, MLSys 2026).**

Status: LOCKED v1.3 (2026-09-08). Changes to any section marked 🔒 require an entry in the Changelog (§12) with a reason and date, and may never be made after looking at test-set results.

---

## 1. Research question

Under matched prompt and rollout budgets, what portion of low-data RLVR gains comes from **data selection** versus the **RL objective itself**? The procedural counting task is the first controlled testbed for this question, not the whole identity of the project; a second task can be added later without changing the framing.

Concretely: with the same underlying prompts, the same verifier, and the same number of sampled completions, does GRPO outperform rejection-sampling fine-tuning (RFT)? And is the "mixed-difficulty data is ~5× more sample-efficient" effect reported by Bauer et al. a property of RL, or a property of the data that SFT would exhibit too?

### Hypotheses (pre-registered)

- **H1 (data effect):** Mixed-difficulty training data improves held-out performance relative to easy-only data *for both* RFT and GRPO. Test: (RFT-Mixed − RFT-Easy) and (GRPO-Mixed − GRPO-Easy), each vs. seed spread. If both are positive, the Bauer et al. effect is broader than RLVR.
- **H2 (implicit filtering):** RFT trained on prompts selected by the *same current-policy criterion that makes GRPO groups informative* (1–7 correct of 8) recovers a substantial fraction of GRPO's advantage. Test: (RFT-Curated − RFT-Mixed) relative to (GRPO-Curated − RFT-Mixed). If large, RL's edge is mostly an expensive data-selection mechanism (cf. Lu et al. 2026, DC-SFT, in VLMs).
- **H3 (on-policy / negative-feedback effect):** GRPO-Curated outperforms RFT-Curated by more than run-to-run variability despite identical prompt and sampling budgets. Residual gain can only come from what RFT lacks: negative samples, relative reward, on-policy generation, repeated policy-dependent exploration. **This is the headline comparison.**

Any pattern of outcomes is reportable. The write-up is written for whichever pattern the data shows.

## 2. Anchor paper and gaps

Bauer, Walshe, Pham, Vishwakarma, Parchami, Sala, Varma. *Learning from Less: Measuring the Effectiveness of RLVR in Low Data and Compute Regimes.* MLSys 2026 (arXiv 2604.18381).

- Setup: Qwen3-4B, LoRA r=64 α=16 all linear layers, GRPO, 8 generations/prompt, lr 5e-5, AdamW, cosine w/ 10% warmup, grad clip 1.0, 300 steps, effective batch 8 (counting), 4× A100.
- Counting result: base 31.3% → Easy-100 21.9 / Easy-200 40.0 / Easy-500 44.2; Mixed-100 44.2 / Mixed-200 43.4 / Mixed-500 35.5 (greedy, held-out test of 200).
- Gaps we fill: (a) no SFT/RFT control; (b) no multi-seed repetitions; (c) no transfer to natural-language benchmarks; (d) difficulty tiers defined by external frontier models rather than the trained model's own pass rate.

Adjacent 2026 work that shapes the design: Lu et al. (2602.10815) "implicit difficulty filtering" hypothesis and DC-SFT; Lu/JHU causal-inference RLVR-vs-SFT (2512.20760) showing regime dependence on initial competence; "SFT-then-RL outperforms mixed-policy" (2604.23747) showing published RL gains evaporate against tuned SFT baselines; "Rethinking Generalization in Reasoning SFT" (2604.06628) on SFT optimization length.

## 3. 🔒 Models

| Role | HF id | Notes |
|---|---|---|
| Primary policy | `Qwen/Qwen3-4B-Base` | Matches anchor paper. Non-thinking, plain prompt template (§5). |
| Local dev policy | `Qwen/Qwen3-0.6B-Base` | Smoke tests and the hand-written GRPO reference loop only. Never in results. |
| Optional scale point | `Qwen/Qwen3-8B-Base` | Only after the core grid is complete. |
| Reference models (eval only) | `Qwen/Qwen3-4B` (non-thinking), `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-8B-Instruct`, `google/gemma-4-E4B-it` | Same test set, same decoding, same cap. Verify exact Gemma id at eval time. |
| Stretch teacher | `Qwen/Qwen3.5-9B` | On-policy distillation arm only. |

Precision: bf16 everywhere. No quantization in any result-bearing run.

## 4. 🔒 Task: procedural counting problems

Follows the Bauer et al. generator description. Each problem is:

1. An integer range `[lo, hi]` (inclusive).
2. A sequence of 1–4 **filters** applied in order, drawn from: even, odd, positive, negative, divisible-by-N, not-divisible-by-N, below-threshold, above-threshold, digit-sum-equals, contains-digit, prime, perfect-square.
3. 0–3 **transformations** applied in order after filtering, drawn from: add-k, multiply-by-k, square, absolute-value, modulo-m, digit-sum, reverse-digits.
4. One final **operation**: count, unique-count, zero-count, sum, product-mod-m (to keep integers bounded), mean (integer-rounded), median, mode, min, max, range, bitwise-AND/OR/XOR over the set.
5. A deterministic integer answer computed by executing the pipeline in Python.

**Structural complexity knobs:** `range_scale` = span `hi − lo` of `[lo, hi]`, with **disjoint bands**: S ∈ [10, 50], M ∈ [51, 200], L ∈ [201, 1000] (lo ≥ 1); `n_filters` (1–4), `n_transforms` (0–3), `total_steps = n_filters + n_transforms + 1` (2–8), operator families in use.

**Every step must do work (v1.2).** A pipeline is rejected if any filter or transform leaves the multiset unchanged (no-op), if two consecutive filters are identical, if fewer than 3 values reach the final operation, if the final op is `mode` and no value repeats, or if the final op is `unique-count` and no value repeats. Because `lo ≥ 1`, the `positive` and `negative` filters can never do work in this pool and are **excluded from the main-pool taxonomy** (documented divergence from Bauer et al.'s operator list).

**Balanced pool (v1.2).** Generation is stratified: equal target counts per (`range_scale` × `total_steps`) cell, retrying until each cell is full, so rejection does not skew the pool toward short pipelines or small ranges. Cell counts are printed and stored with the pool.

**Rendering:** natural-language template, e.g. "Consider the integers from 1 to 100, inclusive. First, keep only the numbers that are even. Then, keep only the numbers that are divisible by 3. Of these numbers, count how many values remain." Templates are paraphrased across ≥3 surface forms per operator so the model cannot key on exact wording. Connectives are chosen by position ("First" for step 1, "Then"/"Next" in the middle, "Finally" for the last op); only the operation phrase is paraphrased.

**Guarantees the generator must provide:** deterministic given `(seed, config)`; every problem carries a `problem_id = sha256(canonical_pipeline_json)`; the answer is recomputed by an *independent* reference implementation in tests; splits are disjoint by `problem_id` and additionally by pipeline structure (no test pipeline appears in train with only the range changed).

## 5. 🔒 Prompt and answer format

Base models are prompted with a single plain template (no chat template, no system prompt, no thinking tags), identical for every arm and every reference model that is a base model. Instruct reference models receive the same text inside their own chat template with thinking disabled.

```
Solve the following problem. Think step by step, then give the final answer on the last line in the form "Answer: <integer>".

Problem: {problem_text}
```

Answer extraction: last line matching `^Answer:\s*(-?\d+)\s*$`. Anything else scores 0 (and is logged as `extraction_failed`). Reward for the primary arms is binary correctness only (§8); the format bonus/penalty from the anchor paper is a control condition, not the primary reward.

## 6. 🔒 Data and splits

Generate a **training/eval pool** of 6,000 problems (seed 20260901) restricted to `range_scale ∈ {S, M}` and `total_steps ∈ {2..5}`. Complexity extrapolation is measured on a separately generated pool with `range_scale = L` and `total_steps ∈ {6..8}` (`ood_hard_200`). Training never sees 6–8-step or L-range problems. Then:

1. **Tiering by the base model's own pass rate.** Sample K=8 completions per problem from `Qwen3-4B-Base` at T=1.0 (§7 decoding) and record `pass8 ∈ {0..8}`. Tiers: **easy** ≥ 6/8, **medium** 2–5/8, **hard** ≤ 1/8. (This replaces the anchor paper's 10-frontier-model tiering; the difference is documented in the write-up.)
2. **Splits (disjoint by problem_id and pipeline structure):**
   - `train_easy_100`: 100 easy.
   - `train_mixed_100`: 33 easy / 33 medium / 34 hard.
   - `val_mixed_100`: 100, 33/33/34, used for all model selection.
   - `test_300`: 100/100/100 stratified. **Never used for any decision.**
   - `train_curated`: the subset of `train_mixed_100` with **1 ≤ pass8 ≤ 7**, using the first 8 of the base model's samples in generation order. Frozen at initialization; never re-selected as any policy improves. Its size and tier composition are reported (expected ~50–75 prompts).
   - `ood_hard_200`: generated separately with `range_scale=L` and `total_steps ∈ {6,7,8}`, tiered post hoc but not filtered. This is the **complexity-extrapolation** set; `test_300` is the **in-distribution fresh-instance** set.
   - Optional (priority 2): `train_medium_100`, `train_hard_100`, `train_easy_500`, `train_mixed_500`.
3. **Generalization is measured in two tiers.** Primary: in-distribution fresh instances (`test_300`) and complexity extrapolation (`ood_hard_200`) — the specific question is whether RFT and GRPO *differ* in extrapolation, since Bauer et al. already report RLVR extrapolating upward. Secondary (eval-only, cheap, expected small): one Reasoning Gym task chosen in Phase 1 from {`basic_arithmetic`, `number_filtering`, `count_primes`} — whichever the base scores 20–60% on — 300 instances; GSM8K test subset, first 500 by index. Same prompt template and cap. Cross-domain transfer is not a headline claim.

## 7. 🔒 Decoding and the token cap

- Sampling (training rollouts, RFT sampling, mean@k, pass@k): T=1.0, top_p=1.0, no repetition penalty.
- Greedy (primary metric): T=0.
- **Cap rule (v1.3):** `max_completion_tokens = max(2048, ceil_to_256(1.25 × p99_max))`, where `p99_max` is the largest per-cell p99 length of *correct* base-model completions on val_mixed_100 at T=1.0, cells = `range_scale × total_steps`. The 2048 floor matches Bauer et al. (Table 1); the measured term raises the cap only if 2048 would truncate more than 1% of correct completions in any cell, and any raise is recorded in `cap.yaml` with the per-cell p99s. Computed once in Phase 1, written to `configs/locked/cap.yaml`, then identical for every arm, control, and reference model, training and evaluation. Truncation and extraction-failure rates are reported per tier and per split in every results table, alongside the fraction of correct completions above 2048. A run with truncation > 5% on test_300 is flagged and its numbers are not headline numbers; on ood_hard_200 truncation is reported, not flagged.
- Max prompt tokens: 4096 (matches Bauer et al.; counting prompts are far shorter).

## 8. 🔒 Arms

All arms train LoRA on `Qwen3-4B-Base` with the shared hyperparameters in §9 and the same verifier. The matrix crosses **prompt distribution** with **training signal**:

| Arm | Prompt set | Training signal | Purpose |
|---|---|---|---|
| Base | — | none | reference |
| RFT-Easy | `train_easy_100` | verified-correct completions (off-policy SFT) | H1 |
| RFT-Mixed | `train_mixed_100` | verified-correct completions | H1, H2 |
| RFT-Curated | `train_curated` | verified-correct completions | H2, H3 |
| GRPO-Easy | `train_easy_100` | binary verifiable reward, group-relative | H1 |
| GRPO-Mixed | `train_mixed_100` | binary verifiable reward | H1, H2 |
| GRPO-Curated | `train_curated` | binary verifiable reward | **H3 headline vs RFT-Curated**; also a check that explicit pre-filtering changes little for GRPO (it should not, if implicit filtering is real) |
| C1 | `train_mixed_100` | random reward ~ Bernoulli(0.5) | spurious-reward control |
| C2 | `train_mixed_100` | 1 iff answer line parses | format-only control |
| Priority 2: Iterated RFT (ReST-style, 3 rounds) | `train_curated` | verified-correct, re-sampled from the current policy each round | isolates on-policy re-sampling from advantage weighting |
| Stretch: On-policy distillation | `train_curated` | per-token teacher log-probs (`Qwen3.5-9B`) | — |

**Core grid** = 6 trained arms × 3 seeds = 18 runs (9 GRPO, 9 RFT), plus Base, reference models, and C1/C2 (1 seed). If budget allows, the H3 pair (GRPO-Curated, RFT-Curated) runs **5 seeds**.

### Budgets (three, all reported)

1. **Prompt budget:** the underlying prompt set is identical for the arm pair being compared (Easy/Easy, Mixed/Mixed, Curated/Curated).
2. **Generation (rollout) budget:** GRPO consumes 300 steps × 8 prompts × 8 generations = **19,200 sampled completions**. Every RFT arm is given the same 19,200: K = 192 samples per prompt from the base model on `train_mixed_100` (RFT-Easy samples on `train_easy_100`), sampled once and reused; `train_curated` uses the samples belonging to its prompts. Report both budget *available* and budget *consumed* (RFT arms discard incorrect completions; curated arms use fewer prompts).
3. **Gradient/token budget:** GRPO trains on all 8 completions per group; RFT trains only on correct ones. These cannot be equalized without changing the method — that asymmetry *is* the mechanism under test. Report training tokens consumed and optimizer steps per arm, and present results both per source prompt and per rollout.

RFT gets the positive trajectories obtainable under the matched generation budget; GRPO additionally exploits relative outcomes across successes and failures. H3 asks whether that additional information matters.

**Selection is frozen.** The curated set is chosen once from base-model samples. GRPO's implicit, adaptive filtering (zero-advantage groups) happens inside the algorithm on the same frozen prompt set; that is part of what "the RL objective" means here. Static vs. adaptive prompt selection is an optional later ablation, not part of the primary comparison.

## 9. 🔒 Shared training hyperparameters (match anchor paper)

| | Value |
|---|---|
| LoRA | r=64, α=16, dropout 0, all linear projections (q,k,v,o,gate,up,down) |
| Optimizer | AdamW, weight decay 0, grad clip 1.0 |
| LR schedule | cosine, 10% warmup |
| GRPO | 300 steps; 8 prompts/step; G=8 generations/prompt; T=1.0; lr 5e-5; KL coef β=0.04 (TRL default at time of writing — record actual); clip ε=0.2; loss = standard GRPO (not Dr.GRPO/DAPO) |
| RFT SFT | lr ∈ {1e-5, 5e-5, 1e-4}, epochs ∈ {2, 4, 8}, batch 16, completion-only loss; best-on-val config reported, all configs logged |
| Seeds | 1, 2, 3 control sampling, data order, LoRA init; 4, 5 added to the H3 pair if budget allows |
| Checkpointing | every 25 steps to S3; final = last step (no early stopping on test) |

## 10. 🔒 Evaluation protocol

- **Primary:** greedy accuracy on `test_300` (overall and per tier) and on `ood_hard_200`.
- **Secondary:** mean@8 at T=1.0; pass@k for k ∈ {1,2,4,8,16,32,64} from n=64 samples on a fixed 100-problem subset of test_300 (unbiased estimator, Chen et al. 2021); mean completion length; truncation rate; extraction-failure rate.
- **Uncertainty:** per-problem bootstrap 95% CI (10,000 resamples) for each run; mean ± std across seeds for each arm **with every individual seed shown**; paired seed-wise differences between arms.
- **Budget reporting:** every results table carries the three budgets from §8 (prompts, completions available/consumed, training tokens and optimizer steps).
- **Model selection:** only on `val_mixed_100`. Test is evaluated once per final checkpoint.
- **Transfer:** RG task and GSM8K-500, greedy, before/after, all arms.
- **Reference models:** same test_300 / ood_hard_200, same template, same cap, greedy and mean@8.
- **Generation path:** every number in the paper comes from vLLM on the GPU. No MLX or `model.generate()` numbers in results.

### Pre-registered success criterion

An arm-vs-arm difference "counts" if (a) |Δ greedy accuracy on test_300| > 2 × the pooled seed std of the two arms, and (b) the sign of Δ is the same on ood_hard_200. Report all comparisons regardless.

## 11. Compute plan

- **GPU:** one NVIDIA GPU per run. Preferred `p5.4xlarge` (1× H100 80GB) if on-demand/spot exists in the account's region; otherwise `g6e.xlarge/2xlarge` (1× L40S 48GB). Multi-seed runs may be parallelized across a `g6e.12xlarge` (4× L40S), one run per GPU.
- **Stack:** TRL (GRPOTrainer, colocated vLLM) + PEFT + vLLM for all sampling/eval. See `setup/setup_gpu.sh`.
- **Local (Mac Studio M3 Ultra 96GB):** generator, verifier, tests, analysis, hand-written core functions, and the reference GRPO loop on Qwen3-0.6B-Base. No result-bearing runs.
- **Estimates:** GRPO 4B run ≈ 2–4 h (H100) / 5–8 h (L40S). RFT sampling ≈ 20 min per set. SFT sweep ≈ 9 short runs ≈ 1–2 h. Core grid ≈ 40–100 GPU-hours. Budget ceiling for the whole project incl. follow-ons: $3,000 of the $10k credit pool.
- **Hygiene:** idle auto-shutdown on every instance; checkpoints and eval JSONL synced to S3; every run has a config hash and a notebook entry.

## 12. Deliverables, timeline, changelog

**Deliverables:** public repo (generator, verifier, harness, configs, run logs); LoRA adapters on HF; write-up (workshop format) with plots, seeds, CIs, truncation rates, compute log, negative results; lab notebook excerpts.

**Timeline (target, revised 2026-09-07 for ~10 h/week):** Sep 7–20 (light): verify/logprobs/evaluate, generator merged and reviewed, AWS quota, pre-registration §1–2. Sep 21–Oct 4: base + reference evals, cap, tiering, splits, pre-registration §3; RFT arms + SFT sweep. Oct 5–18: reference GRPO loop on 0.6B → TRL GRPO grid. Oct 19–31: seeds, controls, extrapolation, secondary transfer, first results. November: write-up.

**Changelog**
- v1.0 (2026-09-01): initial lock.
- v1.3 (2026-09-08), before any GPU run: §7 cap changed to max(2048, per-cell data-derived) and max prompt tokens 1024 → 4096. Reason: (a) match Bauer et al. Table 1 by default; (b) the v1.2 rule calibrated on all correct completions, which are dominated by easy problems, so it could sit below what correct 5-step reasoning needs; per-cell calibration fixes that; (c) a fixed cap borrowed from a length-shaped reward could truncate binary-reward completions.
- v1.2 (2026-09-08), after reviewing the first generator output, before any tiering or training: §4 gains disjoint span bands per range scale, the every-step-must-do-work rejection rules, degenerate-final-op rejection, stratified generation per (scale × steps) cell, position-aware connectives, and removal of the dead `positive`/`negative` filters from the main pool. Motivated by no-op filters inflating `total_steps` (which would corrupt the complexity-extrapolation axis) and by tiny ranges producing single-element sets. (Transcription errors in the v1.2 text as first applied were corrected on 2026-09-07 without changing meaning.)
- v1.1 (2026-09-07), before any training run: (a) research question generalized so counting is the testbed, not the identity; (b) arms restructured to a prompt-distribution × signal matrix — added GRPO-Curated, dropped RFT-Curated-on-easy; (c) training pool restricted to S/M ranges and 2–5 steps so `ood_hard_200` is a clean complexity-extrapolation test; (d) three budgets (prompt, rollout, gradient/token) defined and all reported; (e) H3 pair gets 5 seeds if budget allows; (f) timeline revised for ~10 h/week. Prompted by an external review; the review's 100×8=800 rollout figure was rejected — GRPO's budget is 19,200 (§8) and matching RFT to 800 would have under-budgeted it 24×.

## 13. Component contracts (summary; signatures live in `src/rlordata/core/`)

Hand-written by Laksh (agent must not edit): `core/logprobs.py`, `core/sft_loss.py`, `core/verify.py`, `core/rft_select.py`, `core/grpo.py`, `core/evaluate.py`. Agent-owned: everything else. Interfaces are frozen in the stubs; changes require a SPEC changelog entry.

## 14. Glossary (for the write-up and for Laksh)

- **RFT** — rejection-sampling fine-tuning: sample, keep verified-correct, SFT.
- **GRPO** — group relative policy optimization: G samples per prompt, advantage = (r − mean_G)/std_G, PPO-style clipped ratio, KL to reference.
- **pass@k** — probability that at least one of k samples is correct; estimated without bias from n ≥ k samples.
- **Implicit prompt filtering** — in GRPO, prompts where all G samples share the same reward have zero advantage and contribute no gradient; the effective training set is the "medium" prompts.
