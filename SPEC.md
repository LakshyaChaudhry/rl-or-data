# SPEC.md — Is it the RL or the data?

**A matched-budget comparison of rejection-sampling SFT and GRPO on procedurally generated counting tasks, in the low-data regime of Bauer et al. (Snorkel, MLSys 2026).**

Status: LOCKED v1.0 (2026-09-01). Changes to any section marked 🔒 require an entry in the Changelog (§12) with a reason and date, and may never be made after looking at test-set results.

---

## 1. Research question

Under a matched budget — same prompts, same verifier, same number of sampled completions — does GRPO outperform rejection-sampling fine-tuning (RFT) on procedurally generated multi-step counting problems? And is the "mixed-difficulty data is ~5× more sample-efficient" effect reported by Bauer et al. a property of RL, or a property of the data that SFT would exhibit too?

### Hypotheses (pre-registered)

- **H1 (data effect):** Mixed-difficulty training data beats easy-only data at equal size *for every arm*, including RFT. If true, the Bauer et al. effect is a data effect.
- **H2 (implicit filtering):** RFT-curated (prompts filtered to 1–7 correct out of 8, replicating GRPO's implicit prompt filter) closes most of the gap between RFT-all and GRPO. If true, GRPO's advantage is mostly data selection (cf. Lu et al. 2026, DC-SFT, in VLMs).
- **H3 (on-policy/negatives effect):** GRPO exceeds RFT-curated by more than the seed spread on held-out and OOD-hard. If true, learning from negatives and iterative on-policy updates contribute beyond data selection.

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

**Structural complexity knobs:** `range_scale` (span of `[lo, hi]`: S ≤ 50, M ≤ 200, L ≤ 1000), `n_filters` (1–4), `n_transforms` (0–3), `total_steps = n_filters + n_transforms + 1` (2–8), operator families in use.

**Rendering:** natural-language template, e.g. "Consider the integers from 1 to 100, inclusive. First, keep only the numbers that are even. Then, keep only the numbers that are divisible by 3. Of these numbers, count how many values remain." Templates are paraphrased across ≥3 surface forms per operator so the model cannot key on exact wording.

**Guarantees the generator must provide:** deterministic given `(seed, config)`; every problem carries a `problem_id = sha256(canonical_pipeline_json)`; the answer is recomputed by an *independent* reference implementation in tests; splits are disjoint by `problem_id` and additionally by pipeline structure (no test pipeline appears in train with only the range changed).

## 5. 🔒 Prompt and answer format

Base models are prompted with a single plain template (no chat template, no system prompt, no thinking tags), identical for every arm and every reference model that is a base model. Instruct reference models receive the same text inside their own chat template with thinking disabled.

```
Solve the following problem. Think step by step, then give the final answer on the last line in the form "Answer: <integer>".

Problem: {problem_text}
```

Answer extraction: last line matching `^Answer:\s*(-?\d+)\s*$`. Anything else scores 0 (and is logged as `extraction_failed`). Reward for the primary arms is binary correctness only (§8); the format bonus/penalty from the anchor paper is a control condition, not the primary reward.

## 6. 🔒 Data and splits

Generate a pool of 6,000 problems (seed 20260901) across the full knob space, then:

1. **Tiering by the base model's own pass rate.** Sample K=8 completions per problem from `Qwen3-4B-Base` at T=1.0 (§7 decoding) and record `pass8 ∈ {0..8}`. Tiers: **easy** ≥ 6/8, **medium** 2–5/8, **hard** ≤ 1/8. (This replaces the anchor paper's 10-frontier-model tiering; the difference is documented in the write-up.)
2. **Splits (disjoint by problem_id and pipeline structure):**
   - `train_easy_100`: 100 easy.
   - `train_mixed_100`: 33 easy / 33 medium / 34 hard.
   - `val_mixed_100`: 100, 33/33/34, used for all model selection.
   - `test_300`: 100/100/100 stratified. **Never used for any decision.**
   - `ood_hard_200`: generated separately with `range_scale=L` and `total_steps ∈ {6,7,8}`, tiered post hoc but not filtered.
   - Optional (priority 2): `train_medium_100`, `train_hard_100`, `train_easy_500`, `train_mixed_500`.
3. **Transfer sets (fixed before training):** one Reasoning Gym task chosen in Phase 1 from {`basic_arithmetic`, `number_filtering`, `count_primes`} — whichever the base scores 20–60% on — 300 instances; GSM8K test subset, first 500 by index. Same prompt template and cap.

## 7. 🔒 Decoding and the token cap

- Sampling (training rollouts, RFT sampling, mean@k, pass@k): T=1.0, top_p=1.0, no repetition penalty.
- Greedy (primary metric): T=0.
- **Cap rule:** `max_completion_tokens = ceil_to_256(1.25 × p99 length of *correct* base-model completions on val_mixed_100 at T=1.0)`, minimum 512. Computed once in Phase 1, written to `configs/locked/cap.yaml`, and then identical for every arm, every control, every reference model, training and evaluation. Truncation rate is reported in every results table. A run with truncation rate > 5% on test is flagged and its numbers are not headline numbers.
- Max prompt tokens: 1024 (counting prompts are short).

## 8. 🔒 Arms

All arms train LoRA on `Qwen3-4B-Base` with the shared hyperparameters in §9, on the same prompt set, using the same verifier.

| # | Arm | What it adds vs. previous rung | Budget definition |
|---|---|---|---|
| 0 | Base | — | — |
| 1 | RFT-all | Sample K per prompt from base; keep verified-correct completions; SFT. | Total sampled completions = GRPO's total (§9): 300 steps × 8 prompts × 8 gen = 19,200 → K = 192 per prompt for a 100-prompt set. SFT epochs/LR chosen on val (§10). |
| 2 | RFT-curated | Same samples as Arm 1; keep only prompts whose base pass@8 (first 8 of the 192) is in [1, 7]; SFT on their correct completions. | Same as Arm 1. |
| 3 | GRPO | Same prompts; group-relative advantages; learns from negatives; iterative on-policy updates. | 19,200 sampled completions. |
| 3b (priority 2) | Iterated RFT (ReST-style) | 3 rounds of sample → filter → SFT from the current policy; isolates on-policy re-sampling from advantage weighting. | 19,200 total across rounds. |
| 4 (stretch) | On-policy distillation | Student rollouts scored per-token by `Qwen3.5-9B` log-probs. | Same rollout count. |
| C1 | GRPO, random reward | Reward ~ Bernoulli(0.5), independent of correctness. Spurious-reward control. | Same as Arm 3. |
| C2 | GRPO, format-only reward | Reward = 1 iff answer line parses. | Same as Arm 3. |

Primary budget unit = **sampled completions (verifier calls)**. Training FLOPs and wall-clock are reported as secondary.

**Core grid** = Arms {1, 2, 3} × {train_easy_100, train_mixed_100} × seeds {1, 2, 3} = 18 runs, plus Arm 0, reference models, and C1/C2 on mixed_100 with 1 seed.

## 9. 🔒 Shared training hyperparameters (match anchor paper)

| | Value |
|---|---|
| LoRA | r=64, α=16, dropout 0, all linear projections (q,k,v,o,gate,up,down) |
| Optimizer | AdamW, weight decay 0, grad clip 1.0 |
| LR schedule | cosine, 10% warmup |
| GRPO | 300 steps; 8 prompts/step; G=8 generations/prompt; T=1.0; lr 5e-5; KL coef β=0.04 (TRL default at time of writing — record actual); clip ε=0.2; loss = standard GRPO (not Dr.GRPO/DAPO) |
| RFT SFT | lr ∈ {1e-5, 5e-5, 1e-4}, epochs ∈ {2, 4, 8}, batch 16, completion-only loss; best-on-val config reported, all configs logged |
| Seeds | 1, 2, 3 control sampling, data order, LoRA init |
| Checkpointing | every 25 steps to S3; final = last step (no early stopping on test) |

## 10. 🔒 Evaluation protocol

- **Primary:** greedy accuracy on `test_300` (overall and per tier) and on `ood_hard_200`.
- **Secondary:** mean@8 at T=1.0; pass@k for k ∈ {1,2,4,8,16,32,64} from n=64 samples on a fixed 100-problem subset of test_300 (unbiased estimator, Chen et al. 2021); mean completion length; truncation rate; extraction-failure rate.
- **Uncertainty:** per-problem bootstrap 95% CI (10,000 resamples) for each run; mean ± std across the 3 seeds for each arm; paired seed-wise differences between arms.
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

**Timeline (target):** W1 Sep 1–7 data + harness + cloud + base evals + cap. W2 Sep 8–14 RFT arms + SFT sweep. W3 Sep 15–21 GRPO (reference loop on 0.6B → TRL grid). W4 Sep 22–28 seeds, controls, OOD, transfer, draft. October: paper.

**Changelog**
- v1.0 (2026-09-01): initial lock.

## 13. Component contracts (summary; signatures live in `src/rlordata/core/`)

Hand-written by Laksh (agent must not edit): `core/logprobs.py`, `core/sft_loss.py`, `core/verify.py`, `core/rft_select.py`, `core/grpo.py`, `core/evaluate.py`. Agent-owned: everything else. Interfaces are frozen in the stubs; changes require a SPEC changelog entry.

## 14. Glossary (for the write-up and for Laksh)

- **RFT** — rejection-sampling fine-tuning: sample, keep verified-correct, SFT.
- **GRPO** — group relative policy optimization: G samples per prompt, advantage = (r − mean_G)/std_G, PPO-style clipped ratio, KL to reference.
- **pass@k** — probability that at least one of k samples is correct; estimated without bias from n ≥ k samples.
- **Implicit prompt filtering** — in GRPO, prompts where all G samples share the same reward have zero advantage and contribute no gradient; the effective training set is the "medium" prompts.
