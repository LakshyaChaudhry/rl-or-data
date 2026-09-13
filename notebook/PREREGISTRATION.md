# Pre-registration — fill §1–2 before the first RL run; §3 after Phase 1 evals; never after seeing test results

## 1. Hypotheses in my own words (Laksh)
- H1: Data affect - mixed difficulty is better training data. 1 - test whether mixed > easy for both RTF & GRPO, if yes then Bauer et al.'s advantage isn't RL unique
- H2: Implicit filtering: GRPO's advantage mostly comes from picking informative prompts. Give RFT the curated 'goldilocks' prompts where the base model gets 1-7/8 correct. IF RFT-Curated closes much of the gap to GRPO, then a large portion of RLVR's benefits is really just data selection/difficulty filtering
- H3: Genuine RL effect - after controlling data, does GRPO still win? - compare GRPO-curated vs. RFT-Curated with the same prompts and sampling budget - If GRPO still wins beyond seed noise, that residual advantage comes from things RFT lacks: negative samples, relative rewards, on-policy generation 
- What result would falsify each:
for H1: Falsified by an asymmetry: GRPO-Mixed > GRPO-Easy by more than seed spread, but RFT-Mixed ≈ RFT-Easy (or worse). That would mean the mixed-difficulty advantage only exists through the RL objective
for H2: Falsified if RFT-Curated ≈ RFT-Mixed — the same prompt filter that makes GRPO groups informative does nothing for SFT — while GRPO-Curated stays well above both. Also falsified if RFT-Curated is worse than RFT-Mixed
for H3: Falsified if the difference on test_300 is within 2× the pooled seed std, or it flips sign on ood_hard_200, or RFT-Curated ≥ GRPO-Curated. Any of those means: once data selection is controlled, the RL objective adds nothing you can measure at this budget.

## 2. Success criterion (copied from SPEC §10, restated)
- two bars that are both required:
    1) big enough gap on the main test set - here the gap in the greedy accuracy on the test_300 set should be larger than twice the typical seed-to-seed gap/wobble of the two pooled arms, as smaller gaps could be attributed to just random seed noise
    2) Same direction for the gap on hard OOD problems - here the gap should go in the same way on the OOD-hard-200 set.

## 3. Phase 1 facts that fix the protocol
- Base greedy accuracy on val_mixed_100: 0.600 (CI [0.510, 0.700]; n=100, seed 1, cap 4352, config hash b2356e9825f0, truncation 1.0 %, extraction failure 1.0 %)
- Base pass@8 on val: 0.830 (n=100×8, T=1.0, seed 1, config hash 8f90d8cf2df5, truncation 0.25 %); tier counts in pool: easy 1177 / medium 2657 / hard 2166 (SPEC v1.6 extractor, splits frozen at ffb604e)
- Locked cap: 4352 tokens (p99 correct = 3477.94 under the v1.5 extractor that cap.yaml records; under v1.6 the same completions give p99 3161.72 and the rule would return 4096 — cap kept at 4352 as the conservative bound; correct-truncation at cap = 0.0%)
- Transfer RG task chosen: ___ (base acc ___) — BLOCKED: rg_basic_arithmetic 0.757 (above band), rg_count_primes 0.023 (floor); SPEC §6.3 amendment pending
- Predicted outcomes (write BEFORE training): RFT-all mixed vs easy: RFT-Mixed > RFT-Easy on test_300 (same direction on ood_hard_200), gap > seed noise — mixed difficulty helps SFT too, so Bauer’s effect is not RL-only (H1). GRPO vs RFT-curated: GRPO-Curated > RFT-Curated on test_300 with the same sign on ood, but the residual is modest; most of GRPO’s edge vs RFT-Mixed is recovered by curation (H2), with a leftover from negatives / relative reward / on-policy (H3).
- dropped RG, put GSM8K-500 @ 0.880 in replacement.

## 4. Deviations log
- (date, what, why, approved by Laksh)
- 2026-09-13 — **RFT trainer now trains the stop token (tasks/03 §3 amended).** What: the SFT labels for every RFT completion end with the base model's EOS token (`<|endoftext|>`, id 151643 — the only stop token in Qwen3-4B-Base's generation config, and the token vLLM stopped on when the draw was sampled; vLLM's returned text omits it). The original §3 rule said "no EOS beyond what the tokenizer produces", so the RFT model was never trained to end a completion. Why: in the first mixed sweep val truncation rose with training (up to 36 % of val completions hit the 4352-token cap; most had already written an answer and kept going), which SPEC §5 scores as wrong. That handicapped RFT against GRPO, which trains on EOS through TRL — a confound for H3 unrelated to the RL objective. Consequence: all completed and in-progress no-EOS sweep runs were discarded (moved to `runs/rft_noeos_ablation/`, kept only as an ablation) and all 27 sweep runs restart from scratch; finals use only the new sweeps. Only `val_mixed_100` numbers from the discarded runs had been seen — no RFT test_300 or ood_hard_200 result exists. Unchanged: the 192-sample draw, the curated selection, the cap, prompt, verifier, and GRPO. Considered and rejected: removing EOS from GRPO instead (GRPO would still learn to stop indirectly because capped completions score 0, and it would make RFT a deflated SFT baseline). Code: 67589f2, f7de481; guard refusing reuse of any run without `append_eos: true`: b877fe3. Approved by Laksh.
