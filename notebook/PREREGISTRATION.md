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
- 

## 3. Phase 1 facts that fix the protocol
- Base greedy accuracy on val_mixed_100: ___ (CI ___)
- Base pass@8 on val: ___; tier counts in pool: easy ___ / medium ___ / hard ___
- Locked cap: ___ tokens (p99 correct = ___; correct-truncation at cap = ___%)
- Transfer RG task chosen: ___ (base acc ___)
- Predicted outcomes (write BEFORE training): RFT-all mixed vs easy: ___; GRPO vs RFT-curated: ___

## 4. Deviations log
- (date, what, why, approved by Laksh)
