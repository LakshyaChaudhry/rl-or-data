# Reviewer A: methods, math, and statistics

Independent review of the provisional condensation (`workshop/aaai27-provisional/latex/main.pdf`, 6 pages; source `shared.tex`), checked against the preprint (`workshop/preserved-arxiv-v1/arxiv-source/main.tex`), `notebook/PREREGISTRATION.md`, `results/phase5/tables/contrasts.md`, `results/phase5/tables/results.md`, `results/phase5/appendix/truncation_bounds.md`, and `results/phase5/appendix/exploratory_grpo_vs_rft.md`. No paper or code was edited. No other review was read.

**Summary.** The condensation's accuracies, seed rows, paired intervals, rollout counts, token totals, and optimizer steps match the preprint and the logs. I do not assign a blocker: nothing in Tables 1--4 is a mis-copied primary number. The problems are inferential. The lead sentences treat a truncation-flagged, budget-unmatched comparison as the pre-registered success, and Table 3's "yes" column applies a three-seed sign rule to contrasts whose own truncation bounds do not support that decision. The body and the limitations already contain most of the right caveats. The abstract, the conclusion, and the "yes" marks go past those caveats.

Severity: blocker, major, minor, nit. Each issue has a location and a one-sentence reason.

## Issues

### 1. Major. The flagged curated comparison is still the headline

PDF p.1 (abstract), p.3 (§3.1 and Table 2), p.5 (conclusion).

Every GRPO-Curated seed is above the authors' 5% test-truncation flag (5.3%, 9.3%, 8.0% in Table 4), and the protocol quoted on p.2 says a flagged run is not a headline number as it stands. The abstract and the conclusion still lead with the 10.9-point curated gap.

Reason: the truncation flag and the success rule are separate requirements, and the lead sentences satisfy only the second.

### 2. Major. "Pre-registered" omits the sampling-budget deviation in the lead sentences

PDF p.1 (abstract) and p.5 (conclusion); the disclosure is on p.2 (§2.3) and p.5 (limitations).

Registered H3 required the same prompts and the same sampling budget (`PREREGISTRATION.md` §1; preprint appendix quote of the committed text). The run matches prompts and not rollouts: GRPO-Curated samples 19,200 completions and RFT-Curated has the 14,016 samples that belong to its 73 prompts (73 x 192). Table 1's 14,016 is the right subset. `results.md` records 19,200 in `budgets.json` because that file stores the parent mixed draw, and the same file's note defines 14,016 as the curated-prompt count.

Reason: a reader of the abstract is told the 10.9-point comparison is the registered one, while the registered sampling-budget sentence was not the experiment that ran.

### 3. Major. The success rule is a coarse yardstick, and "yes" treats it as a decision

PDF p.2 (§2.3) and p.4 (Table 3).

A contrast "counts" when the absolute test gap exceeds twice the pooled seed standard deviation and the mean OOD gap has the same sign. The paper says this is not a formal test. With three seeds per arm the pooled standard deviation has 4 degrees of freedom. Under normality, a 95% range for that standard deviation runs from about 0.60 to 2.87 times the estimate (the factor is already in `results/phase5/how_could_this_be_wrong.md`). For RFT-Mixed minus RFT-Easy, 2σ is 2.1 points, and 2.1 x 2.87 is about 6.0, so the 3.7-point "yes" does not survive uncertainty in σ itself. The 10.9-point gap still clears 2.8 x 2.87, about 8.0.

Reason: Table 3 prints the same "yes" for a gap that is stable under this uncertainty and for gaps that are not.

The OOD clause is a sign check on a three-seed mean. It has no magnitude requirement, and it does not require the seed pairs to agree. GRPO-Curated minus RFT-Curated has OOD seed gaps +6.0, 0.0, and +3.0. IterRFT minus RFT-Curated has OOD seed gaps -2.0, -2.0, and +5.0, and the mean +0.3 is marked "yes".

### 4. Major. The paired interval is problem noise, conditional on three fixed seeds

PDF p.2 (§2.3) and p.4 (Table 3 caption).

The logged interval is a paired per-problem bootstrap of the seed-averaged difference: 10,000 resamples, bootstrap seed 0, analysis hash prefix `3ba220e48a66`. Section 2.3 does say "over problems." The Table 3 caption says only "paired bootstrap 95% interval," next to a 2σ column built from seeds. The interval is conditional on the three trained models. It leaves out seed resampling, the shared RFT draw, the validation sweep, and host-to-host generation noise.

Reason: RFT-Mixed minus RFT-Easy has interval [+0.2, +7.0], so the exclusion of zero is 0.2 points, smaller than the 2.3-point cross-machine movement reported on p.4 and p.5.

### 5. Major. The shared RFT draw makes 2σ too small for the RFT contrasts

PDF p.2 (§2.2), p.4 (§3.3), p.5 (limitations), Table 3.

Kept-sample counts are identical across the three RFT seeds (13,834, 7,961, and 6,653). Those seeds share one base-model draw, so their sample standard deviation omits sampling variance. Any 2σ threshold that includes an RFT arm inherits that omission. The paper states this beside the 3.7-point contrast and again in the limitations. Table 3 still marks RFT-Mixed minus RFT-Easy "yes" (2σ = 2.1, gap = 3.7).

Reason: the rule is easiest to pass where the seed spread leaves out a variance component the design itself removed.

### 6. Major. Hyperparameter selection is part of the RFT data contrasts, and the abstract states a pure data effect

PDF p.1 (abstract), p.2 (§2.3), p.4 (§3.2 and §3.3), p.5 (limitations).

The selected settings match the results-log headers: Easy, 5 x 10^-5 for 8 epochs; Mixed, 5 x 10^-5 for 4 epochs; Curated, 10^-5 for 4 epochs. Each is the best of nine on `val_mixed_100` (n = 100). RFT-Mixed minus RFT-Easy therefore compares mixed prompts at 4 epochs with easy prompts at 8 epochs. RFT-Curated minus RFT-Mixed compares 10^-5 with 5 x 10^-5. The body and the limitations say so. The abstract's sentence "Mixed-difficulty data helps RFT by 3.7 points" does not.

Reason: the 3.7-point contrast is a data contrast and an optimization contrast chosen on one 100-problem validation set.

The direction of the sweep is the one the registration intended for the GRPO-versus-RFT comparison: RFT was tuned and GRPO was not, which makes a GRPO lead harder to dismiss as a weak baseline. That sentence on p.2 is a fair description of H3. It does not license reading the within-RFT rows as clean data effects.

### 7. Major. The truncation bound is quoted where it preserves a GRPO lead and omitted where it removes a small effect

PDF p.1 (abstract), p.3 (§3.1), p.4 (Table 3 and §3.4).

The adverse test bound credits every truncated completion of the lower arm and none of the upper arm. The three reported means, +7.8, +6.0, and +7.2 for curated, easy, and mixed, match `truncation_bounds.md` and `exploratory_grpo_vs_rft.md`. The same logged bound does the following, and the condensation does not say so.

- IterRFT minus RFT-Curated falls from +3.2 to +0.1, interval [-3.0, +3.2], and the 2σ rule fails. Two of the three seed-level bounds are slightly negative.
- RFT-Mixed minus RFT-Easy falls from +3.7 to +1.2, interval [-2.4, +4.7], and the 2σ rule fails.
- On OOD, GRPO-Curated minus RFT-Curated falls from +3.0 to -11.8, interval [-18.7, -5.2]. The sign that produces Table 3's "yes" reverses under the truncation treatment used to defend the test gap.

The prose on p.3 and p.5 refuses an OOD conclusion. That refusal is the warranted claim. "Meets the success criterion" and the abstract's "recover 3.2 of the 10.9-point gap" go past it. The 30% figure is the ratio 3.2/10.9. The two pieces add to 10.9 because they are a partition of the same mean, and the pre-registration says any such fraction is descriptive. The logged bound shows the 3.2-point piece is the part that truncation can erase. Section 3.4 does say the 3.2-point gain is close to the 2.3-point cross-machine figure and that iterated RFT also has more steps (1,963 versus 1,664) and more kept samples (7,828 versus 6,653). Those caveats are real. They are not a substitute for the bound the paper uses on the GRPO rows.

Reason: the shorter text reports the adverse bound for the three gaps that stay positive and still prints "yes" for the small and OOD contrasts whose logged bounds do not.

### 8. Minor. The sweep was seed 1 only

PDF p.2 (§2.3). The fact is in the preprint hyperparameter table and in the `results.md` arm headers, not in the condensation.

Reason: seed 1 chooses the learning rate and the epoch count, and seed 1 is then one of the three seeds in the reported test mean.

### 9. Minor. The easy-set bound of +6.0 hides a thin seed

PDF p.3 (§3.1). Logged test gaps for GRPO-Easy minus RFT-Easy are +11.3, +5.3, and +8.7. Subtracting RFT-Easy's truncation (2.3, 3.3, 1.7 points) leaves about +9.0, +2.0, and +7.0.

Reason: "+6.0" is the mean of three seed pairs, and the middle pair is about +2 points, next to the 2.3-point cross-machine figure.

This row is otherwise the cleanest GRPO-versus-RFT comparison in the paper: rollout counts match at 19,200, neither arm exceeds the 5% test flag (GRPO-Easy seeds 4.3%, 3.3%, 1.3%), and the OOD interval [+3.7, +13.0] excludes zero. It was defined after the test results were seen, and Table 3 says so. Tokens and steps remain unmatched (RFT-Easy 47.2M tokens and 6,920 steps, GRPO-Easy 12.4M tokens and 300 steps).

### 10. Minor. The reference row is not identified as prompt-mismatched

PDF p.2 (scoring paragraph) and p.3 (Table 2).

Qwen3-4B at 74.7% test and 45.5% OOD matches the log and is a single run. The preprint states that reference models were scored with the same prompt inside each model's chat template. The condensation says every run uses one plain-text prompt with no chat template, and it extends only the cap and the extractor, explicitly, to the reference model.

Reason: Table 2 places 74.7 beside the LoRA arms without the template difference the preprint records.

### 11. Minor. Two Table 3 intervals are attributed to the wrong log file

PDF p.6 (reproducibility).

The easy and mixed GRPO-minus-RFT intervals match `results/phase5/appendix/exploratory_grpo_vs_rft.md` (+0.084 [+0.048, +0.121], 2σ = 0.058; +0.092 [+0.048, +0.137], 2σ = 0.016; OOD +0.083 [+0.037, +0.130] and +0.035 [-0.020, +0.090]). They are not rows of `results/phase5/tables/contrasts.md`. The reproducibility paragraph cites only that file. The hash prefix `3ba220e48a66` matches both files.

Reason: the provenance sentence misses the file that holds two of the Table 3 intervals.

### 12. Minor. Iterated RFT resets the optimizer each round

PDF p.2 (§2.2) and p.4 (§3.4). Pre-registration §6: each round continues the previous adapter and starts a fresh AdamW and cosine schedule. Logged steps are 1,956, 1,972, and 1,960, averaging to the reported 1,963.

Reason: the published step count is a sum of three fresh schedules, and the condensation's "continues the previous adapter" leaves that reset out.

### 13. Nit. The fallback extractor is the last integer, not an unspecified integer

PDF p.2 (scoring). SPEC §5 and the preprint appendix: last integer on an answer line, else a boxed integer, else the last integer anywhere; a capped completion is never scored from a trailing integer. The condensation says "else anywhere in the completion."

Reason: a completion that states several integers is scored on the one it ends with, and the shortened sentence does not say so.

### 14. Nit. "Three refreshes" counts the base draw

PDF p.2 and p.4 (§3.4). Round 1 is the first 64 samples per prompt of the shared base draw. Rounds 2 and 3 are sampled from the current policy (pre-registration §6).

Reason: "refreshes the sampler 3 times, against GRPO's 300" counts the initial base draw as a refresh. The pre-registration uses the same 3-versus-300 wording, so this is a label, not a new numerical disagreement.

### 15. Nit. "Most of the gain is on the hard tier" is thin for GRPO-Easy

PDF p.3 (§3.1). Logged hard-tier means: base 29.0%; RFT arms 23.3%, 29.7%, and 26.0%; GRPO-Easy 40.7%, GRPO-Mixed 58.0%, GRPO-Curated 52.7%. Hard-tier movement, weighted by 100/300, is about 56% of GRPO-Easy's gain over base, about 84% of GRPO-Mixed's, and about 75% of GRPO-Curated's.

Reason: "most" is a slim majority for the easy arm. The hard-tier GRPO numbers are also the tier with the highest GRPO truncation (about 5%, 13%, and 13%), which the text states.

### 16. Nit. "Within 1.0 point" is the point estimate

PDF p.4 (§3.2). GRPO-Curated minus GRPO-Mixed is -1.0 with interval [-3.7, +1.7], and the rule is "no".

Reason: the sentence "within 1.0 point" describes the mean, while the logged interval is several points wide. The conclusion that curation is undetectable for GRPO follows from the interval and the rule, not from the point estimate alone.

## Claims checked and found consistent

Recomputed from the logged per-seed greedy accuracies in `results/phase5/tables/results.md`, unless noted as a direct match to a logged contrast. Percentage points are rounded to one decimal as in the paper.

- Table 2 and Table 4, test and OOD greedy accuracy and truncation, for base, six primary arms, iterated RFT, C1, C2, and Qwen3-4B. Means and sample standard deviations match the seeds. The dagger on GRPO-Mixed and GRPO-Curated matches every seed being strictly above 5% on test. GRPO-Easy's largest test truncation is 4.3%, so it is correctly unflagged. The flag is "> 5%", and the paper uses "exceeds" and "more than 5%".
- Table 3 point estimates, 2σ values, and intervals for the nine rows that appear in `contrasts.md`: H3 +10.9 [+6.7, +15.1], 2σ 2.8, OOD +3.0 [-2.5, +8.5]; curation -2.7 [-5.2, 0.0] and -1.0 [-3.7, +1.7]; mixed data +3.7 [+0.2, +7.0] and +4.4 [+1.1, +7.9]; iterated RFT +3.2 [+0.2, +6.2] and +7.7 [+3.7, +11.7]. The upper endpoint 0.0 on the RFT curation interval matches the logged +0.000 and is reported as including zero.
- Table 3 easy and mixed GRPO-minus-RFT rows, matched to `exploratory_grpo_vs_rft.md` rather than to `contrasts.md`: +8.4 [+4.8, +12.1], 2σ 5.8, OOD +8.3 [+3.7, +13.0]; +9.2 [+4.8, +13.7], 2σ 1.6, OOD +3.5 [-2.0, +9.0]. Per-seed test gaps +12.0, +10.3, +10.3 for the curated pair match +0.120 / +0.103 / +0.103.
- Adverse test bounds +7.8, +6.0, and +7.2 match the logged lower bounds. The curated lower bound's logged interval is [+3.3, +12.2] and still clears its scenario-specific 2σ. The paper reports the point estimate only.
- Table 1 budgets. Means of the logged per-seed rows: RFT tokens 47,237,096 / 15,852,320 / 13,377,076 and steps 6,920 / 1,992 / 1,664; GRPO tokens 12,401,097 / 22,994,240 / 20,633,003 and 300 steps; iterated RFT 16,729,128 tokens, 7,828 kept samples, 1,963 steps; C1 9,268,987 tokens; C2 10,844,177 tokens. Easy and mixed available rollouts are 19,200. GRPO's 19,200 equals 300 x 8 x 8. Curated RFT available 14,016 equals 73 x 192.
- RFT-Mixed keeps a correct sample for 95 of 100 prompts, and 595 of 7,961 kept samples are from hard prompts (4,605 + 595 + 2,761 = 7,961). Identical kept counts across RFT seeds support the shared-draw sentence.
- Controls are seed 1 only, and the text uses GRPO-Mixed seed 1 (78.7%), not the three-seed mean (78.8%), against C2 63.7%, C1 47.7%, and base 67.3%. Mean@8 values 42.5% (base) and 60.3% (C2) match the log. The paper calls these numbers descriptive.
- Mean@8 ranges 51.2--54.4 (single-round RFT) and 70.1--77.1 (GRPO), and pass@64 98.0% (base) and 95.7--96.7% (GRPO), match `results.md`. RFT-Easy is 1.4 points below base (67.3 - 65.9). Easy-tier primary-arm means all lie in [92.3, 97.0], so "between 92% and 97%" matches the tier table.
- GSM8K greedy: base 88.0%; RFT arm means 91.9, 90.9, and 91.5, so the stated range 90.9--91.9; GRPO arm means 93.3, 92.7, and 92.9, so 92.7--93.3; iterated RFT 92.1. The largest logged GSM8K greedy truncation in these runs is 1.0%. The paper computes no comparisons on this set.
- The 8.2-point GRPO-Curated minus RFT-Mixed gap matches +0.082. Calling the recovered share negative matches the logged ratio -0.32. The paper does not print -0.32, and it refuses the claim that curation hurts RFT.
- 3.2 + 7.7 = 10.9 holds for the rounded means and for the unrounded logged means 0.032 + 0.077 = 0.109. "About 30%" is 3.2/10.9. The pre-registration says this fraction is descriptive, and §3.4 says so.
- Selected learning rates and epoch counts match the results-log headers. The tie rule (fewer epochs, then lower learning rate) is stated and is the rule the preprint sweep table uses for the Mixed tie at 67%.
- Dates and the deviation list match the pre-registration: hypotheses 2026-09-07, predictions 2026-09-12, first training job 2026-09-13, EOS restart with only validation seen, iterated RFT registered 2026-09-20 after the primary test scores and before iterated-RFT samples, exploratory 8,704-token regeneration excluded from the registered rule. The condensation does not promote the 8,704 numbers into the primary tables.
- "20--55%" OOD GRPO truncation is the logged per-seed range 19.5% to 54.5%, rounded. The refusal of an OOD conclusion matches the size of that range.
- Zero-gradient sentence: the condensation attributes 26--29% in the first 25 steps and 61--70% in the last 25 to the preprint figure, and it says this draft did not recompute them. The wording matches the preprint. See unverified list for the raw logs.
- Cross-machine "up to 2.3 points" on test and "up to 5.0 points" on OOD matches the statement in `results/phase5/README.md`. See unverified list for a recomputation from completions.

## Claims I could not verify

- The 26--29% and 61--70% zero-gradient ranges, from raw step logs. The condensation says they were not recomputed for this draft. I confirmed only that the sentence matches the preprint.
- The cross-machine 2.3-point and 5.0-point figures, from the underlying completions. I confirmed only that `results/phase5/README.md` states them (6--10% of problems flip, 1,396/1,396 identical on the same host).
- The full 3 x 3 validation grid (the preprint's sweep table, including the Mixed tie at 67). I confirmed the selected configuration labels against the results-log headers, not each discarded run's `metrics.json`.
- That the GSM8K prompt is byte-identical to the counting prompt. The accuracies match the log. I did not open the GSM8K eval prompt.
- The reference-model chat template, from an eval config. I relied on the preprint's statement. The condensation does not repeat it (issue 10).
- The notebook claim that truncated greedy completions contain no answer before the cut. The truncation bounds are arithmetic on the logged rates either way; the "no answer" claim is what makes the adverse corner an extreme rather than a correction.
- Bauer et al.'s sample-efficiency result. The condensation says "much more sample-efficient" and does not repeat the preprint's "up to 5x". I did not check the cited paper.
- The exploratory 8,704-token accuracies. The condensation correctly keeps them out of the registered rule. I did not recompute them.
- The bootstrap implementation itself. I checked that the published intervals match the logged tables. I did not rerun the 10,000 resamples.
