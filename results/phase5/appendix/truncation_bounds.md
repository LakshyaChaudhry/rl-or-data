# Appendix — post-hoc truncation bounds (tasks/06b B; approved by Laksh 2026-09-20)

Analysis config hash `3ba220e48a66`; seeds 1–3, greedy, cap 4,352; paired problem bootstrap, seed 0, 10,000 resamples. **Post hoc: defined after the primary results were seen. Not a correction, not an estimate, and not an input to any SPEC §10 criterion line or to `tables/results.md`.** A completion cut at the cap is scored wrong by rule (SPEC §5 v1.6), so each run's accuracy lies in [observed, observed + truncation rate]. The rows give Δ = A − B at the corners of that region; the 2 × pooled-seed-std yardstick is recomputed inside each scenario from that scenario's per-seed accuracies. 'All truncated completions right' is an extreme, not a likely value: in the primary units no truncated greedy completion contains an answer before the cut (notebook 2026-09-20).

## H1_rft: RFT-Mixed − RFT-Easy

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.707 / 0.687 / 0.693 | 0.663 / 0.647 / 0.667 | +0.043 / +0.040 / +0.027 | +0.037 [+0.002, +0.070] | 0.021 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.707 / 0.687 / 0.693 | 0.687 / 0.680 / 0.683 | +0.020 / +0.007 / +0.010 | +0.012 [-0.024, +0.047] | 0.015 | no |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.733 / 0.703 / 0.710 | 0.663 / 0.647 / 0.667 | +0.070 / +0.057 / +0.043 | +0.057 [+0.022, +0.091] | 0.027 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.733 / 0.703 / 0.710 | 0.687 / 0.680 / 0.683 | +0.047 / +0.023 / +0.027 | +0.032 [-0.003, +0.067] | 0.023 | yes |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.280 / 0.280 / 0.260 | 0.250 / 0.260 / 0.255 | +0.030 / +0.020 / +0.005 | +0.018 [-0.020, +0.057] | 0.018 | yes |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.280 / 0.280 / 0.260 | 0.390 / 0.355 / 0.335 | -0.110 / -0.075 / -0.075 | -0.087 [-0.135, -0.040] | 0.043 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.400 / 0.390 / 0.395 | 0.250 / 0.260 / 0.255 | +0.150 / +0.130 / +0.140 | +0.140 [+0.088, +0.192] | 0.010 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.400 / 0.390 / 0.395 | 0.390 / 0.355 / 0.335 | +0.010 / +0.035 / +0.060 | +0.035 [-0.012, +0.082] | 0.040 | no |

- test greedy: truncation A 2.7% / 1.7% / 1.7% vs B 2.3% / 3.3% / 1.7%; Δ lies in [+0.012, +0.057] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 12.0% / 11.0% / 13.5% vs B 14.0% / 9.5% / 8.0%; Δ lies in [-0.087, +0.140] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## H1_grpo: GRPO-Mixed − GRPO-Easy

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.787 / 0.793 / 0.783 | 0.777 / 0.700 / 0.753 | +0.010 / +0.093 / +0.030 | +0.044 [+0.011, +0.079] | 0.056 | no |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.787 / 0.793 / 0.783 | 0.820 / 0.733 / 0.767 | -0.033 / +0.060 / +0.017 | +0.014 [-0.022, +0.052] | 0.062 | no |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.853 / 0.897 / 0.873 | 0.777 / 0.700 / 0.753 | +0.077 / +0.197 / +0.120 | +0.131 [+0.098, +0.166] | 0.063 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.853 / 0.897 / 0.873 | 0.820 / 0.733 / 0.767 | +0.033 / +0.163 / +0.107 | +0.101 [+0.069, +0.133] | 0.069 | yes |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.295 / 0.295 / 0.335 | 0.345 / 0.305 / 0.365 | -0.050 / -0.010 / -0.030 | -0.030 [-0.073, +0.012] | 0.054 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.295 / 0.295 / 0.335 | 0.725 / 0.515 / 0.560 | -0.430 / -0.220 / -0.225 | -0.292 [-0.352, -0.232] | 0.160 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.760 / 0.840 / 0.750 | 0.345 / 0.305 / 0.365 | +0.415 / +0.535 / +0.385 | +0.445 [+0.388, +0.500] | 0.082 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.760 / 0.840 / 0.750 | 0.725 / 0.515 / 0.560 | +0.035 / +0.325 / +0.190 | +0.183 [+0.138, +0.228] | 0.171 | yes |

- test greedy: truncation A 6.7% / 10.3% / 9.0% vs B 4.3% / 3.3% / 1.3%; Δ lies in [+0.014, +0.131] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 46.5% / 54.5% / 41.5% vs B 38.0% / 21.0% / 19.5%; Δ lies in [-0.292, +0.445] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## H2_num: RFT-Curated − RFT-Mixed

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.673 / 0.657 / 0.677 | 0.707 / 0.687 / 0.693 | -0.033 / -0.030 / -0.017 | -0.027 [-0.052, +0.000] | 0.021 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.673 / 0.657 / 0.677 | 0.733 / 0.703 / 0.710 | -0.060 / -0.047 / -0.033 | -0.047 [-0.076, -0.019] | 0.027 | yes |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.700 / 0.690 / 0.710 | 0.707 / 0.687 / 0.693 | -0.007 / +0.003 / +0.017 | +0.004 [-0.022, +0.032] | 0.020 | no |
| test greedy (n=300) | both arms' truncated completions all right | 0.700 / 0.690 / 0.710 | 0.733 / 0.703 / 0.710 | -0.033 / -0.013 / +0.000 | -0.016 [-0.041, +0.011] | 0.026 | no |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.300 / 0.290 / 0.265 | 0.280 / 0.280 / 0.260 | +0.020 / +0.010 / +0.005 | +0.012 [-0.025, +0.053] | 0.030 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.300 / 0.290 / 0.265 | 0.400 / 0.390 / 0.395 | -0.100 / -0.100 / -0.130 | -0.110 [-0.162, -0.057] | 0.026 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.440 / 0.445 / 0.415 | 0.280 / 0.280 / 0.260 | +0.160 / +0.165 / +0.155 | +0.160 [+0.112, +0.210] | 0.028 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.440 / 0.445 / 0.415 | 0.400 / 0.390 / 0.395 | +0.040 / +0.055 / +0.020 | +0.038 [-0.005, +0.085] | 0.024 | yes |

- test greedy: truncation A 2.7% / 3.3% / 3.3% vs B 2.7% / 1.7% / 1.7%; Δ lies in [-0.047, +0.004] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.
- ood greedy: truncation A 14.0% / 15.5% / 15.0% vs B 12.0% / 11.0% / 13.5%; Δ lies in [-0.110, +0.160] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## H2_den: GRPO-Curated − RFT-Mixed

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.793 / 0.760 / 0.780 | 0.707 / 0.687 / 0.693 | +0.087 / +0.073 / +0.087 | +0.082 [+0.040, +0.126] | 0.028 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.793 / 0.760 / 0.780 | 0.733 / 0.703 / 0.710 | +0.060 / +0.057 / +0.070 | +0.062 [+0.019, +0.107] | 0.033 | yes |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.847 / 0.853 / 0.860 | 0.707 / 0.687 / 0.693 | +0.140 / +0.167 / +0.167 | +0.158 [+0.118, +0.198] | 0.017 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.847 / 0.853 / 0.860 | 0.733 / 0.703 / 0.710 | +0.113 / +0.150 / +0.150 | +0.138 [+0.099, +0.177] | 0.024 | yes |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.360 / 0.290 / 0.295 | 0.280 / 0.280 / 0.260 | +0.080 / +0.010 / +0.035 | +0.042 [-0.012, +0.095] | 0.058 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.360 / 0.290 / 0.295 | 0.400 / 0.390 / 0.395 | -0.040 / -0.100 / -0.100 | -0.080 [-0.143, -0.018] | 0.056 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.555 / 0.740 / 0.695 | 0.280 / 0.280 / 0.260 | +0.275 / +0.460 / +0.435 | +0.390 [+0.338, +0.443] | 0.137 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.555 / 0.740 / 0.695 | 0.400 / 0.390 / 0.395 | +0.155 / +0.350 / +0.300 | +0.268 [+0.213, +0.325] | 0.137 | yes |

- test greedy: truncation A 5.3% / 9.3% / 8.0% vs B 2.7% / 1.7% / 1.7%; Δ lies in [+0.062, +0.158] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 19.5% / 45.0% / 40.0% vs B 12.0% / 11.0% / 13.5%; Δ lies in [-0.080, +0.390] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## H3: GRPO-Curated − RFT-Curated

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.793 / 0.760 / 0.780 | 0.673 / 0.657 / 0.677 | +0.120 / +0.103 / +0.103 | +0.109 [+0.067, +0.151] | 0.028 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.793 / 0.760 / 0.780 | 0.700 / 0.690 / 0.710 | +0.093 / +0.070 / +0.070 | +0.078 [+0.033, +0.122] | 0.028 | yes |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.847 / 0.853 / 0.860 | 0.673 / 0.657 / 0.677 | +0.173 / +0.197 / +0.183 | +0.184 [+0.143, +0.226] | 0.018 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.847 / 0.853 / 0.860 | 0.700 / 0.690 / 0.710 | +0.147 / +0.163 / +0.150 | +0.153 [+0.113, +0.193] | 0.017 | yes |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.360 / 0.290 / 0.295 | 0.300 / 0.290 / 0.265 | +0.060 / +0.000 / +0.030 | +0.030 [-0.025, +0.085] | 0.061 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.360 / 0.290 / 0.295 | 0.440 / 0.445 / 0.415 | -0.080 / -0.155 / -0.120 | -0.118 [-0.187, -0.052] | 0.060 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.555 / 0.740 / 0.695 | 0.300 / 0.290 / 0.265 | +0.255 / +0.450 / +0.430 | +0.378 [+0.322, +0.433] | 0.139 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.555 / 0.740 / 0.695 | 0.440 / 0.445 / 0.415 | +0.115 / +0.295 / +0.280 | +0.230 [+0.170, +0.288] | 0.138 | yes |

- test greedy: truncation A 5.3% / 9.3% / 8.0% vs B 2.7% / 3.3% / 3.3%; Δ lies in [+0.078, +0.184] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 19.5% / 45.0% / 40.0% vs B 14.0% / 15.5% / 15.0%; Δ lies in [-0.118, +0.378] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## S1_iter_vs_rft: IterRFT-Curated − RFT-Curated

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.697 / 0.700 / 0.707 | 0.673 / 0.657 / 0.677 | +0.023 / +0.043 / +0.030 | +0.032 [+0.002, +0.062] | 0.017 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.697 / 0.700 / 0.707 | 0.700 / 0.690 / 0.710 | -0.003 / +0.010 / -0.003 | +0.001 [-0.030, +0.032] | 0.016 | no |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.710 / 0.717 / 0.723 | 0.673 / 0.657 / 0.677 | +0.037 / +0.060 / +0.047 | +0.048 [+0.016, +0.079] | 0.018 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.710 / 0.717 / 0.723 | 0.700 / 0.690 / 0.710 | +0.010 / +0.027 / +0.013 | +0.017 [-0.013, +0.047] | 0.017 | no |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.280 / 0.270 / 0.315 | 0.300 / 0.290 / 0.265 | -0.020 / -0.020 / +0.050 | +0.003 [-0.035, +0.038] | 0.042 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.280 / 0.270 / 0.315 | 0.440 / 0.445 / 0.415 | -0.160 / -0.175 / -0.100 | -0.145 [-0.197, -0.095] | 0.040 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.425 / 0.430 / 0.460 | 0.300 / 0.290 / 0.265 | +0.125 / +0.140 / +0.195 | +0.153 [+0.105, +0.202] | 0.037 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.425 / 0.430 / 0.460 | 0.440 / 0.445 / 0.415 | -0.015 / -0.015 / +0.045 | +0.005 [-0.045, +0.053] | 0.035 | no |

- test greedy: truncation A 1.3% / 1.7% / 1.7% vs B 2.7% / 3.3% / 3.3%; Δ lies in [+0.001, +0.048] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 14.5% / 16.0% / 14.5% vs B 14.0% / 15.5% / 15.0%; Δ lies in [-0.145, +0.153] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

## S2_grpo_vs_iter: GRPO-Curated − IterRFT-Curated

| split | scenario | A per seed | B per seed | Δ per seed | mean Δ [paired bootstrap 95 % CI] | 2 × pooled seed std (this scenario) | abs(Δ) above it |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | observed: every truncated completion wrong (the primary tables) | 0.793 / 0.760 / 0.780 | 0.697 / 0.700 / 0.707 | +0.097 / +0.060 / +0.073 | +0.077 [+0.037, +0.117] | 0.025 | yes |
| test greedy (n=300) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.793 / 0.760 / 0.780 | 0.710 / 0.717 / 0.723 | +0.083 / +0.043 / +0.057 | +0.061 [+0.021, +0.102] | 0.026 | yes |
| test greedy (n=300) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.847 / 0.853 / 0.860 | 0.697 / 0.700 / 0.707 | +0.150 / +0.153 / +0.153 | +0.152 [+0.116, +0.189] | 0.012 | yes |
| test greedy (n=300) | both arms' truncated completions all right | 0.847 / 0.853 / 0.860 | 0.710 / 0.717 / 0.723 | +0.137 / +0.137 / +0.137 | +0.137 [+0.100, +0.172] | 0.013 | yes |
| ood greedy (n=200) | observed: every truncated completion wrong (the primary tables) | 0.360 / 0.290 / 0.295 | 0.280 / 0.270 / 0.315 | +0.080 / +0.020 / -0.020 | +0.027 [-0.022, +0.073] | 0.065 | no |
| ood greedy (n=200) | lower bound of Δ: B's truncated completions all right, A's all wrong | 0.360 / 0.290 / 0.295 | 0.425 / 0.430 / 0.460 | -0.065 / -0.140 / -0.165 | -0.123 [-0.183, -0.065] | 0.061 | yes |
| ood greedy (n=200) | upper bound of Δ: A's truncated completions all right, B's all wrong | 0.555 / 0.740 / 0.695 | 0.280 / 0.270 / 0.315 | +0.275 / +0.470 / +0.380 | +0.375 [+0.323, +0.427] | 0.140 | yes |
| ood greedy (n=200) | both arms' truncated completions all right | 0.555 / 0.740 / 0.695 | 0.425 / 0.430 / 0.460 | +0.130 / +0.310 / +0.235 | +0.225 [+0.175, +0.275] | 0.139 | yes |

- test greedy: truncation A 5.3% / 9.3% / 8.0% vs B 1.3% / 1.7% / 1.7%; Δ lies in [+0.061, +0.152] — the sign of Δ is the same at both ends (positive).
- ood greedy: truncation A 19.5% / 45.0% / 40.0% vs B 14.5% / 16.0% / 14.5%; Δ lies in [-0.123, +0.375] — the interval contains 0: **these bounds do not identify the sign of Δ (uninformative)**.

On ood_hard_200 truncation is 8.0%–54.5% per run in these contrasts, the interval for Δ is 23–74 points wide, and it contains 0 for 7 of 7 contrasts: **the ood bounds are uninformative**, including for criterion (b), which reads a sign on that split. They are printed so that nobody has to take that on trust.
