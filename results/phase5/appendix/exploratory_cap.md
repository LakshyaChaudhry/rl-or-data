# Appendix — EXPLORATORY re-evaluation at a cap of 8,704 tokens\* (tasks/06b C)

\* re-generated at a cap of 8,704 tokens to show what accuracy would have been without truncation; exploratory, run after the primary results were seen; the policies were trained with completions capped at 4,352. Not a SPEC §10 number: nothing here enters the pre-registered criterion, the primary tables or a headline.

Analysis config hash `3ba220e48a66`. A deviation from SPEC §7 and §10, approved by Laksh and logged in PREREGISTRATION §4 (2026-09-20), decided after the primary test results were seen. The cap was fixed by rule before running (8,704 = 2 × the locked 4,352), never chosen from results. Greedy only; same prompts, extractor, seeds and vLLM settings as the primary units; the results loader refuses these paths. Adapter identity (sha256 in the exploratory config = the adapter hashed by the cross-run sanity): 9 of 9 trained models match.

**Reproducibility of greedy decoding, measured here as a by-product.** A completion that the primary run did not truncate does not depend on the cap, so re-generating it should give the same text. same host as the primary run (6 units: IterRFT-Curated): 1,396 of 1,396 untruncated completions are byte-identical (100.0 %); correctness flips on 0.0–0.0 % of problems per unit, net accuracy change from those flips up to 0.0 points; different host as the primary run (14 units: Base, GRPO-Curated, RFT-Curated): 724 of 3,067 untruncated completions are byte-identical (23.6 %); correctness flips on 6.0–10.0 % of problems per unit, net accuracy change from those flips up to 5.0 points. Sampler settings, package versions, sampling code and GPU model are identical between the two runs of every model; what differs is the cap and, where stated, the host. **Consequence for reading this table:** where the host differs, the 'change' column mixes the effect of the larger cap with re-run differences of untruncated completions; only the 'cut → right' counts are attributable to the cap. **Consequence for the primary tables:** a single greedy evaluation carries machine-level re-run noise of this size that the seed std of an arm evaluated on one host does not contain; it is comparable to the smaller contrasts and small next to the larger ones.

## Per model\*

| model (seed) | split | primary, cap 4,352 [95 % CI] · truncation | cap 8,704\* [95 % CI] · truncation | change\* | cut at 4,352 → at 8,704\* | not cut in primary: identical text; right→wrong / wrong→right\* | host of the two runs | mean tokens\* | config hash |
|---|---|---|---|---|---|---|---|---|---|
| Base | test greedy (n=300) | 0.673 [0.617, 0.727] · tr 4.3% | 0.677\* [0.623, 0.727] · tr 5.0% | +0.003\* | 13 → 2 right, 9 still cut | 92 of 287; 11 / 10 | different | 1131 | f2952b57c22f |
| Base | ood greedy (n=200) | 0.250 [0.190, 0.315] · tr 13.0% | 0.230\* [0.175, 0.290] · tr 11.5% | -0.020\* | 26 → 2 right, 14 still cut | 36 of 174; 13 / 7 | different | 2030 | 63986b01e550 |
| RFT-Curated s1 | test greedy (n=300) | 0.673 [0.620, 0.723] · tr 2.7% | 0.657\* [0.600, 0.710] · tr 2.3% | -0.017\* | 8 → 2 right, 4 still cut | 75 of 292; 18 / 11 | different | 965 | bf38492e8997 |
| RFT-Curated s1 | ood greedy (n=200) | 0.300 [0.240, 0.365] · tr 14.0% | 0.280\* [0.220, 0.345] · tr 6.5% | -0.020\* | 28 → 6 right, 9 still cut | 39 of 172; 15 / 5 | different | 1753 | 36054b951978 |
| RFT-Curated s2 | test greedy (n=300) | 0.657 [0.603, 0.710] · tr 3.3% | 0.690\* [0.637, 0.743] · tr 1.3% | +0.033\* | 10 → 5 right, 2 still cut | 89 of 290; 8 / 13 | different | 908 | 9fd36e60cfb5 |
| RFT-Curated s2 | ood greedy (n=200) | 0.290 [0.225, 0.355] · tr 15.5% | 0.295\* [0.235, 0.360] · tr 8.5% | +0.005\* | 31 → 8 right, 13 still cut | 37 of 169; 11 / 4 | different | 1877 | 23f83e7299c5 |
| RFT-Curated s3 | test greedy (n=300) | 0.677 [0.623, 0.730] · tr 3.3% | 0.663\* [0.610, 0.717] · tr 3.0% | -0.013\* | 10 → 2 right, 6 still cut | 90 of 290; 13 / 7 | different | 984 | ea5c6d9adf15 |
| RFT-Curated s3 | ood greedy (n=200) | 0.265 [0.205, 0.325] · tr 15.0% | 0.295\* [0.235, 0.360] · tr 9.5% | +0.030\* | 30 → 4 right, 12 still cut | 33 of 170; 9 / 11 | different | 1903 | 9ad74c9bffda |
| GRPO-Curated s1 | test greedy (n=300) | 0.793 [0.747, 0.840] · tr 5.3% | 0.823\* [0.780, 0.867] · tr 2.3% | +0.030\* | 16 → 7 right, 4 still cut | 81 of 284; 8 / 10 | different | 1548 | 7a033f9dd82a |
| GRPO-Curated s1 | ood greedy (n=200) | 0.360 [0.295, 0.430] · tr 19.5% | 0.405\* [0.335, 0.475] · tr 11.0% | +0.045\* | 39 → 11 right, 13 still cut | 13 of 161; 11 / 9 | different | 2716 | 31c9f4a1ff34 |
| GRPO-Curated s2 | test greedy (n=300) | 0.760 [0.710, 0.810] · tr 9.3% | 0.833\* [0.790, 0.873] · tr 2.3% | +0.073\* | 28 → 15 right, 7 still cut | 34 of 272; 11 / 18 | different | 1957 | 6cef5e2a88e3 |
| GRPO-Curated s2 | ood greedy (n=200) | 0.290 [0.230, 0.355] · tr 45.0% | 0.395\* [0.330, 0.465] · tr 31.0% | +0.105\* | 90 → 21 right, 47 still cut | 5 of 110; 9 / 9 | different | 4671 | dc276740a25b |
| GRPO-Curated s3 | test greedy (n=300) | 0.780 [0.733, 0.827] · tr 8.0% | 0.803\* [0.757, 0.847] · tr 6.0% | +0.023\* | 24 → 8 right, 12 still cut | 84 of 276; 14 / 13 | different | 1859 | 940e9f6303db |
| GRPO-Curated s3 | ood greedy (n=200) | 0.295 [0.235, 0.360] · tr 40.0% | 0.415\* [0.345, 0.485] · tr 23.0% | +0.120\* | 80 → 24 right, 38 still cut | 16 of 120; 6 / 6 | different | 4417 | 3fe99e372266 |
| IterRFT-Curated s1 | test greedy (n=300) | 0.697 [0.643, 0.747] · tr 1.3% | 0.697\* [0.643, 0.747] · tr 1.3% | +0.000\* | 4 → 0 right, 4 still cut | 296 of 296; 0 / 0 | same | 877 | e0a56e0319c7 |
| IterRFT-Curated s1 | ood greedy (n=200) | 0.280 [0.220, 0.345] · tr 14.5% | 0.290\* [0.230, 0.355] · tr 12.0% | +0.010\* | 29 → 2 right, 24 still cut | 171 of 171; 0 / 0 | same | 2058 | f75fb2082f60 |
| IterRFT-Curated s2 | test greedy (n=300) | 0.700 [0.647, 0.750] · tr 1.7% | 0.700\* [0.647, 0.750] · tr 1.3% | +0.000\* | 5 → 0 right, 4 still cut | 295 of 295; 0 / 0 | same | 895 | 25067088e0e3 |
| IterRFT-Curated s2 | ood greedy (n=200) | 0.270 [0.210, 0.335] · tr 16.0% | 0.280\* [0.220, 0.345] · tr 13.0% | +0.010\* | 32 → 2 right, 26 still cut | 168 of 168; 0 / 0 | same | 2257 | fa19dad4d737 |
| IterRFT-Curated s3 | test greedy (n=300) | 0.707 [0.653, 0.757] · tr 1.7% | 0.707\* [0.653, 0.757] · tr 1.3% | +0.000\* | 5 → 0 right, 4 still cut | 295 of 295; 0 / 0 | same | 849 | ab425d4906a3 |
| IterRFT-Curated s3 | ood greedy (n=200) | 0.315 [0.255, 0.380] · tr 14.5% | 0.325\* [0.260, 0.390] · tr 10.0% | +0.010\* | 29 → 2 right, 20 still cut | 171 of 171; 0 / 0 | same | 2063 | 5f8512fc1ea0 |

## Per arm and between arms\* (seed i vs seed i; no SPEC §10 criterion is evaluated on these)

| arm or contrast | split | per seed at cap 8,704\* | mean ± seed std\* [paired bootstrap 95 % CI for Δ] | primary mean at cap 4,352 | truncation per seed\* | note |
|---|---|---|---|---|---|---|
| RFT-Curated | test greedy | 0.657 / 0.690 / 0.663\* | 0.670 ± 0.018\* | 0.669 | 2.3% / 1.3% / 3.0%\* |  |
| GRPO-Curated | test greedy | 0.823 / 0.833 / 0.803\* | 0.820 ± 0.015\* | 0.778 | 2.3% / 2.3% / 6.0%\* |  |
| IterRFT-Curated | test greedy | 0.697 / 0.700 / 0.707\* | 0.701 ± 0.005\* | 0.701 | 1.3% / 1.3% / 1.3%\* |  |
| GRPO-Curated − RFT-Curated | test greedy | +0.167 / +0.143 / +0.140\* | +0.150 ± 0.015\* [+0.111, +0.191] | +0.109 |  | primary Δ at cap 4,352 for comparison |
| IterRFT-Curated − RFT-Curated | test greedy | +0.040 / +0.010 / +0.043\* | +0.031 ± 0.018\* [-0.000, +0.063] | +0.032 |  | primary Δ at cap 4,352 for comparison |
| GRPO-Curated − IterRFT-Curated | test greedy | +0.127 / +0.133 / +0.097\* | +0.119 ± 0.020\* [+0.082, +0.157] | +0.077 |  | primary Δ at cap 4,352 for comparison |
| RFT-Curated | ood greedy | 0.280 / 0.295 / 0.295\* | 0.290 ± 0.009\* | 0.285 | 6.5% / 8.5% / 9.5%\* |  |
| GRPO-Curated | ood greedy | 0.405 / 0.395 / 0.415\* | 0.405 ± 0.010\* | 0.315 | 11.0% / 31.0% / 23.0%\* |  |
| IterRFT-Curated | ood greedy | 0.290 / 0.280 / 0.325\* | 0.298 ± 0.024\* | 0.288 | 12.0% / 13.0% / 10.0%\* |  |
| GRPO-Curated − RFT-Curated | ood greedy | +0.125 / +0.100 / +0.120\* | +0.115 ± 0.013\* [+0.060, +0.170] | +0.030 |  | primary Δ at cap 4,352 for comparison |
| IterRFT-Curated − RFT-Curated | ood greedy | +0.010 / -0.015 / +0.030\* | +0.008 ± 0.023\* [-0.028, +0.045] | +0.003 |  | primary Δ at cap 4,352 for comparison |
| GRPO-Curated − IterRFT-Curated | ood greedy | +0.115 / +0.115 / +0.090\* | +0.107 ± 0.014\* [+0.058, +0.155] | +0.027 |  | primary Δ at cap 4,352 for comparison |

\* re-generated at a cap of 8,704 tokens to show what accuracy would have been without truncation; exploratory, run after the primary results were seen; the policies were trained with completions capped at 4,352. Not a SPEC §10 number: nothing here enters the pre-registered criterion, the primary tables or a headline.
