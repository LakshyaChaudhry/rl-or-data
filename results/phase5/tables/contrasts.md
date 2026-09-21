# Paired contrasts (tasks/05 item 3)

Analysis config hash `3ba220e48a66`; bootstrap seed 0, 10,000 resamples. Δ = A − B, paired seed-wise. The CI is a paired per-problem bootstrap of the seed-averaged difference (both arms are scored on the same problems). Pooled seed std = sqrt of the pooled sample variance (ddof = 1) of the two arms' per-seed accuracies. The SPEC §10 criterion is evaluated mechanically on test_300 greedy and ood_hard_200 greedy; nothing here is a verdict. **Truncation is printed next to every Δ because it differs systematically between RFT and GRPO and a truncated completion is scored wrong by rule.**

## H1_rft (H1): RFT-Mixed − RFT-Easy

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.043 / +0.040 / +0.027 | +0.037 ± 0.009 [+0.002, +0.070] | 0.010 | 2.7% / 1.7% / 1.7% vs 2.3% / 3.3% / 1.7% | -0.4 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.030 / +0.020 / +0.005 | +0.018 ± 0.013 [-0.020, +0.057] | 0.009 | 12.0% / 11.0% / 13.5% vs 14.0% / 9.5% / 8.0% | +1.7 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.000 / +0.030 / -0.040 | -0.003 ± 0.035 [-0.070, +0.063] | 0.022 | 1.0% / 1.0% / 2.0% vs 0.0% / 0.0% / 1.0% | +1.0 pp | yes |
| test mean@8 (n=300) | 1,2,3 | +0.012 / -0.003 / +0.016 | +0.008 ± 0.010 [-0.007, +0.024] | 0.008 | 0.4% / 0.5% / 0.4% vs 0.7% / 0.2% / 0.3% | +0.1 pp | yes |
| ood mean@8 (n=200) | 1,2,3 | -0.016 / -0.006 / -0.003 | -0.008 ± 0.007 [-0.023, +0.007] | 0.006 | 3.1% / 3.4% / 3.1% vs 2.4% / 2.9% / 2.5% | +0.6 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.023 / +0.015 / +0.008 | +0.016 ± 0.008 [+0.003, +0.028] | 0.005 | 0.7% / 0.4% / 0.8% vs 0.4% / 0.5% / 0.3% | +0.2 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.040 / +0.010 / +0.020 | +0.023 ± 0.015 [-0.003, +0.053] | 0.008 | 0.7% / 0.4% / 0.8% vs 0.4% / 0.5% / 0.3% | +0.2 pp | no |
| gsm8k greedy (n=500) | 1,2,3 | -0.016 / -0.014 / +0.000 | -0.010 ± 0.009 [-0.029, +0.009] | 0.006 | 0.0% / 0.0% / 0.0% vs 0.0% / 0.0% / 0.0% | +0.0 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.021: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- no run in this contrast exceeds 5 % truncation on test_300 greedy

## H1_grpo (H1): GRPO-Mixed − GRPO-Easy

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.010 / +0.093 / +0.030 | +0.044 ± 0.044 [+0.011, +0.079] | 0.028 | 6.7% / 10.3% / 9.0% vs 4.3% / 3.3% / 1.3% | +5.7 pp | no |
| ood greedy (n=200) | 1,2,3 | -0.050 / -0.010 / -0.030 | -0.030 ± 0.020 [-0.073, +0.012] | 0.027 | 46.5% / 54.5% / 41.5% vs 38.0% / 21.0% / 19.5% | +21.3 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.040 / +0.100 / +0.030 | +0.057 ± 0.038 [+0.003, +0.113] | 0.033 | 3.0% / 9.0% / 8.0% vs 5.0% / 3.0% / 1.0% | +3.7 pp | no |
| test mean@8 (n=300) | 1,2,3 | +0.055 / +0.093 / +0.060 | +0.069 ± 0.020 [+0.048, +0.090] | 0.019 | 5.3% / 8.2% / 6.8% vs 2.5% / 1.2% / 2.0% | +4.9 pp | no |
| ood mean@8 (n=200) | 1,2,3 | -0.016 / -0.029 / -0.009 | -0.018 ± 0.010 [-0.044, +0.008] | 0.014 | 32.1% / 48.3% / 35.9% vs 22.6% / 10.1% / 11.9% | +23.9 pp | yes |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.003 / +0.072 / +0.047 | +0.041 ± 0.035 [+0.010, +0.074] | 0.027 | 4.7% / 7.3% / 5.2% vs 1.6% / 1.2% / 1.2% | +4.4 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | -0.020 / +0.030 / +0.010 | +0.007 ± 0.025 [-0.007, +0.020] | 0.014 | 4.7% / 7.3% / 5.2% vs 1.6% / 1.2% / 1.2% | +4.4 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.000 / -0.028 / +0.012 | -0.005 ± 0.021 [-0.019, +0.009] | 0.012 | 0.2% / 0.0% / 0.2% vs 0.4% / 0.0% / 0.0% | +0.0 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.056: **no**; (b) same sign on ood: **no** → criterion **NOT MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Mixed seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

## H2_num (H2): RFT-Curated − RFT-Mixed

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | -0.033 / -0.030 / -0.017 | -0.027 ± 0.009 [-0.052, +0.000] | 0.010 | 2.7% / 3.3% / 3.3% vs 2.7% / 1.7% / 1.7% | +1.1 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.020 / +0.010 / +0.005 | +0.012 ± 0.008 [-0.025, +0.053] | 0.015 | 14.0% / 15.5% / 15.0% vs 12.0% / 11.0% / 13.5% | +2.7 pp | yes |
| val greedy (n=100) | 1,2,3 | -0.010 / +0.000 / +0.000 | -0.003 ± 0.006 [-0.060, +0.053] | 0.019 | 0.0% / 1.0% / 0.0% vs 1.0% / 1.0% / 2.0% | -1.0 pp | yes |
| test mean@8 (n=300) | 1,2,3 | -0.037 / -0.030 / -0.030 | -0.032 ± 0.004 [-0.046, -0.018] | 0.004 | 0.5% / 0.7% / 0.5% vs 0.4% / 0.5% / 0.4% | +0.1 pp | no |
| ood mean@8 (n=200) | 1,2,3 | -0.001 / -0.021 / -0.035 | -0.019 ± 0.017 [-0.033, -0.005] | 0.009 | 3.1% / 3.1% / 3.1% vs 3.1% / 3.4% / 3.1% | -0.1 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | -0.013 / -0.019 / -0.023 | -0.018 ± 0.005 [-0.030, -0.008] | 0.004 | 0.6% / 0.4% / 0.5% vs 0.7% / 0.4% / 0.8% | -0.1 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | -0.020 / -0.020 / +0.010 | -0.010 ± 0.017 [-0.030, +0.007] | 0.012 | 0.6% / 0.4% / 0.5% vs 0.7% / 0.4% / 0.8% | -0.1 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.012 / +0.010 / -0.004 | +0.006 ± 0.009 [-0.009, +0.022] | 0.006 | 0.0% / 0.0% / 0.2% vs 0.0% / 0.0% / 0.0% | +0.1 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.021: **yes**; (b) same sign on ood: **no** → criterion **NOT MET** (mechanical evaluation, not a verdict)
- no run in this contrast exceeds 5 % truncation on test_300 greedy

## H2_den (H2): GRPO-Curated − RFT-Mixed

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.087 / +0.073 / +0.087 | +0.082 ± 0.008 [+0.040, +0.126] | 0.014 | 5.3% / 9.3% / 8.0% vs 2.7% / 1.7% / 1.7% | +5.6 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.080 / +0.010 / +0.035 | +0.042 ± 0.035 [-0.012, +0.095] | 0.029 | 19.5% / 45.0% / 40.0% vs 12.0% / 11.0% / 13.5% | +22.7 pp | no |
| val greedy (n=100) | 1,2,3 | +0.100 / +0.110 / +0.140 | +0.117 ± 0.021 [+0.060, +0.177] | 0.015 | 2.0% / 4.0% / 5.0% vs 1.0% / 1.0% / 2.0% | +2.3 pp | no |
| test mean@8 (n=300) | 1,2,3 | +0.222 / +0.220 / +0.215 | +0.219 ± 0.003 [+0.193, +0.246] | 0.007 | 1.8% / 7.1% / 6.0% vs 0.4% / 0.5% / 0.4% | +4.5 pp | no |
| ood mean@8 (n=200) | 1,2,3 | +0.149 / +0.083 / +0.108 | +0.113 ± 0.033 [+0.077, +0.150] | 0.019 | 12.9% / 35.2% / 28.4% vs 3.1% / 3.4% / 3.1% | +22.3 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.076 / +0.083 / +0.076 | +0.078 ± 0.004 [+0.033, +0.125] | 0.002 | 1.8% / 6.0% / 4.2% vs 0.7% / 0.4% / 0.8% | +3.4 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.000 / +0.010 / +0.000 | +0.003 ± 0.006 [-0.027, +0.030] | 0.006 | 1.8% / 6.0% / 4.2% vs 0.7% / 0.4% / 0.8% | +3.4 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.022 / +0.014 / +0.022 | +0.019 ± 0.005 [+0.001, +0.038] | 0.009 | 0.0% / 1.0% / 0.4% vs 0.0% / 0.0% / 0.0% | +0.5 pp | no |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.028: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

## H3 (H3): GRPO-Curated − RFT-Curated

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.120 / +0.103 / +0.103 | +0.109 ± 0.010 [+0.067, +0.151] | 0.014 | 5.3% / 9.3% / 8.0% vs 2.7% / 3.3% / 3.3% | +4.4 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.060 / +0.000 / +0.030 | +0.030 ± 0.030 [-0.025, +0.085] | 0.030 | 19.5% / 45.0% / 40.0% vs 14.0% / 15.5% / 15.0% | +20.0 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.110 / +0.110 / +0.140 | +0.120 ± 0.017 [+0.060, +0.180] | 0.012 | 2.0% / 4.0% / 5.0% vs 0.0% / 1.0% / 0.0% | +3.3 pp | no |
| test mean@8 (n=300) | 1,2,3 | +0.259 / +0.251 / +0.245 | +0.252 ± 0.007 [+0.225, +0.278] | 0.006 | 1.8% / 7.1% / 6.0% vs 0.5% / 0.7% / 0.5% | +4.4 pp | no |
| ood mean@8 (n=200) | 1,2,3 | +0.149 / +0.104 / +0.143 | +0.132 ± 0.025 [+0.099, +0.166] | 0.020 | 12.9% / 35.2% / 28.4% vs 3.1% / 3.1% / 3.1% | +22.4 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.089 / +0.101 / +0.099 | +0.096 ± 0.007 [+0.049, +0.147] | 0.004 | 1.8% / 6.0% / 4.2% vs 0.6% / 0.4% / 0.5% | +3.5 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.020 / +0.030 / -0.010 | +0.013 ± 0.021 [-0.017, +0.043] | 0.012 | 1.8% / 6.0% / 4.2% vs 0.6% / 0.4% / 0.5% | +3.5 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.010 / +0.004 / +0.026 | +0.013 ± 0.011 [-0.005, +0.033] | 0.007 | 0.0% / 1.0% / 0.4% vs 0.0% / 0.0% / 0.2% | +0.4 pp | no |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.028: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

## C1 (controls): C1 random reward − Base

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1 | -0.197 | -0.197 ± n/a (1 seed) [-0.253, -0.143] | n/a (1 seed) | 4.7% vs 4.3% | +0.3 pp | n/a (1 seed) |
| ood greedy (n=200) | 1 | -0.030 | -0.030 ± n/a (1 seed) [-0.090, +0.030] | n/a (1 seed) | 7.5% vs 13.0% | -5.5 pp | n/a (1 seed) |
| val greedy (n=100) | 1 | -0.160 | -0.160 ± n/a (1 seed) [-0.250, -0.070] | n/a (1 seed) | 3.0% vs 1.0% | +2.0 pp | n/a (1 seed) |
| test mean@8 (n=300) | 1 | -0.089 | -0.089 ± n/a (1 seed) [-0.113, -0.066] | n/a (1 seed) | 0.2% vs 0.3% | -0.1 pp | n/a (1 seed) |
| ood mean@8 (n=200) | 1 | -0.033 | -0.033 ± n/a (1 seed) [-0.054, -0.013] | n/a (1 seed) | 1.4% vs 3.1% | -1.8 pp | n/a (1 seed) |
| pass@8 (test first-100, 64 samples) (n=100) | 1 | -0.056 | -0.056 ± n/a (1 seed) [-0.088, -0.025] | n/a (1 seed) | 0.2% vs 0.5% | -0.3 pp | n/a (1 seed) |
| pass@64 (test first-100, 64 samples) (n=100) | 1 | -0.070 | -0.070 ± n/a (1 seed) [-0.130, -0.020] | n/a (1 seed) | 0.2% vs 0.5% | -0.3 pp | n/a (1 seed) |
| gsm8k greedy (n=500) | 1 | -0.030 | -0.030 ± n/a (1 seed) [-0.058, -0.004] | n/a (1 seed) | 0.4% vs 0.0% | +0.4 pp | n/a (1 seed) |

- SPEC §10 criterion: (a) **not evaluable**: a pooled seed std needs ≥ 2 seeds in both arms (this contrast has one); (b) same sign on ood: **yes**
- no run in this contrast exceeds 5 % truncation on test_300 greedy

## C2 (controls): C2 format only − Base

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1 | -0.037 | -0.037 ± n/a (1 seed) [-0.083, +0.010] | n/a (1 seed) | 1.0% vs 4.3% | -3.3 pp | n/a (1 seed) |
| ood greedy (n=200) | 1 | +0.020 | +0.020 ± n/a (1 seed) [-0.045, +0.085] | n/a (1 seed) | 10.5% vs 13.0% | -2.5 pp | n/a (1 seed) |
| val greedy (n=100) | 1 | +0.020 | +0.020 ± n/a (1 seed) [-0.060, +0.110] | n/a (1 seed) | 1.0% vs 1.0% | +0.0 pp | n/a (1 seed) |
| test mean@8 (n=300) | 1 | +0.178 | +0.178 ± n/a (1 seed) [+0.152, +0.204] | n/a (1 seed) | 0.2% vs 0.3% | -0.1 pp | n/a (1 seed) |
| ood mean@8 (n=200) | 1 | +0.091 | +0.091 ± n/a (1 seed) [+0.065, +0.118] | n/a (1 seed) | 2.1% vs 3.1% | -1.1 pp | n/a (1 seed) |
| pass@8 (test first-100, 64 samples) (n=100) | 1 | +0.024 | +0.024 ± n/a (1 seed) [+0.001, +0.049] | n/a (1 seed) | 0.3% vs 0.5% | -0.3 pp | n/a (1 seed) |
| pass@64 (test first-100, 64 samples) (n=100) | 1 | -0.040 | -0.040 ± n/a (1 seed) [-0.080, -0.010] | n/a (1 seed) | 0.3% vs 0.5% | -0.3 pp | n/a (1 seed) |
| gsm8k greedy (n=500) | 1 | +0.052 | +0.052 ± n/a (1 seed) [+0.028, +0.078] | n/a (1 seed) | 0.0% vs 0.0% | +0.0 pp | n/a (1 seed) |

- SPEC §10 criterion: (a) **not evaluable**: a pooled seed std needs ≥ 2 seeds in both arms (this contrast has one); (b) same sign on ood: **no**
- no run in this contrast exceeds 5 % truncation on test_300 greedy

## aux_grpo_curation (aux): GRPO-Curated − GRPO-Mixed

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.007 / -0.033 / -0.003 | -0.010 ± 0.021 [-0.037, +0.017] | 0.012 | 5.3% / 9.3% / 8.0% vs 6.7% / 10.3% / 9.0% | -1.1 pp | yes |
| ood greedy (n=200) | 1,2,3 | +0.065 / -0.005 / -0.040 | +0.007 ± 0.053 [-0.027, +0.040] | 0.032 | 19.5% / 45.0% / 40.0% vs 46.5% / 54.5% / 41.5% | -12.7 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.000 / +0.020 / +0.010 | +0.010 ± 0.010 [-0.040, +0.060] | 0.007 | 2.0% / 4.0% / 5.0% vs 3.0% / 9.0% / 8.0% | -3.0 pp | yes |
| test mean@8 (n=300) | 1,2,3 | -0.006 / -0.004 / -0.012 | -0.007 ± 0.004 [-0.020, +0.005] | 0.008 | 1.8% / 7.1% / 6.0% vs 5.3% / 8.2% / 6.8% | -1.8 pp | yes |
| ood mean@8 (n=200) | 1,2,3 | +0.056 / +0.031 / +0.022 | +0.036 ± 0.017 [+0.018, +0.055] | 0.022 | 12.9% / 35.2% / 28.4% vs 32.1% / 48.3% / 35.9% | -13.2 pp | yes |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.019 / +0.023 / -0.008 | +0.012 ± 0.017 [+0.003, +0.021] | 0.011 | 1.8% / 6.0% / 4.2% vs 4.7% / 7.3% / 5.2% | -1.7 pp | yes |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.020 / +0.000 / -0.010 | +0.003 ± 0.015 [-0.007, +0.013] | 0.009 | 1.8% / 6.0% / 4.2% vs 4.7% / 7.3% / 5.2% | -1.7 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | -0.006 / +0.000 / +0.010 | +0.001 ± 0.008 [-0.011, +0.013] | 0.008 | 0.0% / 1.0% / 0.4% vs 0.2% / 0.0% / 0.2% | +0.3 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.025: **no**; (b) same sign on ood: **no** → criterion **NOT MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 and GRPO-Mixed seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

## S1_iter_vs_rft (secondary (registered after unblinding)): IterRFT-Curated − RFT-Curated

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.023 / +0.043 / +0.030 | +0.032 ± 0.010 [+0.002, +0.062] | 0.008 | 1.3% / 1.7% / 1.7% vs 2.7% / 3.3% / 3.3% | -1.6 pp | no |
| ood greedy (n=200) | 1,2,3 | -0.020 / -0.020 / +0.050 | +0.003 ± 0.040 [-0.035, +0.038] | 0.021 | 14.5% / 16.0% / 14.5% vs 14.0% / 15.5% / 15.0% | +0.2 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.030 / +0.010 / +0.020 | +0.020 ± 0.010 [-0.027, +0.070] | 0.019 | 2.0% / 0.0% / 2.0% vs 0.0% / 1.0% / 0.0% | +1.0 pp | yes |
| test mean@8 (n=300) | 1,2,3 | +0.075 / +0.073 / +0.074 | +0.074 ± 0.001 [+0.060, +0.088] | 0.002 | 0.5% / 0.5% / 0.4% vs 0.5% / 0.7% / 0.5% | -0.1 pp | no |
| ood mean@8 (n=200) | 1,2,3 | +0.025 / +0.037 / +0.041 | +0.034 ± 0.008 [+0.021, +0.048] | 0.008 | 3.1% / 3.5% / 4.1% vs 3.1% / 3.1% / 3.1% | +0.4 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.023 / +0.024 / +0.030 | +0.025 ± 0.004 [+0.013, +0.039] | 0.004 | 0.4% / 0.7% / 0.4% vs 0.6% / 0.4% / 0.5% | +0.0 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.030 / +0.010 / +0.000 | +0.013 ± 0.015 [-0.003, +0.033] | 0.015 | 0.4% / 0.7% / 0.4% vs 0.6% / 0.4% / 0.5% | +0.0 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.002 / +0.004 / +0.012 | +0.006 ± 0.005 [-0.007, +0.019] | 0.004 | 0.0% / 0.0% / 0.0% vs 0.0% / 0.0% / 0.2% | -0.1 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.017: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- no run in this contrast exceeds 5 % truncation on test_300 greedy

## S2_grpo_vs_iter (secondary (registered after unblinding)): GRPO-Curated − IterRFT-Curated

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.097 / +0.060 / +0.073 | +0.077 ± 0.019 [+0.037, +0.117] | 0.012 | 5.3% / 9.3% / 8.0% vs 1.3% / 1.7% / 1.7% | +6.0 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.080 / +0.020 / -0.020 | +0.027 ± 0.050 [-0.022, +0.073] | 0.032 | 19.5% / 45.0% / 40.0% vs 14.5% / 16.0% / 14.5% | +19.8 pp | yes |
| val greedy (n=100) | 1,2,3 | +0.080 / +0.100 / +0.120 | +0.100 ± 0.020 [+0.043, +0.157] | 0.014 | 2.0% / 4.0% / 5.0% vs 2.0% / 0.0% / 2.0% | +2.3 pp | no |
| test mean@8 (n=300) | 1,2,3 | +0.184 / +0.178 / +0.171 | +0.178 ± 0.006 [+0.152, +0.204] | 0.006 | 1.8% / 7.1% / 6.0% vs 0.5% / 0.5% / 0.4% | +4.5 pp | no |
| ood mean@8 (n=200) | 1,2,3 | +0.124 / +0.066 / +0.102 | +0.098 ± 0.029 [+0.065, +0.130] | 0.019 | 12.9% / 35.2% / 28.4% vs 3.1% / 3.5% / 4.1% | +22.0 pp | no |
| pass@8 (test first-100, 64 samples) (n=100) | 1,2,3 | +0.066 / +0.078 / +0.069 | +0.071 ± 0.006 [+0.027, +0.117] | 0.003 | 1.8% / 6.0% / 4.2% vs 0.4% / 0.7% / 0.4% | +3.5 pp | no |
| pass@64 (test first-100, 64 samples) (n=100) | 1,2,3 | -0.010 / +0.020 / -0.010 | +0.000 ± 0.017 [-0.030, +0.023] | 0.012 | 1.8% / 6.0% / 4.2% vs 0.4% / 0.7% / 0.4% | +3.5 pp | yes |
| gsm8k greedy (n=500) | 1,2,3 | +0.008 / +0.000 / +0.014 | +0.007 ± 0.007 [-0.009, +0.024] | 0.008 | 0.0% / 1.0% / 0.4% vs 0.0% / 0.0% / 0.0% | +0.5 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.025: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

## H2 ratio: (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed)

| metric | seeds | numerator Δ (truncation A vs B) | denominator Δ (truncation A vs B) | ratio | ratio per seed | joint problem-bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | -0.027 (tr 2.7% / 3.3% / 3.3% vs 2.7% / 1.7% / 1.7%) | +0.082 (tr 5.3% / 9.3% / 8.0% vs 2.7% / 1.7% / 1.7%) | -0.32 | -0.38 / -0.41 / -0.19 | [-1.00, +0.00] |
| ood greedy (n=200) | 1,2,3 | +0.012 (tr 14.0% / 15.5% / 15.0% vs 12.0% / 11.0% / 13.5%) | +0.042 (tr 19.5% / 45.0% / 40.0% vs 12.0% / 11.0% / 13.5%) | +0.28 | +0.25 / +1.00 / +0.14 | unbounded: the denominator's bootstrap interval includes 0 (6.4% of resamples ≤ 0); percentile values [-2.46, +3.00] are not interpretable |
