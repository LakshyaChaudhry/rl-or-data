# Exploratory GRPO − RFT contrasts on the Easy and Mixed prompt sets (NOT pre-registered)

Defined 2026-09-27, after the test results were seen. Analysis config hash `3ba220e48a66ecea39529d81afa447c80b7382ee25caf2856fd1b3f0ed71c1ec`; Δ = A − B paired seed-wise; CI = paired per-problem bootstrap; criterion = SPEC §10, evaluated mechanically. Bounds: lower = B's truncated completions all right and A's all wrong; upper = the reverse.

| contrast | split | Δ per seed | mean Δ [95 % CI] | 2 × pooled seed sd | bounds [lower, upper] |
|---|---|---|---|---|---|
| grpo_easy − rft_easy | test_300 | +0.113 / +0.053 / +0.087 | +0.084 [+0.048, +0.121] | 0.058 | [+0.060, +0.114] |
| grpo_easy − rft_easy | ood_hard_200 | +0.095 / +0.045 / +0.110 | +0.083 [+0.037, +0.130] |  | [-0.022, +0.345] |
| X_easy_grpo_vs_rft | criterion | | met = True | | |
| grpo_mixed − rft_mixed | test_300 | +0.080 / +0.107 / +0.090 | +0.092 [+0.048, +0.137] | 0.016 | [+0.072, +0.179] |
| grpo_mixed − rft_mixed | ood_hard_200 | +0.015 / +0.015 / +0.075 | +0.035 [-0.020, +0.090] |  | [-0.087, +0.510] |
| X_mixed_grpo_vs_rft | criterion | | met = True | | |
