## 0. Provenance
| item | value |
|---|---|
| date · git commit · SPEC | 2026-09-20; tables and figures written by `make analysis` at git a9ec8af6b01d78b17682fbbcaaa5245b34da9941 (dirty: False), the packet generator is committed together with this file; SPEC v1.8; analysis config hash d9af2a8c5fb5; run root `runs/store_mirror/runs` (read-only); bootstrap seed 0, 10,000 resamples |
| configs/locked/*.yaml sha256[:12] · cap | cap.yaml 16c2ea4d2078; prompt.yaml dc881df9088c; training.yaml 7c3eb151607f; transfer.yaml 2135399fb63e; cap: max_completion_tokens = 4352 (binding term: measured; anchor 2048) |
| eval generation path · sampling params | sampler vllm; vLLM 0.28.0; dtype bfloat16; batch_invariant True; prefix caching False; thinking False; greedy: T=0.0 top_p=1.0 n=1 rep=1.0; mean_at_k: T=1.0 top_p=1.0 n=8 rep=1.0; pass_at_k: T=1.0 top_p=1.0 n=64 rep=1.0; seed scheme: per_prompt: seed + prompt_index * n; vLLM child seeds add sample index |
| runs whose configs/locked differ | 209 units, 20 training runs, 28 git SHAs compared. Differences: git ffb604e (29 run/unit records, e.g. eval/Qwen__Qwen3-4B-Base/val_mixed_100/greedy): training.yaml, transfer.yaml (absent) |
## 1. Sanity (analysis/sanity.py cross-run checks: 13 checks, 0 failures; notes in `sanity.md`)
| # | check | result | failures | notes | what was checked |
|---|---|---|---|---|---|
| 1 | units present | PASS | — | 1 | 209 units loaded for 24 models/runs; 1 expected unit(s) absent. |
| 2 | never-results stay out | PASS | — | 6 | Runs are loaded from an explicit allow-list; every loaded path is re-checked against the never-results fragments. Present under the run root but never read as results: |
| 3 | unit integrity | PASS | — | 0 | Every unit: status finished, n_samples == planned, sampler vllm, thinking off, samples carry the unit's config hash and seed, no truncated completion is scored correct (SPEC §5), and the metrics recomputed from samples.jsonl with core.evaluate equal the stored metrics.json (n_problems, accuracy, ci_low, ci_high, truncation_rate, extraction_failure_rate, answer_line_rate, mean_completion_tokens, pass_at_k, per_tier). |
| 4 | one protocol | PASS | — | 0 | max_completion_tokens, prompt_template, answer_regex, extraction_rule, max_prompt_tokens identical in the resolved config of all 209 units; cap equal to configs/locked/cap.yaml; no provisional cap. |
| 5 | committed splits | PASS | — | 1 | Pairwise disjointness of data/splits by problem_id and pipeline structure; every counting unit's problem_ids_sha256 and sample problem ids equal the committed split (pass@k: its first-100 prefix); gsm8k_500 digests identical across units. |
| 6 | prompt bytes | PASS | — | 1 | For every base-model unit (base, arms, controls) the sha256 over (problem_id, prompt) is identical across runs for the same split/subset — no prompt-template drift between arms. |
| 7 | final checkpoints only | PASS | — | 1 | Each trained run: training finished; every eval/final unit names the run's final adapter (RFT last epoch = adapter/final, GRPO adapter/step_300 with trainer_state.global_step == max_steps) and eval_set 'final'; RFT runs use the val-chosen lr/epochs and append_eos. |
| 8 | seeds actually differ | PASS | — | 20 | sha256 of every trained run's final adapter weights is pairwise distinct across all runs (and equals the store's checksum listing when one is configured); greedy test_300 completions differ between seeds of an arm; greedy val completions differ from the base model's on ≥ 10 % of prompts (adapter really applied). |
| 9 | budgets present | PASS | — | 0 | budgets.json with prompts, completions_available, completions_consumed, training_tokens, optimizer_steps for every trained run; GRPO runs consumed exactly 19,200 completions (reward_records.jsonl line count) and are marked result_bearing; the C1 control's training reward averages ≈ 0.5. |
| 10 | shared hyperparameters | PASS | — | 1 | grpo_config.json identical across GRPO runs except seed, output_dir, run_name, logging_dir; LoRA shape (r, lora_alpha, lora_dropout, bias, use_rslora, use_dora, task_type, target_modules) identical in every adapter_config.json; trainer and vLLM tokenizer hashes identical everywhere. |
| 11 | model selection reads val only | PASS | — | 3 | Each RFT arm's chosen.json was selected on val_mixed_100 and equals the argmax of sweep.json's val accuracy under the recorded tie-break (fewer epochs, then lower learning rate). GRPO ran one fixed recipe: nothing was selected. The analysis itself selects nothing. |
| 12 | SPEC §7 truncation and extraction-failure flags (listed, not fatal) | PASS | — | 26 | A run with truncation > 5 % on test_300 is flagged and its numbers are not headline numbers; on ood_hard_200 truncation is reported, not flagged. One line per model/run that has any flag; flags are carried into every table and contrast that touches them. Under SPEC §5 v1.6 a truncated completion is always an extraction failure, so the two rates are not independent evidence. |
| 13 | provenance (listed, not fatal) | PASS | — | 2 | Units or training runs recorded with a dirty git tree, and the GPU rate meta.json used. |
## 2. Run manifest (result-bearing runs; one row per arm × seed)
| run | arm | seed | status | optimizer steps done / planned | wall-clock h | GPU | adapter sha256[:8] | config hash | final train reward (GRPO) / selected sweep config (RFT) | crashed / restarted / resumed |
|---|---|---|---|---|---|---|---|---|---|---|
| rft/rft_easy/seed1_lr5e-05_ep8 | RFT-Easy | 1 | finished | 6920 / 6920 | 7.11 | NVIDIA H100 PCIe | 73a96947 | d0fda3101856 | selected lr=5e-05, epochs=8 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_easy/seed2_lr5e-05_ep8 | RFT-Easy | 2 | finished | 6920 / 6920 | 7.12 | NVIDIA H100 PCIe | 52384f03 | 08f3b3927d88 | selected lr=5e-05, epochs=8 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_easy/seed3_lr5e-05_ep8 | RFT-Easy | 3 | finished | 6920 / 6920 | 7.05 | NVIDIA H100 PCIe | a095d575 | 443de42ef2c7 | selected lr=5e-05, epochs=8 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_mixed/seed1_lr5e-05_ep4 | RFT-Mixed | 1 | finished | 1992 / 1992 | 2.34 | NVIDIA H100 PCIe | b763bdcd | 9d36b737d93f | selected lr=5e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_mixed/seed2_lr5e-05_ep4 | RFT-Mixed | 2 | finished | 1992 / 1992 | 2.33 | NVIDIA H100 PCIe | f5808968 | 79b8a9478cf1 | selected lr=5e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_mixed/seed3_lr5e-05_ep4 | RFT-Mixed | 3 | finished | 1992 / 1992 | 2.31 | NVIDIA H100 PCIe | fd1999d0 | fb1e463a1027 | selected lr=5e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_curated/seed1_lr1e-05_ep4 | RFT-Curated | 1 | finished | 1664 / 1664 | 1.95 | NVIDIA H100 PCIe | 90e297b9 | 441fa2623cec | selected lr=1e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_curated/seed2_lr1e-05_ep4 | RFT-Curated | 2 | finished | 1664 / 1664 | 1.96 | NVIDIA H100 PCIe | 2ecd0602 | 42312e16991e | selected lr=1e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| rft/rft_curated/seed3_lr1e-05_ep4 | RFT-Curated | 3 | finished | 1664 / 1664 | 1.95 | NVIDIA H100 PCIe | 41e07405 | 3e46440651fb | selected lr=1e-05, epochs=4 (best of 9 on val_mixed_100, seed 1) | none |
| grpo/grpo_easy_s1 | GRPO-Easy | 1 | finished | 300 / 300 | 5.80 | NVIDIA H100 PCIe | 7b5a26bb | 9601e0f7f5ea | final train reward (verify_binary) 0.991 (mean of last 10 steps); last step 0.984 | none |
| grpo/grpo_easy_s2 | GRPO-Easy | 2 | finished | 300 / 300 | 4.66 | NVIDIA H100 PCIe | deccf82b | e654797ea607 | final train reward (verify_binary) 0.989 (mean of last 10 steps); last step 1.000 | none |
| grpo/grpo_easy_s3 | GRPO-Easy | 3 | finished | 300 / 300 | 5.05 | NVIDIA H100 PCIe | d3f4ba37 | 87e14b5f33ed | final train reward (verify_binary) 0.991 (mean of last 10 steps); last step 0.984 | none |
| grpo/grpo_mixed_s1 | GRPO-Mixed | 1 | finished | 300 / 300 | 8.17 | NVIDIA H100 PCIe | 69a7793f | 0b32e0bbbc9a | final train reward (verify_binary) 0.820 (mean of last 10 steps); last step 0.688 | RE-RUN after failed attempt (grpo_mixed_s1_bf16merge_20260914T155049Z; not read) |
| grpo/grpo_mixed_s2 | GRPO-Mixed | 2 | finished | 300 / 300 | 9.54 | NVIDIA H100 PCIe | 99e0f444 | d9b08f9faeaa | final train reward (verify_binary) 0.781 (mean of last 10 steps); last step 0.906 | none |
| grpo/grpo_mixed_s3 | GRPO-Mixed | 3 | finished | 300 / 300 | 9.34 | NVIDIA H100 PCIe | 0bace353 | 73097c93e1bb | final train reward (verify_binary) 0.806 (mean of last 10 steps); last step 0.797 | none |
| grpo/grpo_curated_s1 | GRPO-Curated | 1 | finished | 300 / 300 | 7.23 | NVIDIA H100 PCIe | 2cbd6d91 | ac6693350cf0 | final train reward (verify_binary) 0.891 (mean of last 10 steps); last step 0.891 | RESTARTED from step 0 (earlier attempt's logs kept as 2 *.stale-* files) |
| grpo/grpo_curated_s2 | GRPO-Curated | 2 | finished | 300 / 300 | 9.32 | NVIDIA H100 PCIe | d2ac7789 | cdf988d0d18c | final train reward (verify_binary) 0.886 (mean of last 10 steps); last step 0.906 | none |
| grpo/grpo_curated_s3 | GRPO-Curated | 3 | finished | 300 / 300 | 8.50 | NVIDIA H100 PCIe | 47b2bc75 | 7820651e8cb5 | final train reward (verify_binary) 0.877 (mean of last 10 steps); last step 0.672 | none |
| grpo/grpo_random_reward_s1 | C1 random reward | 1 | finished | 300 / 300 | 5.85 | NVIDIA H100 PCIe | 10fd07fe | 630d7883bff7 | final train reward (random_bernoulli_0_5) 0.522 (mean of last 10 steps); last step 0.656 | none |
| grpo/grpo_format_only_s1 | C2 format only | 1 | finished | 300 / 300 | 5.01 | NVIDIA H100 PCIe | 3bd36cbd | a991d6036f15 | final train reward (format_only) 1.000 (mean of last 10 steps); last step 1.000 | none |
## 3. Base + reference models (one evaluation each; cell = value [bootstrap 95 % CI] tr = truncation %, xf = extraction-fail %; ⚑ = truncation > 5 % on val/test)
| model | seed | val greedy (n=100) | test greedy (n=300) | ood greedy (n=200) | test mean@8 (n=300) | ood mean@8 (n=200) | pass@8 (test first-100, 64 samples) | pass@64 (test first-100, 64 samples) |
|---|---|---|---|---|---|---|---|---|
| Base | 1 | 0.600 [0.510,0.700] tr 1.0 xf 1.0 | 0.673 [0.617,0.727] tr 4.3 xf 4.3 | 0.250 [0.190,0.315] tr 13.0 xf 13.0 | 0.425 [0.389,0.460] tr 0.3 xf 0.6 | 0.173 [0.141,0.207] tr 3.1 xf 3.4 | 0.803 [0.740,0.863] tr 0.5 xf 1.0 | 0.980 [0.950,1.000] tr 0.5 xf 1.0 |
| Qwen3-4B (non-thinking) | 1 | 0.830 [0.750,0.900] tr 3.0 xf 3.0 | 0.747 [0.697,0.797] tr 4.0 xf 4.0 | 0.455 [0.385,0.520] tr 22.5 xf 22.5 | 0.777 [0.740,0.813] tr 0.1 xf 0.1 | 0.459 [0.406,0.512] tr 4.2 xf 4.2 | 0.927 [0.878,0.969] tr 0.1 xf 0.1 | 0.970 [0.930,1.000] tr 0.1 xf 0.1 |
| Qwen2.5-7B-Instruct | 1 | 0.650 [0.560,0.740] tr 3.0 xf 3.0 | 0.653 [0.600,0.707] tr 2.3 xf 2.3 | 0.280 [0.220,0.345] tr 19.0 xf 19.0 | 0.593 [0.551,0.635] tr 0.2 xf 0.2 | 0.276 [0.232,0.323] tr 2.9 xf 2.9 | 0.839 [0.773,0.900] tr 0.2 xf 0.2 | 0.930 [0.880,0.980] tr 0.2 xf 0.2 |
| Llama-3.1-8B-Instruct | 1 | 0.500 [0.400,0.600] tr 8.0 xf 8.0 ⚑ | 0.373 [0.320,0.427] tr 11.3 xf 11.3 ⚑ | 0.090 [0.050,0.130] tr 47.5 xf 47.5 | 0.335 [0.298,0.372] tr 15.2 xf 15.2 ⚑ | 0.113 [0.089,0.138] tr 38.8 xf 38.8 | 0.700 [0.628,0.771] tr 14.8 xf 14.8 ⚑ | 0.880 [0.810,0.940] tr 14.8 xf 14.8 ⚑ |
## 4. Main table (greedy accuracy [bootstrap 95 % CI over problems]; μ ± σ = mean ± sample std across seeds [CI of the seed-averaged score]; ⚑ = truncation > 5 % on val/test, SPEC §7)
| arm | seed | val greedy (n=100) | test greedy (n=300) | ood greedy (n=200) | test mean@8 (n=300) | trunc % (val / test / ood greedy / test mean@8) | extraction-fail % (same order) | config hash |
|---|---|---|---|---|---|---|---|---|
| RFT-Easy | 1 | 0.670 [0.580,0.760] | 0.663 [0.610,0.717] | 0.250 [0.195,0.310] | 0.538 [0.497,0.578] | 0.0 / 2.3 / 14.0 / 0.7 | 0.0 / 2.3 / 14.0 / 0.7 | d0fda3101856 |
| RFT-Easy | 2 | 0.630 [0.530,0.720] | 0.647 [0.593,0.700] | 0.260 [0.200,0.320] | 0.543 [0.502,0.583] | 0.0 / 3.3 / 9.5 / 0.2 | 0.0 / 3.3 / 9.5 / 0.2 | 08f3b3927d88 |
| RFT-Easy | 3 | 0.670 [0.580,0.760] | 0.667 [0.613,0.717] | 0.255 [0.195,0.315] | 0.525 [0.486,0.564] | 1.0 / 1.7 / 8.0 / 0.3 | 1.0 / 1.7 / 8.0 / 0.3 | 443de42ef2c7 |
| **RFT-Easy** | **μ ± σ** | 0.657 ± 0.023 [0.570,0.740] | 0.659 ± 0.011 [0.611,0.708] | 0.255 ± 0.005 [0.203,0.310] | 0.535 ± 0.009 [0.496,0.575] | 0.3 / 2.4 / 10.5 / 0.4 | 0.3 / 2.4 / 10.5 / 0.4 | test trunc > 5 %: seeds none |
| RFT-Mixed | 1 | 0.670 [0.580,0.760] | 0.707 [0.653,0.757] | 0.280 [0.220,0.345] | 0.550 [0.510,0.589] | 1.0 / 2.7 / 12.0 / 0.4 | 1.0 / 2.7 / 12.0 / 0.4 | 9d36b737d93f |
| RFT-Mixed | 2 | 0.660 [0.560,0.750] | 0.687 [0.633,0.740] | 0.280 [0.220,0.345] | 0.540 [0.502,0.579] | 1.0 / 1.7 / 11.0 / 0.5 | 1.0 / 1.7 / 11.0 / 0.5 | 79b8a9478cf1 |
| RFT-Mixed | 3 | 0.630 [0.540,0.720] | 0.693 [0.640,0.743] | 0.260 [0.200,0.320] | 0.541 [0.502,0.580] | 2.0 / 1.7 / 13.5 / 0.4 | 2.0 / 1.7 / 13.5 / 0.5 | fb1e463a1027 |
| **RFT-Mixed** | **μ ± σ** | 0.653 ± 0.021 [0.567,0.737] | 0.696 ± 0.010 [0.648,0.742] | 0.273 ± 0.012 [0.220,0.328] | 0.544 ± 0.006 [0.505,0.582] | 1.3 / 2.0 / 12.2 / 0.4 | 1.3 / 2.0 / 12.2 / 0.5 | test trunc > 5 %: seeds none |
| RFT-Curated | 1 | 0.660 [0.570,0.750] | 0.673 [0.620,0.723] | 0.300 [0.240,0.365] | 0.514 [0.474,0.552] | 0.0 / 2.7 / 14.0 / 0.5 | 0.0 / 2.7 / 14.0 / 0.6 | 441fa2623cec |
| RFT-Curated | 2 | 0.660 [0.560,0.750] | 0.657 [0.603,0.710] | 0.290 [0.225,0.355] | 0.510 [0.472,0.549] | 1.0 / 3.3 / 15.5 / 0.7 | 1.0 / 3.3 / 15.5 / 0.7 | 42312e16991e |
| RFT-Curated | 3 | 0.630 [0.540,0.720] | 0.677 [0.623,0.730] | 0.265 [0.205,0.325] | 0.511 [0.472,0.550] | 0.0 / 3.3 / 15.0 / 0.5 | 0.0 / 3.3 / 15.0 / 0.5 | 3e46440651fb |
| **RFT-Curated** | **μ ± σ** | 0.650 ± 0.017 [0.563,0.733] | 0.669 ± 0.011 [0.619,0.719] | 0.285 ± 0.018 [0.230,0.340] | 0.512 ± 0.002 [0.474,0.550] | 0.3 / 3.1 / 14.8 / 0.5 | 0.3 / 3.1 / 14.8 / 0.6 | test trunc > 5 %: seeds none |
| GRPO-Easy | 1 | 0.730 [0.640,0.820] | 0.777 [0.727,0.820] | 0.345 [0.280,0.410] | 0.724 [0.685,0.762] | 5.0 / 4.3 / 38.0 / 2.5 | 5.0 / 4.3 / 38.0 / 2.5 | 9601e0f7f5ea |
| GRPO-Easy | 2 | 0.650 [0.550,0.740] | 0.700 [0.650,0.750] | 0.305 [0.240,0.370] | 0.672 [0.628,0.716] | 3.0 / 3.3 / 21.0 / 1.2 | 3.0 / 3.3 / 21.0 / 1.2 | e654797ea607 |
| GRPO-Easy | 3 | 0.730 [0.640,0.810] | 0.753 [0.703,0.800] | 0.365 [0.300,0.430] | 0.708 [0.667,0.748] | 1.0 / 1.3 / 19.5 / 2.0 | 1.0 / 1.3 / 19.5 / 2.0 | 87e14b5f33ed |
| **GRPO-Easy** | **μ ± σ** | 0.703 ± 0.046 [0.627,0.777] | 0.743 ± 0.039 [0.701,0.786] | 0.338 ± 0.031 [0.287,0.390] | 0.701 ± 0.026 [0.662,0.740] | 3.0 / 3.0 / 26.2 / 1.9 | 3.0 / 3.0 / 26.2 / 1.9 | test trunc > 5 %: seeds none |
| GRPO-Mixed | 1 | 0.770 [0.680,0.850] | 0.787 [0.740,0.833] ⚑ | 0.295 [0.235,0.360] | 0.778 [0.740,0.815] ⚑ | 3.0 / 6.7 / 46.5 / 5.3 | 3.0 / 6.7 / 46.5 / 5.3 | 0b32e0bbbc9a |
| GRPO-Mixed | 2 | 0.750 [0.660,0.830] ⚑ | 0.793 [0.747,0.837] ⚑ | 0.295 [0.235,0.360] | 0.765 [0.728,0.801] ⚑ | 9.0 / 10.3 / 54.5 / 8.2 | 9.0 / 10.3 / 54.5 / 8.2 | d9b08f9faeaa |
| GRPO-Mixed | 3 | 0.760 [0.670,0.840] ⚑ | 0.783 [0.737,0.830] ⚑ | 0.335 [0.270,0.405] | 0.768 [0.731,0.804] ⚑ | 8.0 / 9.0 / 41.5 / 6.8 | 8.0 / 9.0 / 41.5 / 6.8 | 73097c93e1bb |
| **GRPO-Mixed** | **μ ± σ** | 0.760 ± 0.010 [0.687,0.827] ⚑×2 | 0.788 ± 0.005 [0.749,0.826] ⚑×3 | 0.308 ± 0.023 [0.255,0.362] | 0.771 ± 0.007 [0.736,0.805] ⚑×3 | 6.7 / 8.7 / 47.5 / 6.8 | 6.7 / 8.7 / 47.5 / 6.8 | test trunc > 5 %: seeds [1, 2, 3] |
| GRPO-Curated | 1 | 0.770 [0.690,0.850] | 0.793 [0.747,0.840] ⚑ | 0.360 [0.295,0.430] | 0.772 [0.734,0.809] | 2.0 / 5.3 / 19.5 / 1.8 | 2.0 / 5.3 / 19.5 / 1.8 | ac6693350cf0 |
| GRPO-Curated | 2 | 0.770 [0.680,0.850] | 0.760 [0.710,0.810] ⚑ | 0.290 [0.230,0.355] | 0.761 [0.724,0.797] ⚑ | 4.0 / 9.3 / 45.0 / 7.1 | 4.0 / 9.3 / 45.0 / 7.1 | cdf988d0d18c |
| GRPO-Curated | 3 | 0.770 [0.680,0.850] | 0.780 [0.733,0.827] ⚑ | 0.295 [0.235,0.360] | 0.756 [0.718,0.794] ⚑ | 5.0 / 8.0 / 40.0 / 6.0 | 5.0 / 8.0 / 40.0 / 6.0 | 7820651e8cb5 |
| **GRPO-Curated** | **μ ± σ** | 0.770 ± 0.000 [0.700,0.833] | 0.778 ± 0.017 [0.736,0.818] ⚑×3 | 0.315 ± 0.039 [0.262,0.368] | 0.763 ± 0.008 [0.727,0.798] ⚑×2 | 3.7 / 7.6 / 34.8 / 4.9 | 3.7 / 7.6 / 34.8 / 4.9 | test trunc > 5 %: seeds [1, 2, 3] |
| C1 random reward | 1 | 0.440 [0.340,0.540] | 0.477 [0.420,0.533] | 0.220 [0.165,0.280] | 0.336 [0.302,0.368] | 3.0 / 4.7 / 7.5 / 0.2 | 3.0 / 4.7 / 7.5 / 0.3 | 630d7883bff7 |
| C2 format only | 1 | 0.620 [0.530,0.710] | 0.637 [0.580,0.690] | 0.270 [0.210,0.335] | 0.603 [0.560,0.645] | 1.0 / 1.0 / 10.5 / 0.2 | 1.0 / 1.0 / 10.5 / 0.2 | a991d6036f15 |
## 5. Paired contrasts, seed-wise (Δ = A − B, greedy accuracy; seeds 1 / 2 / 3; [paired problem-bootstrap 95 % CI]; SPEC §10: (a) |Δ test| > 2 × pooled seed std and (b) same sign on ood; H2 numerator and denominator contrasts: section 11)
| contrast | test Δ per seed (n=300) | test mean Δ ± seed std of Δ [CI] | ood Δ per seed (n=200) | ood mean Δ ± seed std of Δ [CI] | 2 × pooled seed std (test) | (a) | (b) | SPEC §10 criterion |
|---|---|---|---|---|---|---|---|---|
| H1a RFT-Mixed − RFT-Easy | +0.043 / +0.040 / +0.027 | +0.037 ± 0.009 [+0.002,+0.070] | +0.030 / +0.020 / +0.005 | +0.018 ± 0.013 [-0.020,+0.057] | 0.021 | yes | yes | PASS |
| H1b GRPO-Mixed − GRPO-Easy | +0.010 / +0.093 / +0.030 | +0.044 ± 0.044 [+0.011,+0.079] | -0.050 / -0.010 / -0.030 | -0.030 ± 0.020 [-0.073,+0.012] | 0.056 | no | no | FAIL |
| H2 recovery fraction (RFT-Curated − RFT-Mixed) / (GRPO-Curated − RFT-Mixed) | -0.385 / -0.409 / -0.192 | -0.027 / +0.082 = -0.324 (on the seed means) [-1.000,+0.000] | +0.250 / +1.000 / +0.143 | +0.012 / +0.042 = +0.280 (on the seed means) unbounded (6.4 % of resamples have denominator ≤ 0) | none registered | — | — | n/a — no threshold registered (notebook/PREREGISTRATION.md holds no number for H2: §1 'much of the gap', §3 'most', §6 (2026-09-20) 'No threshold is set on any "fraction of the GRPO advantage recovered"') |
| H3 GRPO-Curated − RFT-Curated | +0.120 / +0.103 / +0.103 | +0.109 ± 0.010 [+0.067,+0.151] | +0.060 / +0.000 / +0.030 | +0.030 ± 0.030 [-0.025,+0.085] | 0.028 | yes | yes | PASS |
| C1 − Base | -0.197 | -0.197 ± n/a [-0.253,-0.143] | -0.030 | -0.030 ± n/a [-0.090,+0.030] | n/a (1 seed) | n/a | yes | n/a (1 seed) |
| C2 − Base | -0.037 | -0.037 ± n/a [-0.083,+0.010] | +0.020 | +0.020 ± n/a [-0.045,+0.085] | n/a (1 seed) | n/a | no | n/a (1 seed) |
## 6. Per-tier (base pass@8 tier) and per-step-count (`total_steps` of the problem) greedy accuracy on test_300 (μ ± σ across seeds [CI of the seed-averaged score] tr = truncation %)
| arm | seeds | tier easy (n=100) | tier medium (n=100) | tier hard (n=100) | 2 steps (n=69) | 3 steps (n=78) | 4 steps (n=75) | 5 steps (n=78) |
|---|---|---|---|---|---|---|---|---|
| Base | 1 | 0.920 [0.860,0.970] tr 3.0 | 0.810 [0.730,0.880] tr 5.0 | 0.290 [0.210,0.380] tr 5.0 | 0.681 [0.565,0.783] tr 1.4 | 0.731 [0.628,0.821] tr 5.1 | 0.747 [0.653,0.840] tr 5.3 | 0.538 [0.423,0.654] tr 5.1 |
| RFT-Mixed | 1,2,3 | 0.960 ± 0.010 [0.927,0.987] tr 0.7 | 0.830 ± 0.026 [0.767,0.887] tr 1.3 | 0.297 ± 0.021 [0.223,0.373] tr 4.0 | 0.763 ± 0.022 [0.671,0.850] tr 1.4 | 0.739 ± 0.007 [0.654,0.821] tr 1.7 | 0.742 ± 0.015 [0.644,0.827] tr 2.2 | 0.547 ± 0.015 [0.449,0.645] tr 2.6 |
| RFT-Curated | 1,2,3 | 0.957 ± 0.023 [0.917,0.987] tr 1.0 | 0.790 ± 0.046 [0.720,0.857] tr 2.3 | 0.260 ± 0.040 [0.187,0.340] tr 6.0 | 0.729 ± 0.022 [0.628,0.821] tr 1.9 | 0.714 ± 0.027 [0.615,0.803] tr 1.7 | 0.716 ± 0.008 [0.618,0.809] tr 5.3 | 0.526 ± 0.000 [0.423,0.632] tr 3.4 |
| GRPO-Mixed | 1,2,3 | 0.923 ± 0.021 [0.877,0.963] tr 6.0 | 0.860 ± 0.017 [0.803,0.910] tr 6.7 | 0.580 ± 0.026 [0.497,0.660] tr 13.3 | 0.836 ± 0.030 [0.763,0.903] tr 4.8 | 0.821 ± 0.013 [0.744,0.893] tr 9.0 | 0.840 ± 0.035 [0.769,0.902] tr 7.1 | 0.662 ± 0.007 [0.573,0.752] tr 13.2 |
| GRPO-Curated | 1,2,3 | 0.930 ± 0.036 [0.893,0.963] tr 5.3 | 0.877 ± 0.021 [0.823,0.923] tr 4.3 | 0.527 ± 0.012 [0.440,0.613] tr 13.0 | 0.821 ± 0.051 [0.744,0.894] tr 5.3 | 0.812 ± 0.007 [0.735,0.880] tr 4.3 | 0.836 ± 0.056 [0.760,0.902] tr 4.9 | 0.650 ± 0.007 [0.556,0.744] tr 15.4 |
## 7. Budget accounting, actual (per seed unless stated; GPU = 1× H100 PCIe; $ at the $3.29/h billed rate; total 201.12 GPU-h = $662 incl. the 0.58 GPU-h base draw, excl. GRPO checkpoint val evals)
| arm | seeds | prompts used of prompt set (RFT: prompts with ≥ 1 verified-correct sample; GRPO: all prompts are sampled) | completions generated | completions consumed in gradient updates (per seed) | gradient tokens (per seed) | optimizer steps (per seed) | train GPU-h (per seed) | GPU-h all seeds (train + final eval) | GPU-h other sweep configs (train + val eval) | estimated $ (arm) |
|---|---|---|---|---|---|---|---|---|---|---|
| RFT-Easy | 1,2,3 | 100 of 100 | 19,200 (base draw, sampled once, shared by all seeds) | 13,834 distinct × 8 epochs = 110,672 | 47,237,096 / 47,237,096 / 47,237,096 | 6,920 / 6,920 / 6,920 | 7.11 / 7.12 / 7.05 | 22.68 | 30.26 | $174 |
| RFT-Mixed | 1,2,3 | 95 of 100 | 19,200 (base draw, sampled once, shared by all seeds) | 7,961 distinct × 4 epochs = 31,844 | 15,852,320 / 15,852,320 / 15,852,320 | 1,992 / 1,992 / 1,992 | 2.34 / 2.33 / 2.31 | 8.49 | 22.32 | $101 |
| RFT-Curated | 1,2,3 | 73 of 73 | 14,016 (base draw, sampled once, shared by all seeds) | 6,653 distinct × 4 epochs = 26,612 | 13,377,076 / 13,377,076 / 13,377,076 | 1,664 / 1,664 / 1,664 | 1.95 / 1.96 / 1.95 | 7.39 | 18.68 | $86 |
| GRPO-Easy | 1,2,3 | 100 of 100 | 19,200 (on-policy, per seed) | 19,200 × 1 = 19,200 | 13,942,821 / 11,218,987 / 12,041,484 | 300 / 300 / 300 | 5.80 / 4.66 / 5.05 | 18.17 | — | $60 |
| GRPO-Mixed | 1,2,3 | 100 of 100 | 19,200 (on-policy, per seed) | 19,200 × 1 = 19,200 | 19,282,312 / 24,681,340 / 25,019,069 | 300 / 300 / 300 | 8.17 / 9.54 / 9.34 | 31.71 | — | $104 |
| GRPO-Curated | 1,2,3 | 73 of 73 | 19,200 (on-policy, per seed) | 19,200 × 1 = 19,200 | 17,444,246 / 24,076,892 / 20,377,871 | 300 / 300 / 300 | 7.23 / 9.32 / 8.50 | 29.11 | — | $96 |
| C1 random reward | 1 | 100 of 100 | 19,200 (on-policy, per seed) | 19,200 × 1 = 19,200 | 9,268,987 | 300 | 5.85 | 6.27 | — | $21 |
| C2 format only | 1 | 100 of 100 | 19,200 (on-policy, per seed) | 19,200 × 1 = 19,200 | 10,844,177 | 300 | 5.01 | 5.45 | — | $18 |
Matched budgets, RFT vs GRPO per pair (SPEC §8): (1) prompt budget, prompts in the pair's prompt set: equal in every pair (easy 100 vs 100; mixed 100 vs 100; curated 73 vs 73). prompts that contribute training examples: NOT equal for mixed (easy 100 vs 100; mixed 95 vs 100; curated 73 vs 73). (2) generation budget, completions generated (SPEC §8.2 gives train_curated the draw samples of its own prompts): NOT equal for curated (easy 19,200 vs 19,200; mixed 19,200 vs 19,200; curated 14,016 vs 19,200). (3) gradient/token budget, which SPEC §8.3 states cannot be equalized: gradient tokens: NOT equal for easy, mixed, curated (easy 47,237,096 vs 12,401,097; mixed 15,852,320 vs 22,994,240; curated 13,377,076 vs 20,633,003). optimizer steps: NOT equal for easy, mixed, curated (easy 6,920 vs 300; mixed 1,992 vs 300; curated 1,664 vs 300). Largest discrepancy: optimizer steps, rft_easy 6,920 vs grpo_easy 300 (23.1×).
## 8. RFT sweep: 9 configs on seed 1, selected on val_mixed_100 only (◀ = selected; cell = val accuracy [95 % CI] tr = truncation %)
| arm (seed 1, val_mixed_100 greedy, n=100) | lr=1e-05 ep=2 | lr=1e-05 ep=4 | lr=1e-05 ep=8 | lr=5e-05 ep=2 | lr=5e-05 ep=4 | lr=5e-05 ep=8 | lr=0.0001 ep=2 | lr=0.0001 ep=4 | lr=0.0001 ep=8 | selected | selected config on grid edge? |
|---|---|---|---|---|---|---|---|---|---|---|---|
| RFT-Easy | 0.590 [0.500,0.680] tr 3.0 | 0.640 [0.550,0.730] tr 3.0 | 0.610 [0.520,0.700] tr 1.0 | 0.630 [0.540,0.720] tr 2.0 | 0.620 [0.520,0.710] tr 3.0 | 0.670 [0.580,0.760] tr 0.0 ◀ | 0.610 [0.510,0.710] tr 4.0 | 0.640 [0.550,0.730] tr 0.0 | 0.630 [0.530,0.720] tr 0.0 | lr=5e-05 ep=8 (val 0.670; tie-break: fewer epochs, then lower learning rate) | yes (highest epochs) |
| RFT-Mixed | 0.620 [0.530,0.720] tr 1.0 | 0.620 [0.530,0.710] tr 1.0 | 0.660 [0.560,0.750] tr 2.0 | 0.650 [0.560,0.740] tr 2.0 | 0.670 [0.580,0.760] tr 1.0 ◀ | 0.630 [0.530,0.720] tr 2.0 | 0.620 [0.530,0.720] tr 3.0 | 0.670 [0.580,0.760] tr 2.0 | 0.570 [0.470,0.670] tr 0.0 | lr=5e-05 ep=4 (val 0.670; tie-break: fewer epochs, then lower learning rate) | no (interior) |
| RFT-Curated | 0.620 [0.530,0.710] tr 1.0 | 0.660 [0.570,0.750] tr 0.0 ◀ | 0.650 [0.560,0.740] tr 1.0 | 0.630 [0.530,0.720] tr 1.0 | 0.640 [0.540,0.730] tr 1.0 | 0.600 [0.500,0.690] tr 0.0 | 0.640 [0.550,0.730] tr 0.0 | 0.640 [0.550,0.730] tr 1.0 | 0.580 [0.480,0.680] tr 1.0 | lr=1e-05 ep=4 (val 0.660; tie-break: fewer epochs, then lower learning rate) | yes (lowest lr) |
## 9. GRPO training dynamics (cell = seed 1 (μ across seeds 1–3); train-log rows = mean over the 10 steps ending at the step, 640 completions per seed; column 0 = steps 1–10; tier rows weight by completions; val at step 0 = base model, one evaluation)
| arm | quantity | 0 | 50 | 100 | 150 | 200 | 250 | 300 |
|---|---|---|---|---|---|---|---|---|
| GRPO-Easy | train reward | 0.702 (μ 0.705) | 0.930 (μ 0.922) | 0.942 (μ 0.956) | 0.969 (μ 0.977) | 0.984 (μ 0.987) | 0.978 (μ 0.986) | 0.991 (μ 0.990) |
| GRPO-Easy | frac_reward_zero_std: all; easy | 0.050; 0.050 (μ 0.100; 0.100) | 0.725; 0.725 (μ 0.654; 0.654) | 0.700; 0.700 (μ 0.771; 0.771) | 0.825; 0.825 (μ 0.858; 0.858) | 0.887; 0.887 (μ 0.908; 0.908) | 0.887; 0.887 (μ 0.912; 0.912) | 0.938; 0.938 (μ 0.933; 0.933) |
| GRPO-Easy | mean completion length (tokens) · truncation % of training completions | 465 · 0.2 (μ 434 · 0.1) | 563 · 0.2 (μ 551 · 0.2) | 658 · 0.3 (μ 608 · 0.2) | 692 · 0.0 (μ 681 · 0.2) | 926 · 0.9 (μ 727 · 0.3) | 901 · 0.0 (μ 723 · 0.0) | 844 · 0.0 (μ 719 · 0.0) |
| GRPO-Easy | val_mixed_100 greedy (n=100) | 0.600 [0.510,0.700] tr 1.0 (μ 0.600) | — (no checkpoint eval) | 0.710 [0.620,0.800] tr 2.0 (μ 0.677) | — (no checkpoint eval) | 0.730 [0.640,0.820] tr 4.0 (μ 0.707) | — (no checkpoint eval) | 0.730 [0.640,0.820] tr 5.0 (μ 0.703) |
| GRPO-Mixed | train reward | 0.450 (μ 0.421) | 0.572 (μ 0.591) | 0.702 (μ 0.683) | 0.772 (μ 0.770) | 0.805 (μ 0.798) | 0.867 (μ 0.814) | 0.820 (μ 0.803) |
| GRPO-Mixed | frac_reward_zero_std: all; easy/medium/hard | 0.263; 0.240/0.133/0.440 (μ 0.246; 0.130/0.126/0.504) | 0.438; 0.536/0.320/0.444 (μ 0.421; 0.599/0.192/0.462) | 0.475; 0.880/0.200/0.367 (μ 0.533; 0.829/0.389/0.394) | 0.613; 0.958/0.536/0.393 (μ 0.579; 0.896/0.495/0.370) | 0.637; 0.958/0.654/0.367 (μ 0.633; 0.865/0.644/0.404) | 0.675; 0.966/0.615/0.400 (μ 0.646; 0.915/0.626/0.360) | 0.700; 1.000/0.667/0.483 (μ 0.671; 0.962/0.611/0.463) |
| GRPO-Mixed | mean completion length (tokens) · truncation % of training completions | 520 · 0.0 (μ 528 · 0.3) | 698 · 0.2 (μ 706 · 0.4) | 941 · 1.2 (μ 1016 · 1.1) | 1101 · 1.6 (μ 1369 · 2.5) | 1281 · 0.6 (μ 1543 · 3.2) | 1176 · 0.9 (μ 1431 · 2.6) | 1250 · 2.2 (μ 1437 · 2.6) |
| GRPO-Mixed | val_mixed_100 greedy (n=100) | 0.600 [0.510,0.700] tr 1.0 (μ 0.600) | — (no checkpoint eval) | 0.680 [0.590,0.770] tr 2.0 (μ 0.710) | — (no checkpoint eval) | 0.750 [0.660,0.830] tr 7.0 (μ 0.750) | — (no checkpoint eval) | 0.770 [0.680,0.850] tr 3.0 (μ 0.760) |
| GRPO-Curated | train reward | 0.473 (μ 0.462) | 0.658 (μ 0.677) | 0.783 (μ 0.805) | 0.817 (μ 0.843) | 0.892 (μ 0.881) | 0.878 (μ 0.888) | 0.891 (μ 0.884) |
| GRPO-Curated | frac_reward_zero_std: all; easy/medium/hard | 0.125; 0.097/0.091/0.250 (μ 0.146; 0.070/0.076/0.421) | 0.412; 0.704/0.189/0.438 (μ 0.375; 0.622/0.204/0.315) | 0.512; 0.821/0.378/0.267 (μ 0.542; 0.816/0.395/0.378) | 0.588; 0.852/0.513/0.286 (μ 0.596; 0.833/0.566/0.292) | 0.738; 0.933/0.684/0.417 (μ 0.704; 0.916/0.667/0.390) | 0.637; 0.862/0.529/0.471 (μ 0.671; 0.876/0.609/0.396) | 0.725; 0.912/0.645/0.467 (μ 0.704; 0.909/0.679/0.370) |
| GRPO-Curated | mean completion length (tokens) · truncation % of training completions | 484 · 0.2 (μ 525 · 0.3) | 705 · 0.3 (μ 718 · 0.5) | 1028 · 0.5 (μ 1111 · 1.1) | 894 · 0.5 (μ 1213 · 1.5) | 1052 · 0.5 (μ 1267 · 1.4) | 1123 · 0.2 (μ 1268 · 1.2) | 1022 · 0.5 (μ 1267 · 1.5) |
| GRPO-Curated | val_mixed_100 greedy (n=100) | 0.600 [0.510,0.700] tr 1.0 (μ 0.600) | — (no checkpoint eval) | 0.720 [0.630,0.810] tr 2.0 (μ 0.730) | — (no checkpoint eval) | 0.760 [0.670,0.840] tr 2.0 (μ 0.757) | — (no checkpoint eval) | 0.770 [0.690,0.850] tr 2.0 (μ 0.770) |
## 10. "How could this be wrong" gaps per contrast (A − B; test / ood greedy; full text in `how_could_this_be_wrong.md`)
| contrast | truncation gap (pp, test / ood) | extraction-fail gap (pp, test / ood) | budget gap (A vs B, mean per seed) | tier composition of training prompts easy/medium/hard (A vs B) | seed ranges overlap (test / ood) | seeds with test truncation > 5 % |
|---|---|---|---|---|---|---|
| H1a RFT-Mixed − RFT-Easy | -0.4 / +1.7 | -0.4 / +1.7 | generated 19,200 vs 19,200; gradient tokens 15,852,320 vs 47,237,096; optimizer steps 1,992 vs 6,920; configs tried 9 vs 9 | 33/33/34 (SFT examples 4605/2761/595) vs 100/0/0 (SFT examples 13834/0/0) | no / yes | A: none; B: none |
| H1b GRPO-Mixed − GRPO-Easy | +5.7 / +21.3 | +5.7 / +21.3 | generated 19,200 vs 19,200; gradient tokens 22,994,240 vs 12,401,097; optimizer steps 300 vs 300; configs tried 1 vs 1 | 33/33/34 vs 100/0/0 | no / yes | A: 1,2,3; B: none |
| H2 numerator RFT-Curated − RFT-Mixed ‖ denominator GRPO-Curated − RFT-Mixed | +1.1 / +2.7 ‖ +5.6 / +22.7 | +1.1 / +2.7 ‖ +5.6 / +22.7 | generated 14,016 vs 19,200; gradient tokens 13,377,076 vs 15,852,320; optimizer steps 1,664 vs 1,992; configs tried 9 vs 9 ‖ generated 19,200 vs 19,200; gradient tokens 20,633,003 vs 15,852,320; optimizer steps 300 vs 1,992; configs tried 1 vs 9 | 26/33/14 (SFT examples 3481/2761/411) vs 33/33/34 (SFT examples 4605/2761/595) ‖ 26/33/14 vs 33/33/34 (SFT examples 4605/2761/595) | no / yes ‖ no / no | A: none; B: none ‖ A: 1,2,3; B: none |
| H3 GRPO-Curated − RFT-Curated | +4.4 / +20.0 | +4.4 / +20.0 | generated 19,200 vs 14,016; gradient tokens 20,633,003 vs 13,377,076; optimizer steps 300 vs 1,664; configs tried 1 vs 9 | 26/33/14 vs 26/33/14 (SFT examples 3481/2761/411) | no / yes | A: 1,2,3; B: none |
## 11. `hypotheses.md`, verbatim (verdict lines blank)
# Hypothesis grading sheet (tasks/05 item 5)

Generated by `rlordata.analysis.report`; regenerating overwrites this file, so grade a copy (or commit the graded file elsewhere). **The agent computes and evaluates the pre-registered criterion mechanically; every verdict line is blank for Laksh.**

Inputs: final units of seeds 1–3 under `runs/store_mirror/runs`; analysis config hash `d9af2a8c5fb5`; bootstrap seed 0, 10,000 resamples. Δ = A − B on greedy accuracy, paired seed-wise; CI = paired per-problem bootstrap; pooled seed std uses sample variances (ddof = 1). Full tables: `tables/results.md`, `tables/contrasts.md`; per-contrast caveats: `how_could_this_be_wrong.md`; checks: `sanity.md`.

## Criterion (SPEC §10, PREREGISTRATION §2)

An arm-vs-arm difference counts if (a) |Δ greedy accuracy on test_300| > 2 × the pooled seed std of the two arms, **and** (b) the sign of Δ is the same on ood_hard_200.

## Standing caveats — read before any line below

1. **Truncation differs systematically between methods.** SPEC §7: a run with truncation > 5 % on test_300 is flagged and its numbers are not headline numbers. Flagged on test_300 greedy: GRPO-Mixed seeds 1,2,3; GRPO-Curated seeds 1,2,3. No RFT run is flagged. A truncated completion is scored wrong whatever it contains, so every accuracy is a lower bound on its uncapped value and the bound is looser on the GRPO side of every RFT-vs-GRPO Δ; on ood_hard_200 (GRPO greedy truncation up to 54.5%) a large share of the ood result is 'ran out of tokens'. This lands directly on H3 and on the H2 denominator. **No correction is applied anywhere in this sheet**; whether one is warranted is a protocol decision for Laksh.
2. **Tuning asymmetry (PREREGISTRATION §5).** RFT tried 9 configurations per arm and reports the best on a 100-problem val set ('best on val', not 'optimal'; partly noise); GRPO ran one fixed recipe. The asymmetry tilts toward RFT, so an H3 result in GRPO's favour is the more convincing direction. Test and ood were never used to choose anything (`sanity.md`, 'model selection').
3. **Three seeds per arm.** The seed std in the criterion is estimated from 3 values per arm.
4. Base and the controls C1/C2 have one run each, so criterion (a) cannot be evaluated for them.

## H1 — data effect

> SPEC §1: Mixed-difficulty training data improves held-out performance relative to easy-only data *for both* RFT and GRPO. Test: (RFT-Mixed − RFT-Easy) and (GRPO-Mixed − GRPO-Easy), each vs. seed spread.
>
> Falsifier (PREREGISTRATION §1): an asymmetry — GRPO-Mixed > GRPO-Easy by more than seed spread, but RFT-Mixed ≈ RFT-Easy (or worse).

**RFT-Mixed − RFT-Easy** (`H1_rft`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.043 / +0.040 / +0.027 | +0.037 ± 0.009 [+0.002, +0.070] | 0.010 | 2.7% / 1.7% / 1.7% vs 2.3% / 3.3% / 1.7% | -0.4 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.030 / +0.020 / +0.005 | +0.018 ± 0.013 [-0.020, +0.057] | 0.009 | 12.0% / 11.0% / 13.5% vs 14.0% / 9.5% / 8.0% | +1.7 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.021: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- no run in this contrast exceeds 5 % truncation on test_300 greedy

**GRPO-Mixed − GRPO-Easy** (`H1_grpo`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.010 / +0.093 / +0.030 | +0.044 ± 0.044 [+0.011, +0.079] | 0.028 | 6.7% / 10.3% / 9.0% vs 4.3% / 3.3% / 1.3% | +5.7 pp | no |
| ood greedy (n=200) | 1,2,3 | -0.050 / -0.010 / -0.030 | -0.030 ± 0.020 [-0.073, +0.012] | 0.027 | 46.5% / 54.5% / 41.5% vs 38.0% / 21.0% / 19.5% | +21.3 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.056: **no**; (b) same sign on ood: **no** → criterion **NOT MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Mixed seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

Both contrasts are within one method, but truncation still differs inside GRPO: GRPO-Mixed minus GRPO-Easy is +5.7 pp on test_300 and +21.3 pp on ood_hard_200, the split criterion (b) reads its sign from (RFT: -0.4 pp and +1.7 pp).

**Verdict (Laksh):** ______________________________________________

## H2 — implicit filtering

> SPEC §1: RFT trained on prompts selected by the same current-policy criterion that makes GRPO groups informative (1–7 correct of 8) recovers a substantial fraction of GRPO's advantage. Test: (RFT-Curated − RFT-Mixed) relative to (GRPO-Curated − RFT-Mixed).
>
> Falsifier (PREREGISTRATION §1): RFT-Curated ≈ RFT-Mixed while GRPO-Curated stays well above both; also falsified if RFT-Curated is worse than RFT-Mixed.

**Threshold.** tasks/05 asks for 'the pre-registered threshold' on this ratio. No numeric threshold exists in SPEC.md, PREREGISTRATION.md or any task file (SPEC says 'a substantial fraction', the pre-registration 'much of the gap'). None is invented here: the sheet reports the ratio, its interval and the two falsifier conditions evaluated mechanically.

Threshold Laksh applies to the ratio: ______

| metric | seeds | numerator Δ (truncation A vs B) | denominator Δ (truncation A vs B) | ratio | ratio per seed | joint problem-bootstrap 95 % CI |
|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | -0.027 (tr 2.7% / 3.3% / 3.3% vs 2.7% / 1.7% / 1.7%) | +0.082 (tr 5.3% / 9.3% / 8.0% vs 2.7% / 1.7% / 1.7%) | -0.32 | -0.38 / -0.41 / -0.19 | [-1.00, +0.00] |
| ood greedy (n=200) | 1,2,3 | +0.012 (tr 14.0% / 15.5% / 15.0% vs 12.0% / 11.0% / 13.5%) | +0.042 (tr 19.5% / 45.0% / 40.0% vs 12.0% / 11.0% / 13.5%) | +0.28 | +0.25 / +1.00 / +0.14 | unbounded: the denominator's bootstrap interval includes 0 (6.4% of resamples ≤ 0); percentile values [-2.46, +3.00] are not interpretable |

- Falsifier 'RFT-Curated worse than RFT-Mixed': the numerator's sign on test is **negative** (-0.027; on ood +0.012); its CI and the §10 evaluation are in the `H2_num` table below. Whether that amounts to 'worse' is part of the verdict.

**RFT-Curated − RFT-Mixed** (`H2_num`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | -0.033 / -0.030 / -0.017 | -0.027 ± 0.009 [-0.052, +0.000] | 0.010 | 2.7% / 3.3% / 3.3% vs 2.7% / 1.7% / 1.7% | +1.1 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.020 / +0.010 / +0.005 | +0.012 ± 0.008 [-0.025, +0.053] | 0.015 | 14.0% / 15.5% / 15.0% vs 12.0% / 11.0% / 13.5% | +2.7 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.021: **yes**; (b) same sign on ood: **no** → criterion **NOT MET** (mechanical evaluation, not a verdict)
- no run in this contrast exceeds 5 % truncation on test_300 greedy

**GRPO-Curated − RFT-Mixed** (`H2_den`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.087 / +0.073 / +0.087 | +0.082 ± 0.008 [+0.040, +0.126] | 0.014 | 5.3% / 9.3% / 8.0% vs 2.7% / 1.7% / 1.7% | +5.6 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.080 / +0.010 / +0.035 | +0.042 ± 0.035 [-0.012, +0.095] | 0.029 | 19.5% / 45.0% / 40.0% vs 12.0% / 11.0% / 13.5% | +22.7 pp | no |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.028: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

The denominator is an RFT-vs-GRPO contrast and all three GRPO-Curated seeds carry the §7 flag: GRPO-Curated's accuracy is the looser lower bound of the two, so the denominator — and with it the ratio — is sensitive to how truncated completions are treated. Not estimated here.

**Verdict (Laksh):** ______________________________________________

## H3 — on-policy / negative-feedback effect (**the headline comparison**)

> SPEC §1: GRPO-Curated outperforms RFT-Curated by more than run-to-run variability despite identical prompt and sampling budgets.
>
> Falsifier (PREREGISTRATION §1): the difference on test_300 is within 2 × the pooled seed std, or it flips sign on ood_hard_200, or RFT-Curated ≥ GRPO-Curated.

**GRPO-Curated − RFT-Curated** (`H3`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1,2,3 | +0.120 / +0.103 / +0.103 | +0.109 ± 0.010 [+0.067, +0.151] | 0.014 | 5.3% / 9.3% / 8.0% vs 2.7% / 3.3% / 3.3% | +4.4 pp | no |
| ood greedy (n=200) | 1,2,3 | +0.060 / +0.000 / +0.030 | +0.030 ± 0.030 [-0.025, +0.085] | 0.030 | 19.5% / 45.0% / 40.0% vs 14.0% / 15.5% / 15.0% | +20.0 pp | yes |

- SPEC §10 criterion: (a) |Δ test| > 2 × pooled seed std = 0.028: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- ⚑ SPEC §7: GRPO-Curated seeds 1,2,3 exceed 5 % truncation on test_300 greedy — **not a headline number as it stands**

Stated next to H3 as required: **RFT tried 9 configurations per arm (best on val, n = 100, seed 1) and GRPO ran one fixed recipe** (PREREGISTRATION §5.2); the asymmetry tilts toward RFT. **Every GRPO-Curated seed exceeds 5 % truncation on test_300 greedy and no RFT-Curated seed does**, so under SPEC §7 this Δ is not a headline number as it stands; GRPO-Curated's accuracy is the looser lower bound of the two. Same prompts (train_curated, 73) and the same 19,200-completion generation budget on both sides; consumed completions, training tokens and optimizer steps differ by design (see `how_could_this_be_wrong.md`, H3).

**Verdict (Laksh):** ______________________________________________

## Pre-registered predictions (PREREGISTRATION §3, written before training)

### P1 — 'RFT-Mixed > RFT-Easy on test_300 (same direction on ood_hard_200), gap > seed noise'

- Δ test = +0.037 (seeds 1–3, n = 300, CI [+0.002, +0.070], truncation 2.7% / 1.7% / 1.7% vs 2.3% / 3.3% / 1.7%): sign **as predicted**.
- Δ ood = +0.018 (n = 200, CI [-0.020, +0.057], truncation 12.0% / 11.0% / 13.5% vs 14.0% / 9.5% / 8.0%).
- 'gap > seed noise' read as SPEC §10: (a) |Δ test| > 2 × pooled seed std = 0.021: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)

**Verdict (Laksh):** ______________________________________________

### P2 — 'GRPO-Curated > RFT-Curated on test_300 with the same sign on ood, but the residual is modest; most of GRPO's edge vs RFT-Mixed is recovered by curation'

- Δ test = +0.109 (seeds 1–3, n = 300, CI [+0.067, +0.151], truncation 5.3% / 9.3% / 8.0% vs 2.7% / 3.3% / 3.3% ⚑): sign **as predicted**; Δ ood = +0.030 (n = 200, CI [-0.025, +0.085], truncation 19.5% / 45.0% / 40.0% vs 14.0% / 15.5% / 15.0%).
- SPEC §10 on this contrast: (a) |Δ test| > 2 × pooled seed std = 0.028: **yes**; (b) same sign on ood: **yes** → criterion **MET** (mechanical evaluation, not a verdict)
- 'most of GRPO's edge … recovered by curation' is the H2 ratio: -0.32 on test (per seed -0.38 / -0.41 / -0.19), +0.28 on ood. 'Most' and 'modest' have no pre-registered number; if 'most' is read as a ratio > 0.5 the condition is **not met** on test — that reading is the agent's, not the protocol's.

**Verdict (Laksh):** ______________________________________________

## Controls (SPEC §8): C1 random reward and C2 format-only vs Base

**C1 random reward − Base** (`C1`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1 | -0.197 | -0.197 ± n/a (1 seed) [-0.253, -0.143] | n/a (1 seed) | 4.7% vs 4.3% | +0.3 pp | n/a (1 seed) |
| ood greedy (n=200) | 1 | -0.030 | -0.030 ± n/a (1 seed) [-0.090, +0.030] | n/a (1 seed) | 7.5% vs 13.0% | -5.5 pp | n/a (1 seed) |

- SPEC §10 criterion: (a) **not evaluable**: a pooled seed std needs ≥ 2 seeds in both arms (this contrast has one); (b) same sign on ood: **yes**
- no run in this contrast exceeds 5 % truncation on test_300 greedy

**C2 format only − Base** (`C2`)

| metric | seeds | Δ per seed (seed i − seed i) | mean Δ ± seed std of Δ [paired bootstrap 95 % CI] | pooled seed std | truncation A vs B (per seed) | truncation gap A − B | seed ranges overlap |
|---|---|---|---|---|---|---|---|
| test greedy (n=300) | 1 | -0.037 | -0.037 ± n/a (1 seed) [-0.083, +0.010] | n/a (1 seed) | 1.0% vs 4.3% | -3.3 pp | n/a (1 seed) |
| ood greedy (n=200) | 1 | +0.020 | +0.020 ± n/a (1 seed) [-0.045, +0.085] | n/a (1 seed) | 10.5% vs 13.0% | -2.5 pp | n/a (1 seed) |

- SPEC §10 criterion: (a) **not evaluable**: a pooled seed std needs ≥ 2 seeds in both arms (this contrast has one); (b) same sign on ood: **no**
- no run in this contrast exceeds 5 % truncation on test_300 greedy

One training seed per control and one base evaluation: Δ and its paired problem-bootstrap CI are reported, the seed-std criterion is not evaluable. For scale, the seed std of GRPO-Mixed (same prompts, same recipe) on test greedy is 0.005.

**Verdict (Laksh):** ______________________________________________
## 12. Anomalies, deviations and workarounds noticed while producing this packet (rows 1–2 computed from the run root; the rest from `configs/analysis/packet_notes.yaml`)
| # | item | recorded in |
|---|---|---|
| 1 | NaN / inf in loss, grad_norm, reward_mean: 0. Lowest 10-step mean train reward per GRPO run: grpo_easy_s1 0.702; grpo_easy_s2 0.714; grpo_easy_s3 0.700; grpo_mixed_s1 0.380; grpo_mixed_s2 0.361; grpo_mixed_s3 0.378; grpo_curated_s1 0.469; grpo_curated_s2 0.447; grpo_curated_s3 0.447; grpo_random_reward_s1 0.463; grpo_format_only_s1 0.772. Mean training completion length, steps 1–10 → 291–300, tokens (truncation % of training completions in the last window): grpo_easy_s1 465 → 844 (0.0 %); grpo_easy_s2 401 → 605 (0.0 %); grpo_easy_s3 437 → 709 (0.0 %); grpo_mixed_s1 520 → 1250 (2.2 %); grpo_mixed_s2 559 → 1459 (3.1 %); grpo_mixed_s3 506 → 1603 (2.5 %); grpo_curated_s1 484 → 1022 (0.5 %); grpo_curated_s2 550 → 1544 (3.1 %); grpo_curated_s3 540 → 1235 (0.9 %); grpo_random_reward_s1 515 → 430 (0.2 %); grpo_format_only_s1 507 → 547 (0.0 %) | train_log.jsonl; section 9 |
| 2 | Seeds with truncation > 5 % on test_300 greedy (SPEC §7 flag; no correction applied anywhere): GRPO-Mixed seeds 1,2,3; GRPO-Curated seeds 1,2,3; highest ood_hard_200 greedy truncation 54.5 % (grpo_mixed/seed2). Lowest answer-line rates on test_300 greedy: grpo_easy/seed3 81.0 %; grpo_mixed/seed2 82.0 %; grpo_mixed/seed3 88.3 % (base 94.0 %). Eval units recorded with git_dirty = true: 32 at SHAs 52b4b42, ffb604e | sections 4 and 10; sanity.md; tables/units.csv |
| 3 | Scope. The packet request names 7 trained arms; SPEC §8 defines 6 (RFT/GRPO × Easy/Mixed/Curated) plus C1 and C2, and 6 + 2 exist; seeds 4–5 were not run. Runs are read from runs/store_mirror/runs (read-only mirror of the artifact store, SHA-256 verified against the store on 2026-09-19), not from runs/eval, runs/rft, runs/grpo; the local runs/grpo/grpo_mixed_s1 is a dry-run stub and is not read. | SPEC.md §8–9; results/phase4/README.md; notebook 2026-09-19 |
| 4 | Crashes and re-runs (none resumed from a checkpoint; section 2 flags them). grpo_mixed_s1: a first attempt OOMed in the step-1 backward (micro-batch 8×8 → 4×16, used by every GRPO run); a second attempt collapsed at step ~150 (TRL merged the LoRA into bf16 weights for vLLM) and is kept under grpo/_failed, never read; the result-bearing run started from step 0 with native vLLM LoRA. grpo_curated_s1: OOM in the step-5 backward, relaunched from step 0 with expandable_segments; the first attempt's logs are the *.stale-* files, never read. Also never read: RFT runs trained before the EOS amendment (rft_noeos_ablation) and the RFT selection evaluated through a bf16 merge (*_bf16merge_stale). | notebook 2026-09-13, 2026-09-14, 2026-09-16, 2026-09-17; PREREGISTRATION §4 |
| 5 | Protocol files vs runs. configs/locked/prompt.yaml records extraction_rule v1.5; every unit's resolved config records v1.6 (SPEC §5 v1.6, core/verify.py); cap.yaml was computed under v1.5 (PREREGISTRATION §3: v1.6 would give 4096; 4352 kept). The evaluated GRPO adapter (adapter/step_300) is not in the mirror; adapter hashes are of adapter/final, whose equality with step_300 follows from train/grpo_trl.py, not from a byte comparison. | configs/locked/prompt.yaml; SPEC.md §12 (v1.6); PREREGISTRATION §3; sanity.md 'final checkpoints only' |
| 6 | Gaps and bookkeeping. Base gsm8k_500 mean@8 was never evaluated ('missing'); google/gemma-4-E4B-it has stub-sampler units only and is excluded; reference models have no gsm8k units. RFT-Curated's budgets.json records 19,200 completions available (the whole train_mixed_100 draw); section 7 reports the completions that belong to its prompts. meta.json cost fields use $4.29/h, Lambda billed $3.29/h; dollar figures here use the billed rate. | sanity.md 'units present'; tables/results.md §6; results/phase2/README.md, results/phase3/README.md |
