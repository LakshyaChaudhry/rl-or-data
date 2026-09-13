# tasks/03 — RFT arms: sample → verify → select → SFT → evaluate

Owner: agent. Files: `src/rlordata/train/rft.py`, `src/rlordata/train/common.py` (run dirs, config hashing, cost printout, S3 sync — shared with tasks/04), `src/rlordata/sampling/draw.py`, `scripts/rft_sweep.py`, `analysis/sanity.py` additions, CLI `rft`. Do not touch `core/**` or `configs/locked/**`.

**Prerequisites (do not start a result-bearing run before all four exist):**
1. `tasks/02` complete: `configs/locked/cap.yaml`, `data/splits/*.jsonl` with `pass8` and `tier` on every problem, base evals in `runs/eval/`.
2. `core/logprobs.py`, `core/sft_loss.py`, `core/rft_select.py` implemented by Laksh with their tests passing. Build the trainer and its tests against the stubs now; wire the real functions when they land. Never substitute a library loss or your own selection logic "temporarily."
3. `train_curated` persisted by `tiers.py` (tasks/01b) or by this task's step 2 — one source of truth, written once.
4. Pre-registrd (Laksh). If it isn't, stop and say so.

## What this task produces

Nine result-bearing runs — `{rft_easy, rft_mixed, rft_curated}` × seeds `{1,2,3}` — each with a resolved config, config hash, adapter, val/test/ood metrics with CIs, truncation and extraction-failure rates, and the three budgets (SPEC §8). Plus the SFT sweep that chose each arm's hyperparameters, fully logged.

## Steps

### 1. The 192-sample draw (`sampling/draw.py`)
- For each prompt in `train_easy_100` and `train_mixed_100`, the base model's samples are: the **8 tiering samples already drawn in tasks/02** (generation order preserved) **plus 184 new samples** at T=1.0, top_p=1.0, locked cap, seed 1 → **exactly 192 per prompt**, matching SPEC §8's 19,200 budget. The 8 tiering samples come first so `core.rft_select`'s "first 8 in generation order" is the same 8 that defined tiers. Document this in the run notes.
- Draw once. Write `data/samples/base_{split}_k192_seed1.jsonl` (one `Sample` record per completion, with `sample_indexn `extra`). All RFT arms and seeds read from these files; no arm re-samples.
- Score every completion with `core.verify`. Print per-split: mean pass rate, pass8 histogram, truncation %, extraction-failure %.

### 2. Selection (calls `core.rft_select`, never reimplements it)
- `rft_easy`: `rft_select(train_easy_100, samples, mode="all")`.
- `rft_mixed`: `rft_select(train_mixed_100, samples, mode="all")`.
- `rft_curated`: `rft_select(train_mixed_100, samples, mode="curated")`. Persist the selected `problem_id`s to `data/splits/train_curated.jsonl` if tasks/01b did not already; if it did, assert the two sets are identical and stop if they are not.
- For each arm print: #prompts used, #correct completions before/after dedup, per-tier composition. Write these into the run's `budgets.json` as `prompts`, `completions_available` (192 × #prompts in the parent split), `completions_consumed` (# used for SFT).

### 3. SFT trainer (`train/rft.py`)
- PEFT LoRA from `configs/locked/training.yaml` (r=64, α=16, dropout 0,ll seven projections). bf16. AdamW, wd 0, grad clip 1.0, cosine with 10% warmup. Batch 16 (use gradient accumulation if memory requires; log the micro-batch).
- Loss: `core.sft_loss(core.completion_logprobs(...), completion_mask)`. Prompt tokens masked. The prompt is `sampling.prompts.format_prompt(problem, "base")` — identical bytes to what vLLM saw.
- Tokenization: the completion is appended to the prompt with no extra whitespace, followed by the base model's EOS token id (the token vLLM stopped on; its returned text omits it), and that EOS is inside the loss mask; assert that re-decoding the concatenation reproduces `prompt + completion + eos_token` exactly, assert the EOS is a `generation_config` stop token, and assert the trainer tokenizer hash equals the vLLM tokenizer hash (record both). *(Amended 2026-09-13, authorized by Laksh: the original "no EOS" rule meant the SFT model was never trained to stop; val truncation rose 1% → 36% with training in the first mixed sweep, kept as the no-EOS ablation. GRPO already trains on EOS via TRL.)*
- Seeds control: data order, LoRA init, dropout (none), and any sampling in the trainer. Set `torch`, `numpy`, `random`.
- Log every 10 steps to `train_log.jsonl`: step, loss, lr, tokens seen, wall-clock. Record `optimizer_steps` and `training_tokens` in `budgets.json` at the end.
- Checkpoint: adapter only, at the end of each epoch; final = last epoch (no early stopping).

### 4. Hyperparameter sweep (`scripts/rft_sweep.py`) — the "deflated SFT baseline" guard
- Per arm, seed 1 only: lr ∈ {1e-5, 5e-5, 1e-4} × epochs ∈ {2, 4, 8} = 9 short runs. Selection metric: greedy accuracy on `val_mixed_100` **only**. Ties → fewer epochs.
- Write `runs/rft/<arm>/sweep.json` with every config's val accuracy, CI, truncation %, optimizer steps. The chosen config is then used for seeds 2 and 3 unchanged.
- Nothing in this script may read `test_300` or `ood_hard_200`. Add a test that greps the sweep code for those split names and fails if found.

### 5. Evaluation (single generation path)
- Merge the adapter into the base in bf16, save to a temp dir, evaluate with `sampling.vllm_sampler` at the locked cap on `val_mixed_100` (sweep + final), `test_300` and `ood_hard_200` (final checkpoint only, **once**), plus the secondary transfer sets from `configs/locked/transfer.yaml`. Greedy and mean@8; pass@k(n=64) on the test_300 first-100 subset. Metrics via `core.evaluate.compute_metnity (`analysis/sanity.py`, must pass before metrics are written): adapter non-trivial (‖ΔW‖ > 0 and merged-model greedy outputs differ from base on ≥ 10% of val prompts), same cap and template as the base eval, checkpoint is the final one, splits disjoint.
- Delete the merged weights after eval; keep the adapter. Sync run dir to S3.

### 6. Reporting
- One table per arm: seeds as rows; columns val / test / ood greedy (± CI), mean@8, pass@8, trunc %, extraction-fail %, prompts, completions available/consumed, training tokens, optimizer steps, GPU-hours, cost.
- Aggregate: μ ± σ across seeds with every seed shown. No arm-vs-arm claims in this task; that is Phase 4.
- Cost printed at start (estimate) and end (actual) of every run.

## Acceptance
- Nine final runs + three sweeps present in `runs/rft/`, each with resolved config, config hash, git SHA, package versions, `budgets.json`, `metrics.json`, `train_log.jsonl`, `NOTES.md` stub.
- All sanity checks pass; the test-split grep test passes; `make make lint` clean.
- The 192-sample files exist once and are read-only after step 1 (`chmod 444`).
- A report listing: per-arm chosen config, dataset sizes per arm, pass8 histograms, anything you had to decide that this file didn't specify.

## Do not
- Do not choose hyperparameters per seed. Do not tune on test or ood. Do not cap completions per problem unless Laksh sets `max_per_problem` in the arm config (it is `null`).
- Do not train the curated arm on a re-derived selection; it reads the persisted `train_curated` set.
- Do not start `tasks/04`.
