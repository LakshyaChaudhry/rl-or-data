# tasks/04 — GRPO arms: TRL GRPOTrainer + colocated vLLM, controls, diagnostics

Owner: agent. Files: `src/rlordata/train/grpo_trl.py`, `src/rlordata/train/rewards.py`, `src/rlordata/train/callbacks.py`, `scripts/run_queue.py` + `queue.yaml`, `analysis/sanity.py` additions, CLI `grpo`. Reuse `train/common.py` from tasks/03 (run dirs, config hash, cost printout, S3 sync). Do not touch `core/**` or `configs/locked/**`.

**Prerequisites:** tasks/02 outputs (cap, splits with pass8/tier, base evals); `data/splits/train_curated.jsonl` persisted; SPEC v1.8 §9 applied. `core/verify.py` implemented (it is the reward). `core/grpo.py` is NOT required for this task — it is required for tasks/04b. This task can run on a separate GPU instance in parallel with tasks/03.

## What this task produces

Eleven runs: `{grpo_easy, grpo_mixed, grpo_curated}` × seeds `{1,2,3}`, plus controls `C1 grpo_random_reward` and `C2 grpo_format_only` (mixed, seed 1). Each with resolved config, config hash, adapter checkpoints every 25 steps, per-step training diagnostics, offline val evals at steps {100, 200, 300}, test/ood once at step 300, budgets, cost.

## Steps

### 1. Dataset for TRL
- Rows: `prompt` = `sampling.prompts.format_prompt(problem, "base")` as **plain text** (not conversational — no chat template may be applied; assert the trainer's rendered prompt bytes equal the RFT-sampling prompt bytes for the same problem_id, hash-compared), plus `problem_id`, `answer`, `tier` columns forwarded to reward functions via kwargs.
- Data conditions: `train_easy_100`, `train_mixed_100`, `train_curated` (read the persisted file; never re-derive).

### 2. GRPOConfig — every value explicit, read from `configs/locked/training.yaml` and `configs/locked/cap.yaml`
```
num_generations=8            per_device_train_batch_size × gradient_accumulation_steps = 64  (8 prompts × 8)
max_steps=300                learning_rate=5e-5   lr_scheduler_type="cosine"  warmup_ratio=0.1
max_grad_norm=1.0            weight_decay=0.0     bf16=True
temperature=1.0  top_p=1.0   max_completion_length=<cap>   max_prompt_length=1024
beta=0.0  epsilon=0.2 (epsilon_high unset → symmetric)  scale_rewards="group"  loss_type="dapo"  num_iterations=1
use_vllm=True  vllm_mode="colocate"  vllm_gpu_memory_utilization=<from arm config>
logging_steps=1  save_steps=25  log_completions=True (sample 2/step)  seed=<seed>
```
PEFT `LoraConfig(r=64, lora_alpha=16, lora_dropout=0.0, target_modules=[q,k,v,o,gate,up,down]_proj, task_type="CAUSAL_LM")`. Reference = adapter-disabled base (TRL handles this with PEFT; assert `ref_model is None` and PEFT is active).
Assert at startup: `max_steps × generation_batch_size == 19200` and `generation_batch_size == 64`. Record the resolved GRPOConfig, TRL, vLLM, PEFT, torch versions, and GPU in the run dir. **If any TRL/vLLM version pair fails the TRL compatibility check in setup, stop and report; do not fall back to `model.generate()`.**

### 3. Reward functions (`train/rewards.py`) — thin wrappers, never reimplementations
- `verify_binary(completions, problem_id, answer, **kw)` → `core.verify` per completion, returns 1.0/0.0. Primary for all GRPO arms.
- `random_bernoulli(completions, problem_id, **kw)` → `Bernoulli(0.5)` from `np.random.default_rng(hash(seed, problem_id, step, i))`; **independent of content**. C1 only.
- `format_only(completions, **kw)` → 1.0 iff `core.verify.extract_answer` returns an int. C2 only.
- Each reward function also records, per completion: `correct` (always computed via `core.verify` even for C1/C2 so the true accuracy curve exists), `n_tokens`, `truncated`, `extraction_failed`, `tier`, `step`.

### 4. Diagnostics callback (`train/callbacks.py`) — logged every step to `train_log.jsonl`
- From TRL: `reward` mean/std, `frac_reward_zero_std` (the fraction of groups with identical rewards = **implicit-filtering rate**; the key H2 diagnostic), `completions/mean_length`, `completions/clipped_ratio` (truncation), `kl` (if present), `clip_ratio` metrics, `grad_norm`, `lr`, `step_time`, `num_tokens`.
- Ours: per-tier mean reward and per-tier zero-std fraction; per-prompt rolling pass count so we can see which prompts are "active" over training; cumulative completions sampled; cumulative training tokens.
- Print a cost estimate at step 1 and actual at the end.

### 5. Evaluation (single generation path; same as tasks/03)
- Offline, after training: merge adapter at steps {100, 200, 300} → vLLM at the locked cap → greedy on `val_mixed_100`. Final (300) only: `test_300`, `ood_hard_200` (once), transfer sets, mean@8, pass@k(n=64) on the test first-100 subset. Metrics via `core.evaluate.compute_metrics`. Delete merged weights; keep adapters.
- Save the training-reward curve and the val-accuracy-at-checkpoints curve side by side in `curves.json` (the "training reward is not held-out accuracy" separation; tasks/05 plots it).

### 6. Sanity (`analysis/sanity.py`; must pass before metrics are written)
- Prompt bytes identical to RFT sampling for the same problem_id; cap equal to `cap.yaml`; tokenizer hash equal between trainer and vLLM; adapter non-trivial and merged-model greedy outputs differ from base on ≥ 10% of val prompts; exactly 19,200 completions were sampled (count the reward records); checkpoint evaluated on test is step 300; C1's training reward is ≈ 0.5 throughout while its `correct` rate is logged separately.

### 7. Queueing (`scripts/run_queue.py`, `queue.yaml`)
- `queue.yaml` lists jobs in order: `{name, cmd, requires: [paths that must exist], produces: [marker path]}`. The runner executes sequentially, skips jobs whose marker exists, stops on first failure, syncs the run dir to S3 after each job, and prints cumulative GPU-hours and cost. Supports `--resume` for GRPO jobs via the latest adapter checkpoint.
- Default queue for this instance: `grpo_mixed s1 → grpo_curated s1 → grpo_easy s1 → C1 → C2 → seeds 2,3 for the three arms → evals`. (Seed 1 of the H3 pair first, so an early look at the headline comparison is possible without waiting for everything.)

## Acceptance
- 11 runs in `runs/grpo/` with the artifacts above; `make test` and `make lint` clean; sanity passes for every run.
- A report with: resolved config; per-run wall-clock and cost; mean training reward and `frac_reward_zero_std` at steps 1/100/200/300 per arm; val curve per arm; anything you decided that this file did not specify.
- No arm-vs-arm claims; that is tasks/05.

## Do not
- Do not use thinking mode, a chat template, or a different cap. Do not add a format or length reward to the primary arms. Do not enable dynamic sampling, clip-higher, or overlong shaping. Do not early-stop or pick a checkpoint by test/ood.
- Do not start tasks/05.
