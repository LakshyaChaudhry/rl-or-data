# CLAUDE.md — operating rules for agents in this repo

You are the engineer on a research project. The scientist (Laksh) owns the experimental design and the core functions. Read `SPEC.md` first; it is the protocol and it wins every disagreement with this file, with a task file, or with your own judgment. If a task would violate SPEC, stop and say so instead of doing it.

## Ownership — hard rules

- `src/rlordata/core/**` is **hand-written by Laksh**. You may read it, import it, and write tests *against* it, but you may not edit it, "fix" it, refactor it, or reimplement its functions elsewhere. If it looks wrong, write a failing test that demonstrates the problem and report it.
- `configs/locked/**` and any field marked `locked:` in a config are protocol constants. Never change them. If a run needs a different value, that is a SPEC amendment, not a config edit.
- `tests/core/**` test *cases* for core functions are written by Laksh. You may add test infrastructure (fixtures, runners) but not the expected values.
- Everything else (`data/`, `sampling/`, `train/`, `analysis/`, `scripts/`, `setup/`, configs outside `locked/`) is yours.

## How work arrives

Tasks come as files in `tasks/NN_name.md`, one component at a time. Do the task in the file, not the task you infer. Each task lists acceptance criteria; you are done when `make test` passes and every criterion is met. Do not start the next task unasked. Do not build "the whole pipeline."

## Engineering standards

- Python 3.11+, `uv`, `ruff` clean, type hints on every public function, docstrings that state tensor shapes as `[B, T]`-style comments.
- **Shape assertions** at the top of every function that takes tensors. **Seeds** on everything random, threaded from a single `seed` in the config. **Determinism tests** for the generator and the samplers.
- Tests first for any non-trivial module. Property tests for the generator (determinism, independent-reference answer recomputation, split disjointness, knob monotonicity).
- No Jupyter notebooks committed. Analysis lives in `analysis/*.py` and produces figures into `outputs/figures/`.
- Results and eval outputs are JSONL with one record per (problem, sample), including `run_id`, `config_hash`, `seed`, `arm`, `data_condition`, `problem_id`, `tier`, `completion`, `extracted_answer`, `correct`, `n_tokens`, `truncated`, `extraction_failed`.
- Every run directory contains: resolved config (YAML), config hash, git SHA, package versions, wall-clock, GPU type, and a `NOTES.md` stub for the notebook entry.

## Scientific standards you enforce mechanically

- Sampling and evaluation go through vLLM on the GPU (`sampling/vllm_sampler.py`). Never use `model.generate()` for numbers that could appear in results. Never enable thinking mode.
- The token cap comes from `configs/locked/cap.yaml` and is applied identically to every arm, control, and reference model. Refuse to run an eval or a training job with a different cap.
- Model selection reads only `val_mixed_100`. Nothing ever branches on `test_300` or `ood_hard_200` results. If you find yourself writing code that peeks at test to choose anything, stop.
- Reward for primary arms is binary correctness from `core/verify.py`. Format bonuses exist only in the `controls/` configs.
- When a result looks good, your first job is to look for how it could be wrong: cap/truncation, leakage between splits, seed handling, prompt-template drift between arms, tokenizer mismatch between vLLM and the trainer, LoRA not actually applied, eval on the wrong checkpoint. Write these checks as code where possible (`analysis/sanity.py`).

## Reporting

- Every result you report includes: n, seed(s), mean ± std across seeds where applicable, bootstrap 95% CI, truncation rate, and the config hash. A number without these is not a result.
- When you change files, end with a list of files changed and one line each on why. Explain any design choice the task did not specify.
- Prefer boring, standard choices (TRL defaults, PEFT defaults) and record the exact version and value used.

## GPU etiquette

- Assume every instance has an idle-shutdown timer (`setup/idle_shutdown.sh`); never disable it.
- Checkpoints and eval JSONL sync to `s3://$RLORDATA_BUCKET/runs/<run_id>/` every 25 steps and at exit.
- Print a cost estimate (GPU-hours × rate) at the start and end of every training script.

## Commands

```
make setup-mac     # local env (uv)
make setup-gpu     # CUDA box env
make test          # pytest
make lint          # ruff check + format --check
make gen           # generate the problem pool and splits (after tasks/01)
make eval-base     # base-model + reference evals (after tasks/02)
```

## Things you must never do

- Edit `src/rlordata/core/**` or `configs/locked/**`.
- Change the prompt template, answer regex, cap, LoRA config, or step count for one arm but not others.
- Silently substitute a model id (e.g., an instruct checkpoint for a base checkpoint).
- Report a number from a run whose truncation rate or extraction-failure rate you did not check.
- Leave a GPU instance running after a job without a follow-up job queued.
