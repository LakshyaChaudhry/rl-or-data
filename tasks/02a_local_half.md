# tasks/02a — tasks/02, local half (everything that can be built and tested without a GPU)

Owner: agent. Parent task: `tasks/02_sampler_and_base_eval.md` (read it first; this file scopes its
CPU-side work and adds the Lambda Cloud changes). Read `CLAUDE.md` and `SPEC.md` (v1.2) before starting.
SPEC wins every disagreement. Do not touch `src/rlordata/core/**` or `configs/locked/**`.

## Where the repo is (2026-09-07)

- **Generator done** (tasks/01, 01b). `make gen` → `data/pool/pool.jsonl` (6000 problems, 8 cells of
  750 = {S,M} × {2,3,4,5} steps, seed 20260901, config hash `4af3d8eac9b1`), `make gen-ood` →
  `data/pool/ood_hard_200.jsonl`. Both gitignored; regenerate with make (deterministic, < 1 s).
  Provenance sidecar `<pool>.meta.json`. `scripts/sample_pool.py` shows how to read pools.
- **Core functions implemented by Laksh** (use them, never edit them): `core/verify.py`
  (`extract_answer`, `verify`, `verify_batch`, `Verdict`), `core/logprobs.py`, `core/evaluate.py`
  (`pass_at_k`, `bootstrap_ci`, `compute_metrics` → `Metrics`). `core/grpo.py`, `core/sft_loss.py`,
  `core/rft_select.py` are still stubs and are not needed here.
- `data/tiers.py`: `tier_from_pass8`, `build_splits` (tested) and a `tier` CLI whose live pass@8 path
  is a placeholder that exits, waiting for the sampler.
- `types.py`: `Problem` and `Sample` (fields: run_id, config_hash, seed, arm, data_condition,
  problem_id, tier, prompt, completion, extracted_answer, correct, reward, n_tokens, truncated,
  extraction_failed, extra). Agent may add optional fields, not rename or remove.
- `sampling/prompts.py`: locked `TEMPLATE`; `format_prompt(problem, "base")` works; `"instruct"`
  raises NotImplementedError. `sampling/vllm_sampler.py` is a docstring-only stub with the contract.
- `configs/eval/base.yaml`, `configs/data/tiering.yaml` define the runs. `configs/locked/README.md`
  says how `cap.yaml` comes into existence (only via `scripts/compute_cap.py`).
- `analysis/sanity.py` and `scripts/compute_cap.py` are stubs.
- `make test` → 29 passed, 3 skipped (Laksh's GRPO stubs). `make lint` clean.

## Compute decision: Lambda Cloud, 1× H100 80 GB (replaces the AWS plan in SPEC §11 / setup/AWS_LAUNCH.md)

Consequences you must implement:
1. **Idle shutdown must terminate, not shut down.** On Lambda a `shutdown -h` instance keeps billing.
   Rewrite `setup/idle_shutdown.sh` so that when `LAMBDA_API_KEY` is set it finds its own instance
   id (match the public IP against `GET https://cloud.lambdalabs.com/api/v1/instances`) and calls
   `POST .../api/v1/instance-operations/terminate`; verify the endpoints and auth header against
   docs.lambda.ai before wiring. Sync artifacts (item 7 below) before terminating. Without the key,
   fall back to `shutdown -h` and print a loud warning that Lambda keeps billing. Keep the 30-min /
   <5 % util rule. Never make it easy to disable.
2. **Local disk is ephemeral.** Add an artifact-store abstraction (item 7). HF cache goes on the
   persistent filesystem when one is mounted (`/lambda/nfs/<name>`); update `setup/setup_gpu.sh`.
3. Write `setup/LAMBDA_LAUNCH.md` (region must have 1×H100 on-demand; create the persistent
   filesystem in that region first; API key → `.env`; first-boot commands). Add `LAMBDA_API_KEY` and
   `RLORDATA_ARTIFACTS` to `.env.example`.

## Models (ids verified on the Hub 2026-09-07)

| id | kind | notes |
|---|---|---|
| `Qwen/Qwen3-4B-Base` | base | policy; plain TEMPLATE, no chat template |
| `Qwen/Qwen3-4B` | instruct | has a thinking mode → `enable_thinking=False` in `apply_chat_template` |
| `Qwen/Qwen2.5-7B-Instruct` | instruct | no thinking mode |
| `meta-llama/Llama-3.1-8B-Instruct` | instruct | gated; Laksh has access; needs `HF_TOKEN` |
| `google/gemma-4-E4B-it` | instruct | `Gemma4ForConditionalGeneration` (multimodal), has a thinking mode → disable per its chat-template docs; text-only use. Check vLLM supports the architecture; if not, report, do not hack around it |

Record the exact chat-template kwargs used per model in the run's resolved config.

## Build (all unit-tested here, against a stub sampler)

1. **`sampling/vllm_sampler.py` — `VLLMSampler`** per the contract in its docstring.
   `__init__(model_id, model_kind, lora_path=None, max_completion_tokens, seed, dtype="bfloat16",
   gpu_memory_utilization=0.9, allow_provisional_cap=False)`. Cap rule: if
   `configs/locked/cap.yaml` exists, `max_completion_tokens` must equal its value or construction
   raises; if it does not exist, construction is allowed only with `allow_provisional_cap=True`
   (the provisional cap run uses 4096). `sample(prompts, n, temperature, top_p=1.0) ->
   list[list[Completion]]`, one vLLM call for all prompts × n, `Completion(text, n_tokens,
   truncated)` with `truncated = finish_reason == "length"`. Import `vllm` lazily inside `__init__`
   so the module imports on a Mac. Deterministic given seed. Max prompt tokens 1024 (SPEC §7).
   Add `sampling/stub_sampler.py` with the same interface returning scripted completions (a
   configurable fraction correct, some truncated, some unparseable) for tests and dry runs.
2. **`sampling/prompts.py`** — instruct wrapping: `tokenizer.apply_chat_template([{"role": "user",
   "content": TEMPLATE.format(problem_text=...)}], tokenize=False, add_generation_prompt=True,
   **thinking_off_kwargs)`. Do not change `TEMPLATE`. Unit-test with a fake tokenizer object; a
   test that loads the real Qwen3-4B tokenizer is fine if marked `slow`.
3. **`scripts/compute_cap.py`** — SPEC §7 *(superseded by `tasks/02b_cap_v1.3.md`: cap = max(2048, per-cell 1.25×p99), prompt limit 4096; the 512 floor is gone)*: read the provisional run's `samples.jsonl` (T=1.0, n=8,
   cap 4096, on `val_candidates`), keep correct completions, p99 of `n_tokens` × 1.25, ceil to a
   multiple of 256, floor 512 → write `configs/locked/cap.yaml` with provenance
   (`max_completion_tokens`, `p99_correct_len`, `n_correct_used`, `computed_on`, `config_hash`,
   `run_id`, `git_sha`). Refuse to overwrite an existing cap.yaml. Print the length histogram
   (correct vs incorrect) and the fraction of correct completions that would be truncated at the
   cap (must be < 1 %). Test end-to-end into `tmp_path` with synthetic samples.
4. **`val_candidates`** — 500 random pool problems, seed 1, defined once in code and reused
   by the cap run. Deterministic; test it.
5. **`rlordata eval` CLI** (agent-owned module, e.g. `sampling/eval_runner.py`, dispatched from
   `cli.py`) per `configs/eval/base.yaml`: for each model × split × decoding, format prompts,
   sample, verify with `core.verify`, write `runs/eval/<model>/<split>/<decoding>/samples.jsonl`
   as `types.Sample` records (all CLAUDE.md fields), and `metrics.json` from
   `core.evaluate.compute_metrics` plus pass@k (mean over problems of `core.evaluate.pass_at_k`
   for k in {1,2,4,8,16,32,64}) on the n=64 subset (first 100 of test_300 by index). Every run
   directory gets: resolved config YAML, config hash, git SHA, package versions, wall-clock, GPU
   type, `NOTES.md` stub. Print a GPU-hour × `RLORDATA_GPU_RATE_USD_PER_HOUR` cost estimate at
   start and end. `test_300` is evaluated once per model: refuse to re-run if its samples exist
   unless `--force`. `--sampler stub` runs the whole pipeline locally on ~20 problems and must
   produce every file; that is the acceptance test. Print the one-table summary
   (model × split × {greedy, mean@8, pass@8, trunc %}).
6. **`rlordata tier` live path** in `data/tiers.py`: k=8, T=1.0, cap from `cap.yaml`, base
   template. Store all 8 completions per problem in generation order in
   `data/samples/tiering_pass8.jsonl` (needed later for `train_curated` and RFT), write the
   tiered pool and `data/splits/*.jsonl`, print tier counts and the pass8 histogram. Test with
   the stub.
7. **`src/rlordata/artifacts.py`** — `sync_run_dir(run_dir)` to `RLORDATA_ARTIFACTS`, which is
   either `s3://bucket/prefix` (boto3) or a local directory (Lambda persistent filesystem).
   Called at exit by eval/tier and by the idle-terminate script. Test the local path.
8. **`analysis/sanity.py`** — implement and test: split disjointness by `problem_id` and
   `structure_id` across train/val/test/ood; identical cap and prompt template across the
   resolved configs of runs being compared; truncation > 5 % and extraction-failure flags.
9. **Transfer sets (tasks/02 §5, code only)**: loaders for the three Reasoning Gym candidates
   (300 instances each, fixed seed; `transfer` extra) and GSM8K test first 500 by index, rendered
   with the same TEMPLATE and scored with the same regex. Tests skip cleanly if the optional
   deps are missing; list them.
10. **Makefile**: `cap`, `tier`, `eval-dry` (stub sampler) targets; keep `eval-base`.

## Acceptance

- `make test` and `make lint` clean. Every test that needs vllm or a GPU is marked `gpu` and
  every skipped test is listed by name and reason in the report.
- `make eval-dry` produces the full run-directory layout with the stub sampler.
- `scripts/compute_cap.py` tested into tmp_path; refuses to overwrite.
- Report ends with: files changed (one line each), design choices the task did not specify, and
  the exact command sequence for the GPU half on the box (provisional cap run → compute_cap →
  tier → base + reference evals → transfer-task pick) with an estimated GPU-hour count.

## Do not

- Run anything on a GPU or download model weights. Do not create `configs/locked/cap.yaml`
  with real numbers; it is written only by the script on the box.
- Change `TEMPLATE`, the answer regex, the cap logic, or the decoding settings; enable thinking;
  use `model.generate()` for anything that could be a number.
- Commit `data/`, `runs/`, or `.env`. Do not start tasks/03. Work in small commits on main.
