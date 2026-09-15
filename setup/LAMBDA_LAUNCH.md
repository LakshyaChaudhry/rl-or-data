# Lambda Cloud launch notes (replaces `AWS_LAUNCH.md`; decision 2026-09-07)

Target: **1× H100 80 GB on-demand** (`gpu_1x_h100_sxm5`, $4.29/h list on 2026-09-07; the PCIe
variant `gpu_1x_h100_pcie` is $3.29/h and fine for everything in this project). One GPU per run.

Two Lambda facts drive everything below:

1. **A shut-down instance keeps billing.** `sudo shutdown -h now` puts the instance in *Alert*
   status and the meter keeps running. Only **terminate** (console or API) stops billing.
   `setup/idle_shutdown.sh` therefore terminates through the API; give it `LAMBDA_API_KEY`.
2. **Local disk is ephemeral** and a persistent filesystem **cannot be attached after launch**
   and must be in the **same region** as the instance. Create it first, attach at launch.

## 0. Before launching (once)

1. Create an API key: cloud.lambda.ai → *API keys*. Put it in `.env` as `LAMBDA_API_KEY=...`.
2. Find a region with 1×H100 capacity *right now*:
   ```
   curl -s -H "Authorization: Bearer $LAMBDA_API_KEY" https://cloud.lambda.ai/api/v1/instance-types \
     | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]
   for k,v in d.items():
       if k.startswith("gpu_1x_h100"): print(k, v["instance_type"]["price_cents_per_hour"]/100, [r["name"] for r in v["regions_with_capacity_available"]])'
   ```
3. Create a **persistent filesystem in that region** (console → *Filesystems* → Create; e.g. name
   `rlordata`). Filesystems bill per GiB-month even when unmounted; ~200 GB is plenty
   (HF cache for five models ≈ 60 GB + runs).

## 1. Launch

Console → *Launch instance* → type `1x H100 (80 GB SXM5)` (or PCIe) → the region from step 0 →
**attach the filesystem** (mounted at `/lambda/nfs/<name>`) → your SSH key → Launch.

Or via the API (`instance_type_name`, `region_name`, `ssh_key_names`, `file_system_names`):
```
curl -s -X POST -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" \
  -d '{"region_name":"<region>","instance_type_name":"gpu_1x_h100_sxm5","ssh_key_names":["<key>"],"file_system_names":["rlordata"],"name":"rlordata-h100"}' \
  https://cloud.lambda.ai/api/v1/instance-operations/launch
```
Billing starts when the instance passes health checks.

## 2. First boot

```
ssh ubuntu@<ip>
git clone <repo> && cd rl-or-data
cp .env.example .env && nano .env
#   HF_TOKEN=...                      (Llama-3.1-8B-Instruct is gated)
#   LAMBDA_API_KEY=...                (lets the idle guard terminate)
#   RLORDATA_ARTIFACTS=/lambda/nfs/rlordata/rlordata-artifacts     (or s3://bucket/prefix)
#   RLORDATA_GPU_RATE_USD_PER_HOUR=4.29
bash setup/setup_gpu.sh              # env, HF_HOME on the filesystem, make test, weights, idle guard + API check
```
`setup_gpu.sh` ends with `bash setup/idle_shutdown.sh test-api`, which must print
`this instance: <id> (terminate wiring OK)`. If it does not, fix it before leaving the box unattended.

## 3. Run (tasks/02, in order — the cap must exist before anything else samples)

`setup_gpu.sh` already verified the committed pool (`make gen-check`) and pre-downloaded the five
models. Cheap smoke test of the real sampler before the 4B run (Qwen3-0.6B-Base, ~2 min):
```
uv run pytest -m gpu -q
```
Then:
```
make cap-run        # provisional cap run: base, T=1, n=8, cap 4096, val_candidates
make cap            # -> configs/locked/cap.yaml (refuses to overwrite)
git add configs/locked/cap.yaml && git commit -m "Phase 1: lock token cap" && git push
make tier           # pass@8 tiering -> data/splits/, data/samples/tiering_pass8.jsonl (+ ood post hoc)
make sanity
make eval-base      # base + 4 reference models × {val, test, ood} × {greedy, mean@8} + pass@k n=64
make transfer-pick  # base greedy on RG basic_arithmetic / count_primes (300) + GSM8K-500
make sync           # everything under runs/, data/splits, data/samples, configs/locked -> RLORDATA_ARTIFACTS
```
Every eval/tier run syncs its own directory at exit; `make sync` is the belt to that suspenders.
After `make tier`, commit `data/splits/` (the frozen protocol splits) the same way as `cap.yaml`.

Resume: finished units (`samples.jsonl` present) are skipped, so re-running `make eval-base` after a
crash only does the missing work; `test_300` is never re-sampled without `--force`. If vLLM does not
free the GPU cleanly between models, run one model per process:
```
uv run rlordata eval --config configs/eval/base.yaml --models google/gemma-4-E4B-it
```

## 3b. Relaunching the GRPO box (one command)

Two GRPO boxes were lost on 2026-09-15 to the idle guard, with nothing synced (the guard's sync ran
`uv` with root's HOME; fixed). Relaunch is now one command on a fresh instance with the filesystem
attached — no manual keys, no manual restore, no manual queueing:

```
git clone https://github.com/LakshyaChaudhry/rl-or-data.git && cd rl-or-data && bash setup/grpo_box.sh
```

`setup/grpo_box.sh` copies `.env` from `<filesystem>/bootstrap/.env` (and Claude Code credentials from
`<filesystem>/bootstrap/claude/` if you keep them there), builds the env, restores finished GRPO runs
(without checkpoints) and unfinished ones (whole, so the queue resumes), restores the base-model evals
the adapter-eval sanity gate needs, prints the queue and a cost estimate, runs `setup_gpu.sh` (guard
armed last), and launches the queue seconds later with its log, a checkpoint sync loop and a GPU-idle
evidence logger all writing to the store's `logs/`. `--dry-run` stops after the preview.

## 4. Leaving the box

- Nothing queued? **Terminate** (console: select → *Terminate* → type `erase data on instance`), or
  ```
  curl -s -X POST -H "Authorization: Bearer $LAMBDA_API_KEY" -H "Content-Type: application/json" \
    -d '{"instance_ids":["<id>"]}' https://cloud.lambda.ai/api/v1/instance-operations/terminate
  ```
- The idle guard does the same after 30 min at <5 % GPU util (after `make sync`). Never rely on it
  instead of terminating yourself; it is the backstop.
- Do **not** `shutdown -h` — it keeps billing.
- The filesystem survives termination and keeps billing per GiB until you delete it.

## 5. Cost guardrails

- Idle guard installed by `setup_gpu.sh`; no uninstall, no hold file. Never disable it.
- Every eval/tier/training script prints an estimated (start) and actual (end) GPU-hour cost from
  `RLORDATA_GPU_RATE_USD_PER_HOUR`.
- Budget ceiling for the project: $3,000. Track spend weekly in the lab notebook.
